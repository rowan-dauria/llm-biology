"""Build per-layer top-K activating prompt windows for transcoder features.

For each chosen layer L and feature f, keep the K (prompt_id, token_pos) pairs
with the highest post-ReLU encoder activation across the corpus. Implementation
is a streaming top-K maintained on-device: each pre-hook on the MLP runs the
transcoder encoder, computes the per-feature top-K within the current batch,
and merges with the running top-K. Decoding is never invoked.

Output: one .pt file per layer at ``llm-biology/data/feature_topk/``,
containing topk_vals (K, F), topk_prompt_id (K, F) int32, topk_token_pos
(K, F) int16, plus the corpus_spec / max_seq_len needed to recover windows.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp
from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    load_transcoder,
)
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .corpus import iter_batches, iter_texts
except ImportError:
    from corpus import iter_batches, iter_texts

MODEL_ID = "Qwen/Qwen3-4B"
TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"
NUM_LAYERS = 36
LAYERS_TO_HOOK = [2, NUM_LAYERS // 3, (2 * NUM_LAYERS) // 3, NUM_LAYERS - 3]

CACHE_DIR = os.getenv("HF_HOME")
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "feature_topk"
PROGRESS_LOG_EVERY = 50  # batches between per-worker progress lines


def pick_worker_device(worker_idx: int) -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_idx = worker_idx % num_gpus
        return torch.device(f"cuda:{gpu_idx}"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def _final_topk_path(layer: int) -> Path:
    return OUTPUT_DIR / f"topk_layer_{layer}.pt"


def _worker_topk_path(layer: int, worker_idx: int, workers: int) -> Path:
    return OUTPUT_DIR / f"topk_layer_{layer}_worker_{worker_idx}_of_{workers}.pt"


def _merge_topk(
    vals_list: list[torch.Tensor],
    pid_list: list[torch.Tensor],
    tpos_list: list[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate candidate (vals, pid, tpos) sets and keep the per-feature top-K."""
    all_vals = torch.cat(vals_list, dim=0)
    merged_vals, idx = all_vals.topk(k, dim=0)
    merged_pid = torch.gather(torch.cat(pid_list, dim=0), dim=0, index=idx)
    merged_tpos = torch.gather(torch.cat(tpos_list, dim=0), dim=0, index=idx)
    return merged_vals, merged_pid, merged_tpos


def _save_topk(
    path: Path,
    layer: int,
    vals: torch.Tensor,
    pid: torch.Tensor,
    tpos: torch.Tensor,
    k: int,
    corpus_spec: str,
    max_seq_len: int,
    model_id: str,
    num_parts: int,
) -> None:
    torch.save(
        {
            "layer": layer,
            "topk_vals": vals,
            "topk_prompt_id": pid,
            "topk_token_pos": tpos,
            "K": k,
            "corpus_spec": corpus_spec,
            "max_seq_len": max_seq_len,
            "model_id": model_id,
            # num_parts is needed to decode prompt_id back to corpus text (see corpus.iter_texts).
            "num_parts": num_parts,
        },
        path,
    )


@dataclass
class LayerState:
    vals: torch.Tensor  # (K, F) fp32, running top-K activations
    pid: torch.Tensor  # (K, F) int32, prompt id of each top-K entry
    tpos: torch.Tensor  # (K, F) int16, token position within that prompt


@dataclass
class RunState:
    k: int
    layer_state: dict[int, LayerState] = field(default_factory=dict)
    # Per-batch context, set by the main loop before each forward.
    attention_mask: torch.Tensor | None = None
    prompt_ids: torch.Tensor | None = None


def make_topk_hook(
    layer_idx: int,
    transcoder: SingleLayerTranscoder,
    state: RunState,
):
    """Forward-pre-hook on an MLP module: encode and update running top-K."""

    def pre_hook(_module, args):
        assert state.attention_mask is not None
        assert state.prompt_ids is not None

        with torch.no_grad():
            mlp_input = args[0]
            # Stay in the device dtype (fp16 on MPS) for the big (B, T, F) tensor.
            feats = transcoder.encode(mlp_input)
            mask = state.attention_mask.bool().unsqueeze(-1)
            feats.masked_fill_(~mask, float("-inf"))

            b, t, f = feats.shape
            flat_feats = feats.view(b * t, f)

            batch_vals, batch_pos = flat_feats.topk(state.k, dim=0)  # (K, F) each
            del feats, flat_feats  # release the big tensor before doing more work

            prompt_idx_in_batch = batch_pos // t
            token_pos = (batch_pos % t).to(torch.int16)
            batch_pid = state.prompt_ids[prompt_idx_in_batch].to(torch.int32)

            ls = state.layer_state[layer_idx]
            ls.vals, ls.pid, ls.tpos = _merge_topk(
                [ls.vals, batch_vals.to(torch.float32)],
                [ls.pid, batch_pid],
                [ls.tpos, token_pos],
                state.k,
            )

    return pre_hook


def load_transcoders(device: torch.device, dtype: torch.dtype) -> dict[int, SingleLayerTranscoder]:
    wanted = [f"layer_{layer}.safetensors" for layer in LAYERS_TO_HOOK]
    transcoder_dir = Path(snapshot_download(TRANSCODER_REPO, allow_patterns=wanted))
    out: dict[int, SingleLayerTranscoder] = {}
    for layer in LAYERS_TO_HOOK:
        path = transcoder_dir / f"layer_{layer}.safetensors"
        out[layer] = load_transcoder(
            str(path),
            layer,
            device=device,
            dtype=dtype,
            lazy_encoder=False,
            lazy_decoder=True,
        )
    return out


_PREFETCH_STOP = object()


def prefetch(iterable: Iterable, max_prefetch: int = 2) -> Iterator:
    """Run ``iterable`` in a background thread, buffering up to ``max_prefetch`` items.

    Stream/decompress/tokenise all release the GIL, so this overlaps CPU ingest with
    the GPU forward on the main thread and stops the GPU idling between batches.
    """
    q: queue.Queue = queue.Queue(maxsize=max_prefetch)
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:  # surface producer failures to the consumer
            err.append(exc)
        finally:
            q.put(_PREFETCH_STOP)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is _PREFETCH_STOP:
            break
        yield item
    if err:
        raise err[0]


def worker_fn(worker_idx: int, args: argparse.Namespace) -> None:
    device, dtype = pick_worker_device(worker_idx)
    print(f"[Worker {worker_idx}] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Worker {worker_idx}] loading model")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=CACHE_DIR,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[Worker {worker_idx}] model loaded in {time.time() - t0:.1f}s")

    transcoders = load_transcoders(device, dtype)
    f_dim = transcoders[LAYERS_TO_HOOK[0]].W_enc.shape[0]
    print(f"[Worker {worker_idx}] d_transcoder={f_dim}, layers={LAYERS_TO_HOOK}")

    state = RunState(k=args.k)
    for layer in LAYERS_TO_HOOK:
        state.layer_state[layer] = LayerState(
            vals=torch.full((args.k, f_dim), float("-inf"), dtype=torch.float32, device=device),
            pid=torch.zeros((args.k, f_dim), dtype=torch.int32, device=device),
            tpos=torch.zeros((args.k, f_dim), dtype=torch.int16, device=device),
        )

    handles = []
    for layer in LAYERS_TO_HOOK:
        mlp = model.model.layers[layer].mlp
        h = mlp.register_forward_pre_hook(make_topk_hook(layer, transcoders[layer], state))
        handles.append(h)

    try:
        batches_iter = iter_batches(
            args.corpus_spec,
            tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=device,
            part_idx=worker_idx,
            num_parts=args.workers,
        )
        if args.prefetch > 0:
            batches_iter = prefetch(batches_iter, max_prefetch=args.prefetch)

        n_docs = 0
        t_loop = time.time()
        for i, batch in enumerate(batches_iter):
            state.attention_mask = batch.attention_mask
            state.prompt_ids = batch.prompt_ids
            with torch.no_grad():
                model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )
            if device.type == "mps":
                torch.mps.empty_cache()
            n_docs += int(batch.input_ids.shape[0])
            if (i + 1) % PROGRESS_LOG_EVERY == 0:
                elapsed = time.time() - t_loop
                rate = n_docs / elapsed if elapsed > 0 else 0.0
                print(
                    f"[Worker {worker_idx}] {i + 1} batches, {n_docs} docs, "
                    f"{elapsed:.0f}s elapsed, {rate:.1f} docs/s",
                    flush=True,
                )
        print(
            f"[Worker {worker_idx}] forward loop done: {n_docs} docs in "
            f"{time.time() - t_loop:.0f}s",
            flush=True,
        )
    finally:
        for h in handles:
            h.remove()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for layer, ls in state.layer_state.items():
        if args.workers > 1:
            path = _worker_topk_path(layer, worker_idx, args.workers)
        else:
            path = _final_topk_path(layer)

        _save_topk(
            path,
            layer,
            ls.vals.cpu(),
            ls.pid.cpu(),
            ls.tpos.cpu(),
            args.k,
            args.corpus_spec,
            args.max_seq_len,
            args.model_id,
            args.workers,
        )
        finite = torch.isfinite(ls.vals).sum().item()
        print(f"[Worker {worker_idx}] saved {path}  finite-entries={finite}/{ls.vals.numel()}")


def merge_worker_files(workers: int) -> None:
    print(f"[INFO] merging results from {workers} workers for layers {LAYERS_TO_HOOK}")
    for layer in LAYERS_TO_HOOK:
        worker_files = [_worker_topk_path(layer, i, workers) for i in range(workers)]
        loaded = [torch.load(p, map_location="cpu") for p in worker_files]

        ks = {item["K"] for item in loaded}
        if len(ks) != 1:
            raise ValueError(f"inconsistent K across worker files for layer {layer}: {ks}")
        k = ks.pop()

        merged_vals, merged_pid, merged_tpos = _merge_topk(
            [item["topk_vals"] for item in loaded],
            [item["topk_prompt_id"] for item in loaded],
            [item["topk_token_pos"] for item in loaded],
            k,
        )

        meta = loaded[0]
        final_path = _final_topk_path(layer)
        _save_topk(
            final_path,
            layer,
            merged_vals,
            merged_pid,
            merged_tpos,
            k,
            meta["corpus_spec"],
            meta["max_seq_len"],
            meta["model_id"],
            workers,
        )
        finite = torch.isfinite(merged_vals).sum().item()
        print(f"[MERGE] saved {final_path} (finite-entries={finite}/{merged_vals.numel()})")

        for p in worker_files:
            try:
                p.unlink()
                print(f"[CLEANUP] deleted worker file {p}")
            except OSError as e:
                print(f"[WARNING] could not delete worker file {p}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_spec", default="hf:monology/pile-uncopyrighted:train:1000")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel worker processes to spawn."
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Batches to pre-tokenise on a background thread (0 disables).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.prefetch < 0:
        parser.error("--prefetch must be >= 0")

    # Preflight: fail fast on dataset/network/codec issues before the ~80s model load.
    print(f"[INFO] preflight: opening {args.corpus_spec!r}")
    t0 = time.time()
    sample = next(iter(iter_texts(args.corpus_spec)), None)
    if sample is None:
        raise RuntimeError(f"corpus_spec {args.corpus_spec!r} yielded no texts")
    print(f"[INFO] preflight ok in {time.time() - t0:.1f}s, first doc len={len(sample[1])}")

    if args.workers > 1:
        print(f"[INFO] spawning {args.workers} workers")
        # No-op if the start method was already fixed by an earlier call.
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn")
        mp.spawn(worker_fn, args=(args,), nprocs=args.workers)
        merge_worker_files(args.workers)
    else:
        worker_fn(0, args)


if __name__ == "__main__":
    main()
