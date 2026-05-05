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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    load_relu_transcoder,
)
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .corpus import iter_batches

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"
NUM_LAYERS = 36
LAYERS_TO_HOOK = [2, NUM_LAYERS // 3, (2 * NUM_LAYERS) // 3, NUM_LAYERS - 3]

CACHE_DIR = os.getenv("HF_HOME")
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "feature_topk"


def pick_device_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


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
            feats = transcoder.encode(mlp_input).to(torch.float32)  # (B, T, F)

            mask = state.attention_mask.bool().unsqueeze(-1)
            feats = feats.masked_fill(~mask, float("-inf"))

            b, t, f = feats.shape
            flat_feats = feats.view(b * t, f)

            batch_vals, batch_pos = flat_feats.topk(state.k, dim=0)  # (K, F) each
            prompt_idx_in_batch = batch_pos // t  # (K, F) int64
            token_pos = (batch_pos % t).to(torch.int16)  # (K, F)
            batch_pid = state.prompt_ids[prompt_idx_in_batch].to(torch.int32)  # (K, F)

            ls = state.layer_state[layer_idx]
            combined_vals = torch.cat([ls.vals, batch_vals], dim=0)
            combined_pid = torch.cat([ls.pid, batch_pid], dim=0)
            combined_tpos = torch.cat([ls.tpos, token_pos], dim=0)

            new_vals, new_idx = combined_vals.topk(state.k, dim=0)  # (K, F)
            ls.vals = new_vals
            ls.pid = torch.gather(combined_pid, dim=0, index=new_idx)
            ls.tpos = torch.gather(combined_tpos, dim=0, index=new_idx)

    return pre_hook


def load_transcoders(device: torch.device, dtype: torch.dtype) -> dict[int, SingleLayerTranscoder]:
    wanted = [f"layer_{layer}.safetensors" for layer in LAYERS_TO_HOOK]
    transcoder_dir = Path(snapshot_download(TRANSCODER_REPO, allow_patterns=wanted))
    out: dict[int, SingleLayerTranscoder] = {}
    for layer in LAYERS_TO_HOOK:
        path = transcoder_dir / f"layer_{layer}.safetensors"
        out[layer] = load_relu_transcoder(
            str(path),
            layer,
            device=device,
            dtype=dtype,
            lazy_encoder=False,
            lazy_decoder=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_spec", default="hf:monology/pile-uncopyrighted:train:1000")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--model_id", default=MODEL_ID)
    args = parser.parse_args()

    device, dtype = pick_device_dtype()
    print(f"[INFO] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] loading model")
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
    print(f"[INFO] model loaded in {time.time() - t0:.1f}s")

    transcoders = load_transcoders(device, dtype)
    f_dim = transcoders[LAYERS_TO_HOOK[0]].W_enc.shape[0]
    print(f"[INFO] d_transcoder={f_dim}, layers={LAYERS_TO_HOOK}")

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
        for batch in tqdm(
            iter_batches(
                args.corpus_spec,
                tokenizer,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                device=device,
            ),
            desc="batches",
        ):
            state.attention_mask = batch.attention_mask
            state.prompt_ids = batch.prompt_ids
            with torch.no_grad():
                model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )
    finally:
        for h in handles:
            h.remove()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for layer, ls in state.layer_state.items():
        path = OUTPUT_DIR / f"topk_layer_{layer}.pt"
        torch.save(
            {
                "layer": layer,
                "topk_vals": ls.vals.cpu(),
                "topk_prompt_id": ls.pid.cpu(),
                "topk_token_pos": ls.tpos.cpu(),
                "K": args.k,
                "corpus_spec": args.corpus_spec,
                "max_seq_len": args.max_seq_len,
                "model_id": args.model_id,
            },
            path,
        )
        finite = torch.isfinite(ls.vals).sum().item()
        print(f"[SAVE] {path}  finite-entries={finite}/{ls.vals.numel()}")


if __name__ == "__main__":
    main()
