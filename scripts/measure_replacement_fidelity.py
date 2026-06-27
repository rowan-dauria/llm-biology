#!/usr/bin/env python3
"""Measure KL divergence and delta CE: base Qwen3-4B vs transcoder replacement.

Evaluates three substitution modes:
  - per-layer: one transcoder at a time (diagnostic, identifies which layers hurt most)
  - all-layer: all loaded transcoders simultaneously (the actual replacement model)

Corpus: WikiText-103 test split. Not The Pile, which the transcoders were trained on.

Position 0 of each chunk (the attention-sink token) is excluded from all metrics
— see the POSITION-0 EXCLUSION note above ``chunk_metrics`` / ``SINK_POSITIONS``.

Output: JSON file at --output-dir/replacement_fidelity.json plus a printed table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circuit_tracer.transcoder.single_layer_transcoder import (  # noqa: E402
    SingleLayerTranscoder,
)
from transformer_lens import HookedTransformer  # noqa: E402
from transformer_lens.hook_points import HookPoint  # noqa: E402

from biology_server.attribution import (  # noqa: E402
    DEFAULT_LAYERS,
    MODEL_ID,
    load_transcoders,
    parse_layers,
    pick_device_dtype,
)
from biology_server.tl_forward import ensure_replacement_mlp_hooks  # noqa: E402
from biology_server.tl_model import load_replacement_model  # noqa: E402

# ---------------------------------------------------------------------------
# Fidelity hooks — genuine MLP substitution (not the attribution ghost-skip).
#
# tl_forward.py's _make_mlp_out_hook returns `reconstruction + error.detach()`
# which numerically equals the original MLP output (acts). That is correct for
# attribution (it preserves the linearised gradient path) but wrong here: we
# want the forward pass to actually use the transcoder output.
# ---------------------------------------------------------------------------


def build_fidelity_hooks(
    transcoders: dict[int, SingleLayerTranscoder],
    active_layers: list[int],
) -> list[tuple[str, Callable]]:
    """Per-forward hooks that replace MLP outputs with transcoder reconstructions.

    ensure_replacement_mlp_hooks must be called on the model before the first
    forward pass (done once in main). These hooks are recreated each forward
    to get a fresh mlp_inputs closure.
    """
    mlp_inputs: dict[int, torch.Tensor] = {}
    hooks: list[tuple[str, Callable]] = []

    for layer in active_layers:
        tc = transcoders[layer]

        # TransformerLens invokes forward hooks as hook(tensor, hook=<HookPoint>),
        # so the second parameter MUST be named `hook` to receive that keyword.
        def _in(acts: torch.Tensor, hook: HookPoint, _l: int = layer) -> torch.Tensor:  # noqa: ARG001
            mlp_inputs[_l] = acts
            return acts

        def _out(
            acts: torch.Tensor,
            hook: HookPoint,
            _l: int = layer,
            _tc: SingleLayerTranscoder = tc,  # noqa: ARG001
        ) -> torch.Tensor:
            inp = mlp_inputs[_l]
            feats = _tc.encode(inp)
            recon = _tc.decode(
                feats.to(_tc.W_dec.dtype),
                inp if _tc.W_skip is not None else None,
            ).to(acts.dtype)
            # acts here is the original MLP output (possibly detach'd by the permanent
            # _stop_gradient hook from install_freezes, which fires before ours).
            # We ignore it and return the reconstruction unconditionally.
            return recon

        # hook_in: post-ln2 input to the MLP (ReplacementMLP.hook_in, added by
        # ensure_replacement_mlp_hooks). Transcoders are trained in this space.
        hooks.append((f"blocks.{layer}.mlp.hook_in", _in))
        # hook_mlp_out: block-level MLP output that feeds into the residual stream.
        hooks.append((f"blocks.{layer}.hook_mlp_out", _out))

    return hooks


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def tokenise_corpus(
    tokenizer: PreTrainedTokenizerBase,
    dataset: str,
    dataset_config: str,
    split: str,
    n_chunks: int,
    chunk_len: int,
    device: torch.device,
) -> list[torch.Tensor]:
    from datasets import load_dataset

    print(f"Loading {dataset}/{dataset_config} split={split}")
    ds = load_dataset(dataset, dataset_config, split=split)
    texts: list[str] = [str(row["text"]) for row in ds]  # type: ignore[index]
    text = "\n\n".join(t for t in texts if t.strip())
    print(f"  {len(text):,} characters")

    ids = torch.tensor(
        tokenizer(text, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )
    print(
        f"  {ids.numel():,} tokens -> max {ids.numel() // chunk_len} non-overlapping chunks of {chunk_len}"
    )

    chunks: list[torch.Tensor] = []
    for start in range(0, ids.numel() - chunk_len, chunk_len):
        window = ids[start : start + chunk_len + 1]  # context[0..T-1] + target[1..T]
        if window.numel() < chunk_len + 1:
            break
        chunks.append(window.unsqueeze(0).to(device))  # (1, T+1)
        if len(chunks) >= n_chunks:
            break

    print(f"  Using {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Per-chunk metrics
#
# >>> POSITION-0 EXCLUSION <<<
# Position 0 of every chunk is an attention-sink token. With Qwen3's BOS-free
# tokenisation it is a *real* mid-document token that nonetheless carries an
# outsized residual norm and an excess of active transcoder features (the sink
# artifact). The operational attribution model neutralises this position (it
# prepends a throwaway special token via attribution.prepend_special_prefix and
# zeros its features via DEFAULT_ZERO_POSITIONS), so the transcoder's pos-0
# reconstruction lives in a regime the real circuit never uses. Scoring it would
# let that artifact dominate KL/ΔCE. We therefore DROP the first SINK_POSITIONS
# prediction position(s) from EVERY metric (base CE, ΔCE, KL). Every scored
# quantity in this script flows through _next_token_ce / the slices below, so
# this single constant governs the exclusion end-to-end.
# ---------------------------------------------------------------------------

SINK_POSITIONS = 1  # leading prediction positions excluded from all metrics


def _drop_sink(t: torch.Tensor) -> torch.Tensor:
    """Drop the leading SINK_POSITIONS positions along dim 0 (see exclusion note)."""
    return t[SINK_POSITIONS:]


def _next_token_ce(log_probs: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over *scored* positions (sink dropped).

    ``log_probs`` is (T, V) float32 log-probs, ``target_ids`` is (T,). The first
    SINK_POSITIONS position(s) are excluded before averaging.
    """
    log_probs = _drop_sink(log_probs)
    target_ids = _drop_sink(target_ids)
    rows = torch.arange(target_ids.shape[0], device=log_probs.device)
    return -log_probs[rows, target_ids].mean()


def chunk_metrics(
    log_p_base: torch.Tensor,  # (T, V) float32 log-probs, precomputed once per chunk
    tc_logits: torch.Tensor,  # (1, T+1, V)
    target_ids: torch.Tensor,  # (T,)
) -> tuple[float, float]:
    """Return (delta_ce, mean_kl) for one chunk against precomputed base log-probs.

    Logit[t] predicts token[t+1], so we slice :-1 for predictions and 1: for
    targets. The leading SINK_POSITIONS position(s) are excluded from BOTH
    metrics (attention-sink artifact — see the POSITION-0 EXCLUSION note above).
    ``log_p_base`` is computed once per chunk in the caller and shared across
    configs (it is identical for every substitution mode). Softmax and KL are
    computed in float32 to avoid numerical issues with the large Qwen3 vocabulary
    (~152k tokens).
    """
    log_p_tc = tc_logits[0, :-1].float().log_softmax(dim=-1)  # (T, V)

    delta_ce = (
        _next_token_ce(log_p_tc, target_ids) - _next_token_ce(log_p_base, target_ids)
    ).item()

    # KL(p_base || p_tc) = E_{v~p_base}[log p_base(v) - log p_tc(v)], averaged over
    # scored positions (sink dropped).
    lp_base = _drop_sink(log_p_base)
    lp_tc = _drop_sink(log_p_tc)
    p_base = lp_base.exp()
    mean_kl = (p_base * (lp_base - lp_tc)).sum(dim=-1).mean().item()

    return delta_ce, mean_kl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="Comma-separated layer indices to load transcoders for (default: project default layers)",
    )
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--chunk-len", type=int, default=512, help="Tokens per evaluation chunk")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "fidelity")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    device, dtype = pick_device_dtype()
    print(f"Device={device}  dtype={dtype}  layers={layers}\n")

    print("Loading model...")
    model: HookedTransformer = load_replacement_model(args.model_id, device=device, dtype=dtype)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_id)  # type: ignore[assignment]

    print("\nLoading transcoders...")
    transcoders = load_transcoders(layers, device=device, dtype=dtype)

    # Wrap MLPs once so hook_in (post-ln2) hook points exist for all layers.
    ensure_replacement_mlp_hooks(model, layers)

    print()
    chunks = tokenise_corpus(
        tokenizer,
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.n_chunks,
        args.chunk_len,
        device,
    )
    if not chunks:
        sys.exit("No chunks loaded — check dataset settings.")

    # One config per layer (diagnostic) + all layers together
    configs: dict[str, list[int]] = {f"layer_{layer}": [layer] for layer in layers}
    configs["all_layers"] = layers

    # chunk-level (delta_ce, mean_kl) per config
    per_chunk: dict[str, list[tuple[float, float]]] = {k: [] for k in configs}
    base_ces: list[float] = []

    n_chunks = len(chunks)
    print(
        f"\nExcluding the first {SINK_POSITIONS} position(s) per chunk from all metrics "
        f"(attention-sink artifact); scoring {args.chunk_len - SINK_POSITIONS} tokens/chunk."
    )
    print(f"Evaluating {n_chunks} chunks × {1 + len(configs)} forward passes each...")
    for i, chunk in enumerate(chunks):
        target_ids = chunk[0, 1:]  # (T,)

        with torch.no_grad():
            base_logits = model(chunk)

        # Base log-probs: computed once per chunk and shared across all configs
        # (identical for every substitution mode). base_logits is freed straight
        # after, leaving only the float32 log-probs alive through the config loop.
        log_p_base = base_logits[0, :-1].float().log_softmax(dim=-1)
        del base_logits
        # Base CE over the same scored positions as the deltas (sink excluded).
        base_ce_i = _next_token_ce(log_p_base, target_ids).item()
        base_ces.append(base_ce_i)

        for cfg_name, active_layers in configs.items():
            hooks = build_fidelity_hooks(transcoders, active_layers)
            with torch.no_grad():
                tc_logits = model.run_with_hooks(chunk, fwd_hooks=hooks)  # type: ignore[arg-type]
            delta_ce, mean_kl = chunk_metrics(log_p_base, tc_logits, target_ids)
            per_chunk[cfg_name].append((delta_ce, mean_kl))
            del tc_logits

        del log_p_base

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1:>4}/{n_chunks}]  base_ce={base_ce_i:.4f}")

    # Aggregate
    def _stats(xs: list[float]) -> tuple[float, float]:
        mu = sum(xs) / len(xs)
        sigma = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        return mu, sigma

    mean_base_ce, std_base_ce = _stats(base_ces)

    summary: dict[str, dict] = {}
    for cfg_name, vals in per_chunk.items():
        dces = [v[0] for v in vals]
        kls = [v[1] for v in vals]
        mu_dce, sig_dce = _stats(dces)
        mu_kl, sig_kl = _stats(kls)
        summary[cfg_name] = {
            "delta_ce_mean": round(mu_dce, 6),
            "delta_ce_std": round(sig_dce, 6),
            "mean_kl_mean": round(mu_kl, 6),
            "mean_kl_std": round(sig_kl, 6),
        }

    # Print
    print(
        f"\nBase CE: {mean_base_ce:.4f} ± {std_base_ce:.4f}  "
        f"(n={n_chunks} chunks × {args.chunk_len - SINK_POSITIONS} scored tokens; pos<{SINK_POSITIONS} excluded)\n"
    )
    print(f"  {'Config':<22} {'delta_CE':>10}  {'±':>8}  {'mean_KL':>12}  {'±':>10}")
    print("  " + "-" * 70)
    for cfg_name, s in summary.items():
        print(
            f"  {cfg_name:<22} {s['delta_ce_mean']:>10.4f}  {s['delta_ce_std']:>8.4f}"
            f"  {s['mean_kl_mean']:>12.6f}  {s['mean_kl_std']:>10.6f}"
        )

    # Save
    output = {
        "model_id": args.model_id,
        "corpus": f"{args.dataset}/{args.dataset_config}/{args.dataset_split}",
        "n_chunks": n_chunks,
        "chunk_len": args.chunk_len,
        "sink_positions_excluded": SINK_POSITIONS,
        "scored_tokens_per_chunk": args.chunk_len - SINK_POSITIONS,
        "layers": layers,
        "base_ce_mean": round(mean_base_ce, 6),
        "base_ce_std": round(std_base_ce, 6),
        "configs": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "replacement_fidelity.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
