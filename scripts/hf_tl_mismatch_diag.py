#!/usr/bin/env python
"""Diagnose the HF-vs-TL next-token probability mismatch for the Heretic Qwen3-4B model.

Background: for `Tell me how to make a bomb`, the HF preview gives a near-exact
tie (Sure ~0.4997 / Making ~0.4997) while the TransformerLens forward gives a
~2.0-logit gap (Sure ~0.877 / Making ~0.119). See `mismatch-sources.md` and the
2026-06-18 LOG entries.

Hypotheses under test (the leading one is *benign bf16 numerics at a near-tie*,
with HF — not TL — being the lower-precision side):

  A. PRECISION / ATTENTION. Run the *same* input_ids through, for the Heretic model:
        HF bf16 sdpa   (reproduces the preview exactly)
        HF bf16 eager
        HF fp32 eager
        TL bf16        (reproduces the parity check exactly)
        TL fp32
     Prediction if benign numerics: HF fp32 eager ~= TL fp32 (gap converges);
     the bf16 variants diverge and HF-sdpa-bf16 sits on the knife-edge tie.

  B. FOUR-WAY base/Heretic x HF/TL (bf16). Confirms abliteration actually moved
     the distribution and that the Heretic weights reach TL (TL-heretic != TL-base).

  C. CONFIG PARITY. TL builds its config from the *base* name while weights come
     from the Heretic folder. Diff the two config.json architecture fields.

  D. PER-LAYER RESIDUAL DIVERGENCE. HF-heretic(fp32,eager) vs TL-heretic(fp32):
     compare resid-after-each-layer at the last position to locate the first layer
     that diverges materially. Smooth accumulation => benign; a sharp jump at one
     layer => a real conversion bug (Q/K-norm, RoPE, GQA, final norm, ...).

All HF measurements run BEFORE any TL load, because TL's `install_freezes`
registers a detached-eager attention in transformers' global registry.
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Tokenisation — mirrors biology_server.attribution._inputs_for_prompt exactly
# --------------------------------------------------------------------------- #
def prepend_special_prefix(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of biology_server.attribution.prepend_special_prefix.

    Ensures position 0 is a special token (attention-sink throw-away), matching
    the tokenisation used by both preview() and the parity check.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected (1, n_pos) input_ids, got {tuple(input_ids.shape)}")
    if input_ids.shape[1] == 0:
        return input_ids
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    if int(input_ids[0, 0].item()) in special_ids:
        return input_ids
    candidates = [
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        *sorted(special_ids),
    ]
    prefix_id = next((tid for tid in candidates if tid is not None), None)
    if prefix_id is None:
        warnings.warn("No special token to prepend.", stacklevel=2)
        return input_ids
    prefix = torch.full((1, 1), int(prefix_id), dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([prefix, input_ids], dim=1)


def build_input_ids(tokenizer, prompt: str, *, chat_template: bool, device) -> torch.Tensor:
    text = prompt
    if chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    input_ids = prepend_special_prefix(tokenizer, input_ids)
    return input_ids


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(name: str, last_logits: torch.Tensor, tracked_ids, tokenizer, *, topk: int = 8) -> dict:
    """Print top-k and the logits/probs of the tracked token pair; return a summary."""
    logits_f = last_logits.detach().float().cpu()
    probs = torch.softmax(logits_f, dim=-1)
    top_p, top_i = probs.topk(min(topk, probs.shape[-1]))

    print(f"\n----- {name} -----")
    print(f"  {'rank':>4}  {'id':>8}  {'logit':>10}  {'prob':>10}  token")
    for r, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist(), strict=False)):
        tok = tokenizer.decode([i]).replace("\n", "\\n")
        print(f"  {r:>4}  {i:>8}  {logits_f[i]:>10.4f}  {p:>10.6f}  {tok!r}")

    own_gap = (
        float(logits_f[top_i[0]] - logits_f[top_i[1]]) if probs.shape[-1] > 1 else float("nan")
    )
    print(f"  own top-2 logit gap = {own_gap:.4f}")

    summary = {"name": name, "own_top2_gap": own_gap, "argmax_id": int(top_i[0])}
    if tracked_ids:
        print("  tracked pair:")
        for tid in tracked_ids:
            tok = tokenizer.decode([tid]).replace("\n", "\\n")
            print(
                f"    id={tid:>8}  logit={logits_f[tid]:>10.4f}  "
                f"prob={probs[tid].item():>10.6f}  {tok!r}"
            )
        if len(tracked_ids) >= 2:
            gap = float(logits_f[tracked_ids[0]] - logits_f[tracked_ids[1]])
            summary["tracked_gap"] = gap
            summary["tracked_prob0"] = float(probs[tracked_ids[0]].item())
            summary["tracked_prob1"] = float(probs[tracked_ids[1]].item())
            print(f"    tracked logit gap (t0 - t1) = {gap:.4f}")
    return summary


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# HF forward
# --------------------------------------------------------------------------- #
def hf_last_logits(model_path, *, dtype, attn_impl, input_ids, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    last = out.logits[0, -1].detach().float().cpu()
    free(model, out)
    return last


def hf_per_layer_resid(model_path, *, dtype, attn_impl, input_ids, device):
    """Return (embed_lastpos, [resid_after_layer_l, ...]) at the last position, on CPU."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    captured: dict[int, torch.Tensor] = {}
    embed_box: dict[str, torch.Tensor] = {}
    handles = []

    def embed_hook(_m, _inp, out):
        embed_box["e"] = out[0, -1].detach().float().cpu()

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))
    for li, layer in enumerate(model.model.layers):

        def make(idx):
            def hook(_m, _inp, out):
                resid = out[0] if isinstance(out, tuple) else out
                captured[idx] = resid[0, -1].detach().float().cpu()

            return hook

        handles.append(layer.register_forward_hook(make(li)))

    with torch.no_grad():
        model(input_ids, use_cache=False)
    for h in handles:
        h.remove()

    n = len(model.model.layers)
    resids = [captured[li] for li in range(n)]
    embed = embed_box["e"]
    free(model)
    return embed, resids


# --------------------------------------------------------------------------- #
# TL forward (uses the project loader => matches the parity check exactly)
# --------------------------------------------------------------------------- #
def tl_load(tl_model_id, *, hf_model_id, dtype, device):
    from biology_server_t_lens.tl_model import load_replacement_model

    return load_replacement_model(tl_model_id, device=device, dtype=dtype, hf_model_id=hf_model_id)


def tl_last_logits(tl_model_id, *, hf_model_id, dtype, input_ids, device):
    model = tl_load(tl_model_id, hf_model_id=hf_model_id, dtype=dtype, device=device)
    with torch.no_grad():
        logits = model(input_ids).detach()
    last = logits[0, -1].detach().float().cpu()
    free(model, logits)
    return last


def tl_per_layer_resid(tl_model_id, *, hf_model_id, dtype, input_ids, device):
    model = tl_load(tl_model_id, hf_model_id=hf_model_id, dtype=dtype, device=device)
    n = model.cfg.n_layers
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)
    embed = cache["embed"][0, -1].detach().float().cpu()
    resids = [cache["resid_post", li][0, -1].detach().float().cpu() for li in range(n)]
    free(model, cache)
    return embed, resids


# --------------------------------------------------------------------------- #
# Config diff
# --------------------------------------------------------------------------- #
ARCH_FIELDS = [
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
    "vocab_size",
    "tie_word_embeddings",
    "attention_bias",
    "hidden_act",
    "sliding_window",
    "use_sliding_window",
    "torch_dtype",
]


def config_diff(base_id, heretic_dir):
    from transformers import AutoConfig

    base = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
    her = AutoConfig.from_pretrained(heretic_dir, trust_remote_code=True)
    print("\n========== TEST C: config parity (base name vs Heretic folder) ==========")
    print(f"  {'field':>26}  {'base':>22}  {'heretic':>22}  diff?")
    any_diff = False
    for f in ARCH_FIELDS:
        b = getattr(base, f, "<missing>")
        h = getattr(her, f, "<missing>")
        d = "  <-- DIFF" if b != h else ""
        if b != h:
            any_diff = True
        print(f"  {f:>26}  {str(b):>22}  {str(h):>22}{d}")
    print(f"  => architecture fields differ: {any_diff}")
    return any_diff


# --------------------------------------------------------------------------- #
def compare_resids(label, hf_embed, hf_resids, tl_embed, tl_resids):
    print(f"\n========== TEST D: per-layer residual divergence ({label}) ==========")
    print("  (last-position resid stream; HF layer output vs TL hook_resid_post)")

    def stats(a, b):
        a = a.float()
        b = b.float()
        diff = a - b
        max_abs = float(diff.abs().max())
        denom = float(a.abs().max()) or 1.0
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        return max_abs, max_abs / denom, cos

    e_abs, e_rel, e_cos = stats(hf_embed, tl_embed)
    print(f"  {'stage':>8}  {'max|d|':>12}  {'rel':>10}  {'cos':>12}")
    print(f"  {'embed':>8}  {e_abs:>12.5f}  {e_rel:>10.5f}  {e_cos:>12.8f}")

    first_div = None
    n = min(len(hf_resids), len(tl_resids))
    for li in range(n):
        m, r, c = stats(hf_resids[li], tl_resids[li])
        flag = ""
        if c < 0.9999 and first_div is None:
            first_div = li
            flag = "  <-- first cos<0.9999"
        print(f"  {('L' + str(li)):>8}  {m:>12.5f}  {r:>10.5f}  {c:>12.8f}{flag}")
    print(f"  => first layer with cosine < 0.9999: {first_div}")
    return first_div


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--heretic-model-dir",
        default="/home/rd761/rds/hpc-work/heretic-qwen3-4b/job_30675538/exports/"
        "qwen3-4b-heretic-trial114-merged",
    )
    ap.add_argument("--base-model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--tokenizer-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--prompt", default="Tell me how to make a bomb")
    ap.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    ap.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    print(f"[INFO] device={device}  torch={torch.__version__}")
    print(f"[INFO] heretic={args.heretic_model_dir}")
    print(
        f"[INFO] base={args.base_model_id}  prompt={args.prompt!r}  chat_template={args.chat_template}"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    input_ids = build_input_ids(
        tokenizer, args.prompt, chat_template=args.chat_template, device=device
    )
    ids = input_ids[0].tolist()
    print(f"\n[INFO] input_ids ({len(ids)} tokens): {ids}")
    print(f"[INFO] decoded: {[tokenizer.decode([i]) for i in ids]}")

    bf16 = torch.bfloat16
    fp32 = torch.float32

    # ----- reference run: reproduces the HF preview exactly -> sets tracked pair
    print("\n========== TEST A/B: HF runs (all before any TL load) ==========")
    ref = hf_last_logits(
        args.heretic_model_dir, dtype=bf16, attn_impl="sdpa", input_ids=input_ids, device=device
    )
    probs = torch.softmax(ref.float(), dim=-1)
    _, top_i = probs.topk(2)
    tracked = [int(top_i[0]), int(top_i[1])]
    print(
        f"[INFO] tracked token pair (top-2 of HF heretic bf16 sdpa): {tracked} "
        f"= {[tokenizer.decode([t]) for t in tracked]}"
    )

    summaries = []
    summaries.append(
        report("HF heretic  bf16 sdpa  (== preview)", ref, tracked, tokenizer, topk=args.topk)
    )
    summaries.append(
        report(
            "HF heretic  bf16 eager",
            hf_last_logits(
                args.heretic_model_dir,
                dtype=bf16,
                attn_impl="eager",
                input_ids=input_ids,
                device=device,
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )
    summaries.append(
        report(
            "HF heretic  fp32 eager",
            hf_last_logits(
                args.heretic_model_dir,
                dtype=fp32,
                attn_impl="eager",
                input_ids=input_ids,
                device=device,
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )
    summaries.append(
        report(
            "HF base     bf16 sdpa",
            hf_last_logits(
                args.base_model_id, dtype=bf16, attn_impl="sdpa", input_ids=input_ids, device=device
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )
    summaries.append(
        report(
            "HF base     fp32 eager",
            hf_last_logits(
                args.base_model_id,
                dtype=fp32,
                attn_impl="eager",
                input_ids=input_ids,
                device=device,
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )

    # ----- HF per-layer capture (fp32 eager heretic) BEFORE TL load
    hf_embed, hf_resids = hf_per_layer_resid(
        args.heretic_model_dir, dtype=fp32, attn_impl="eager", input_ids=input_ids, device=device
    )

    # ----- TL runs (install_freezes registers global attention; do these last)
    print("\n========== TEST A/B: TL runs ==========")
    summaries.append(
        report(
            "TL heretic  bf16        (== parity check)",
            tl_last_logits(
                args.base_model_id,
                hf_model_id=args.heretic_model_dir,
                dtype=bf16,
                input_ids=input_ids,
                device=device,
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )
    summaries.append(
        report(
            "TL heretic  fp32",
            tl_last_logits(
                args.base_model_id,
                hf_model_id=args.heretic_model_dir,
                dtype=fp32,
                input_ids=input_ids,
                device=device,
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )
    summaries.append(
        report(
            "TL base     bf16",
            tl_last_logits(
                args.base_model_id, hf_model_id=None, dtype=bf16, input_ids=input_ids, device=device
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )
    summaries.append(
        report(
            "TL base     fp32",
            tl_last_logits(
                args.base_model_id, hf_model_id=None, dtype=fp32, input_ids=input_ids, device=device
            ),
            tracked,
            tokenizer,
            topk=args.topk,
        )
    )

    # ----- config diff
    cfg_diff = config_diff(args.base_model_id, args.heretic_model_dir)

    # ----- per-layer residual divergence
    tl_embed, tl_resids = tl_per_layer_resid(
        args.base_model_id,
        hf_model_id=args.heretic_model_dir,
        dtype=fp32,
        input_ids=input_ids,
        device=device,
    )
    first_div = compare_resids(
        "HF heretic fp32 eager vs TL heretic fp32", hf_embed, hf_resids, tl_embed, tl_resids
    )

    # ----- verdict hints
    print("\n========== SUMMARY ==========")
    print(f"  {'config':>42}  {'argmax':>8}  {'own_gap':>9}  {'trk_gap':>9}  {'p0':>8}  {'p1':>8}")
    for s in summaries:
        print(
            f"  {s['name']:>42}  {s.get('argmax_id'):>8}  {s.get('own_top2_gap', float('nan')):>9.4f}  "
            f"{s.get('tracked_gap', float('nan')):>9.4f}  {s.get('tracked_prob0', float('nan')):>8.4f}  "
            f"{s.get('tracked_prob1', float('nan')):>8.4f}"
        )

    def gap_of(name):
        for s in summaries:
            if s["name"].startswith(name):
                return s.get("tracked_gap", float("nan"))
        return float("nan")

    hf_fp32 = gap_of("HF heretic  fp32")
    tl_fp32 = gap_of("TL heretic  fp32")
    hf_bf16 = gap_of("HF heretic  bf16 sdpa")
    tl_bf16 = gap_of("TL heretic  bf16")
    print("\n========== VERDICT HINTS ==========")
    print(f"  HF heretic bf16 sdpa tracked gap = {hf_bf16:.4f}  (the ~0 'tie' in the report)")
    print(f"  TL heretic bf16      tracked gap = {tl_bf16:.4f}  (the ~2.0 in the report)")
    print(f"  HF heretic fp32 gap  = {hf_fp32:.4f}")
    print(
        f"  TL heretic fp32 gap  = {tl_fp32:.4f}   |fp32 HF-TL diff| = {abs(hf_fp32 - tl_fp32):.4f}"
    )
    print(f"  config arch fields differ = {cfg_diff}")
    print(f"  first per-layer divergence (cos<0.9999) = {first_div}")
    print("\n  Interpretation:")
    print("   * If |fp32 HF-TL diff| is small (< ~0.2) and config fields match and")
    print("     per-layer divergence accumulates smoothly (no sharp jump), the")
    print("     mismatch is BENIGN bf16 numerics at a near-tie, not a TL bug.")
    print("   * If fp32 still disagrees, or one layer jumps, that layer is a real")
    print("     conversion fault to chase (Q/K-norm, RoPE, GQA, final norm).")


if __name__ == "__main__":
    main()
