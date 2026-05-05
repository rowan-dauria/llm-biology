"""Inspect a transcoder feature's top-K activating windows.

Usage:
    python -m feature_lookup.inspect --layer 12 --feature 39989

Re-streams the corpus referenced in the saved top-K file (so windows aren't
materialised to disk; just to memory on demand) and prints each top-K window
with the activating token wrapped in << >>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .corpus import iter_texts

DEFAULT_DIR = Path(__file__).parent.parent / "data" / "feature_topk"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Override path; defaults to data/feature_topk/topk_layer_<L>.pt",
    )
    args = parser.parse_args()

    path = args.file or (DEFAULT_DIR / f"topk_layer_{args.layer}.pt")
    data = torch.load(path, weights_only=False)
    k = int(data["K"])
    corpus_spec = data["corpus_spec"]
    max_seq_len = int(data["max_seq_len"])
    model_id = data["model_id"]

    vals = data["topk_vals"][:, args.feature].tolist()
    pids = data["topk_prompt_id"][:, args.feature].tolist()
    tposs = data["topk_token_pos"][:, args.feature].tolist()

    if not any(v > float("-inf") and v > 0 for v in vals):
        print(f"Feature {args.feature} did not fire on this corpus.")
        return

    needed = {p for p, v in zip(pids, vals, strict=True) if v > 0}
    max_needed = max(needed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_by_id: dict[int, str] = {}
    for pid, text in enumerate(iter_texts(corpus_spec)):
        if pid in needed:
            text_by_id[pid] = text
        if pid >= max_needed:
            break

    print(f"Layer {data['layer']} feature {args.feature} — top-{k} windows from {corpus_spec}")
    for rank, (val, pid, tpos) in enumerate(zip(vals, pids, tposs, strict=True)):
        if val <= 0:
            print(f"#{rank}: (no activation)")
            continue
        text = text_by_id.get(pid)
        if text is None:
            print(f"#{rank}: val={val:.3f} pid={pid} <text not in stream>")
            continue
        enc = tokenizer(text, truncation=True, max_length=max_seq_len, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        if tpos >= len(ids):
            print(f"#{rank}: val={val:.3f} pid={pid} tpos={tpos} out of range ({len(ids)})")
            continue
        lo = max(0, tpos - args.window)
        hi = min(len(ids), tpos + args.window + 1)
        before = tokenizer.decode(ids[lo:tpos])
        target = tokenizer.decode([ids[tpos]])
        after = tokenizer.decode(ids[tpos + 1 : hi])
        print(f"#{rank}: val={val:.3f}  pid={pid}  tpos={tpos}")
        print(f"    {before}<<{target}>>{after}")


if __name__ == "__main__":
    main()
