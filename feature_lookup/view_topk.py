"""View a transcoder feature's top-K activating windows.

Usage:
    python feature_lookup/view_topk.py --layer 12 --feature 39989
    python -m feature_lookup.view_topk --layer 12 --feature 39989

Re-streams the corpus referenced in the saved top-K file (so windows aren't
materialised to disk; just to memory on demand) and prints each top-K window
with the activating token wrapped in << >>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .windows import get_windows, load_tokenizer
except ImportError:
    from windows import get_windows, load_tokenizer

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
    model_id = data["model_id"]

    tokenizer = load_tokenizer(model_id)
    windows = get_windows(data, args.feature, tokenizer, window=args.window)
    if not any(window.active for window in windows):
        print(f"Feature {args.feature} did not fire on this corpus.")
        return

    print(f"Layer {data['layer']} feature {args.feature} — top-{k} windows from {corpus_spec}")
    for window in windows:
        if not window.active:
            print(f"#{window.rank}: {window.rendered}")
            continue
        print(
            f"#{window.rank}: val={window.value:.3f}  pid={window.prompt_id}  tpos={window.token_pos}"
        )
        print(f"    {window.rendered}")


if __name__ == "__main__":
    main()
