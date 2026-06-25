"""Draft visualisation of a supernode steering sweep's top-token redistribution.

Consumes the JSON written by ``sweep_supernode_interventions.py`` (which records
``top_intervened_tokens`` per magnitude) and shows where next-token probability
mass goes as the supernode is steered. Both panels share the magnitude x-axis and
trace the same top-N tokens (by peak probability over the plotted range), labelled
by raw token string:

- Top: stacked-area probability mass of the top-N tokens (+ an "other" remainder),
  the big-picture handoff between tokens.
- Bottom: log-y trajectories of the same tokens, to expose the smaller gains
  (e.g. " Texas", " Houston") that the linear stack hides.

Draft quality, not publication polish.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Raw token strings are used as labels, except for the Chinese-Austin token
# (DejaVu Sans lacks CJK glyphs, so romanise it).
TOKEN_DISPLAY = {"奥斯": "Austin (zh)"}
# Colour cycle for non-target tokens (tab10); the target token is forced black.
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
OTHER_COLOUR = "#e8e8e8"


def load_sweep(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_top_tokens(results: list[dict[str, Any]], n: int) -> list[str]:
    """Return the ``n`` tokens with the highest peak probability over the range."""
    peak: dict[str, float] = defaultdict(float)
    for r in results:
        for t in r["top_intervened_tokens"]:
            peak[t["token"]] = max(peak[t["token"]], float(t["value"]))
    return [tok for tok, _p in sorted(peak.items(), key=lambda kv: -kv[1])[:n]]


def build_series(
    results: list[dict[str, Any]], tokens: list[str]
) -> tuple[list[float], dict[str, np.ndarray]]:
    """Return (magnitudes, token_trajectories) for ``tokens`` (NaN where absent)."""
    mags = [float(r["magnitude"]) for r in results]
    token_traj: dict[str, np.ndarray] = {tok: np.full(len(results), np.nan) for tok in tokens}
    for i, r in enumerate(results):
        present = {t["token"]: float(t["value"]) for t in r["top_intervened_tokens"]}
        for tok in tokens:
            if tok in present:
                token_traj[tok][i] = present[tok]
    return mags, token_traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_json", type=Path, help="sweep_supernode_interventions.py output JSON"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--m-min",
        type=float,
        default=None,
        help="Crop x-axis lower bound (e.g. -12 to hide OOD tail).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="How many top tokens to show in both panels."
    )
    args = parser.parse_args()

    sweep = load_sweep(args.sweep_json.expanduser().resolve())
    results = sweep["results"]
    if args.m_min is not None:
        results = [r for r in results if float(r["magnitude"]) >= args.m_min]
    top_tokens = select_top_tokens(results, args.top_n)
    mags, token_traj = build_series(results, top_tokens)
    target_tok = results[0]["target"]["token"]

    # Shared colour map: target token black, the rest cycle through the palette.
    colour_cycle = iter(PALETTE)
    token_colour = {
        tok: ("#000000" if tok == target_tok else next(colour_cycle, None)) for tok in top_tokens
    }

    def disp(tok: str) -> str:
        return TOKEN_DISPLAY.get(tok, f"{tok!r}")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]}
    )

    # --- Panel A: stacked top-N token mass (+ "other" remainder) --------------
    stack_arrays = [np.nan_to_num(token_traj[tok], nan=0.0) for tok in top_tokens]
    other = np.clip(1.0 - np.sum(stack_arrays, axis=0), 0.0, None)
    ax_top.stackplot(
        mags,
        *stack_arrays,
        other,
        labels=[disp(tok) for tok in top_tokens] + ["other (not in top-N)"],
        colors=[token_colour[tok] for tok in top_tokens] + [OTHER_COLOUR],
        alpha=0.9,
    )
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("next-token probability mass")
    ax_top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

    # --- Panel B: same tokens, log-y individual trajectories -----------------
    floor = 1e-6
    for tok in top_tokens:
        y = token_traj[tok].copy()
        y[y < floor] = np.nan  # don't draw the unrecorded tail
        ax_bot.plot(
            mags,
            y,
            marker="o",
            ms=3,
            lw=1.6,
            color=token_colour[tok],
            label=disp(tok),
        )
    ax_bot.set_yscale("log")
    ax_bot.set_ylim(floor, 1.2)
    ax_bot.set_ylabel("token probability (log)")
    ax_bot.set_xlabel("steering factor  m   (1 = clean, 0 = ablate, <0 = sign-flipped)")

    # reference lines on both panels
    for ax in (ax_top, ax_bot):
        ax.axvline(1.0, color="red", lw=1, ls="-", zorder=10)
        ax.axvline(0.0, color="red", lw=1, ls=":", zorder=10)
        ax.margins(x=0)
    ax_top.text(1.0, 1.005, "clean", ha="center", va="bottom", fontsize=7, color="0.3")
    ax_top.text(0.0, 1.005, "ablate", ha="center", va="bottom", fontsize=7, color="0.3")

    fig.tight_layout()
    out = args.output or args.sweep_json.with_name(
        args.sweep_json.stem + "__top-token-redistribution.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
