"""Visualise a supernode steering sweep's top-token redistribution.

Consumes the JSON written by ``sweep_supernode_interventions.py`` (which records
``top_intervened_tokens`` per magnitude) and shows where next-token probability
mass goes as the supernode is steered, as a stacked area of the top-N tokens
(by peak probability over the plotted range, labelled by raw token string)
plus an "other" remainder.

Use this for the report supplement's top-token redistribution figure.
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
# Paul Tol's colour-blind-safe qualitative palette; the target token gets its
# own reserved entry (sand) so it stands out from the rest of the cycle.
PALETTE = [
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#DDDDDD",  # grey
]
OTHER_COLOUR = "#e8e8e8"
TARGET_COLOUR = "#DDCC77"  # sand
HOUSTON_COLOUR = "#555555"  # dark grey, distinct from the light "other"/palette greys


def load_sweep(path: Path) -> dict[str, Any]:
    """Load a supernode sweep JSON (as written by ``sweep_supernode_interventions.py``)."""
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
    """CLI entry point: plot the stacked top-token redistribution across a steering sweep."""
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
        "--top-n", type=int, default=10, help="How many top tokens to show in the stack."
    )
    args = parser.parse_args()

    sweep = load_sweep(args.sweep_json.expanduser().resolve())
    results = sweep["results"]
    if args.m_min is not None:
        results = [r for r in results if float(r["magnitude"]) >= args.m_min]
    top_tokens = select_top_tokens(results, args.top_n)
    mags, token_traj = build_series(results, top_tokens)
    target_tok = results[0]["target"]["token"]

    # Shared colour map: target token gets the reserved sand swatch, Houston
    # gets a dedicated dark grey, and the rest cycle through the remaining
    # palette entries (sand excluded, so it isn't reused for a second token).
    colour_cycle = iter(c for c in PALETTE if c.lower() != TARGET_COLOUR.lower())
    token_colour = {}
    for tok in top_tokens:
        if tok == target_tok:
            token_colour[tok] = TARGET_COLOUR
        elif tok.strip().lower() == "houston":
            token_colour[tok] = HOUSTON_COLOUR
        else:
            token_colour[tok] = next(colour_cycle, None)

    def disp(tok: str) -> str:
        """Render a raw token string as a capitalised display label for the legend."""
        if tok in TOKEN_DISPLAY:
            return TOKEN_DISPLAY[tok]
        stripped = tok.strip()
        if not stripped:
            return "(Newline)" if "\n" in tok else "(Space)"
        return stripped[0].upper() + stripped[1:]

    label_fontsize = 16
    tick_fontsize = 13

    fig, ax_top = plt.subplots(figsize=(11, 6))

    # --- Stacked top-N token mass (+ "other" remainder) ------------------------
    stack_arrays = [np.nan_to_num(token_traj[tok], nan=0.0) for tok in top_tokens]
    other = np.clip(1.0 - np.sum(stack_arrays, axis=0), 0.0, None)
    polys = ax_top.stackplot(
        mags,
        *stack_arrays,
        other,
        labels=[disp(tok) for tok in top_tokens] + ["Other (not in top-N)"],
        colors=[token_colour[tok] for tok in top_tokens] + [OTHER_COLOUR],
        alpha=0.9,
    )
    other_poly = polys[-1]
    other_poly.set_hatch("///")
    other_poly.set_edgecolor("0.6")
    other_poly.set_linewidth(0.0)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Next-token probability mass", fontsize=label_fontsize)
    ax_top.set_xlabel("Steering factor", fontsize=label_fontsize)
    ax_top.tick_params(axis="both", labelsize=tick_fontsize)
    ax_top.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=label_fontsize,
        frameon=False,
    )

    # reference lines
    ax_top.axvline(1.0, color="red", lw=1, ls="-", zorder=10)
    ax_top.axvline(0.0, color="red", lw=1, ls=":", zorder=10)
    ax_top.margins(x=0)
    ax_top.text(
        1.0, 1.005, "Unsteered", ha="left", va="bottom", fontsize=label_fontsize, color="0.3"
    )
    ax_top.text(0.0, 1.005, "Ablate", ha="right", va="bottom", fontsize=label_fontsize, color="0.3")

    fig.tight_layout()
    out = args.output or args.sweep_json.with_name(
        args.sweep_json.stem + "__top-token-redistribution.png"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
