#!/usr/bin/env python3
"""Overlay two supernode intervention sweeps on the *same* graph/target.

Use this to show how a supernode's *composition* changes its causal effect:
both ``…__intervention-sweep__….json`` files must steer the same target token on
the same prompt, differing only in which features make up the supernode.

The figure has two panels:

* left: probability of the target token vs steering magnitude ``m`` for each
  sweep, over the full magnitude range, with the clean reference and the
  ablation (``m=0``) / clean (``m=1``) magnitudes marked;
* right: the per-magnitude difference ``P_a − P_b``. Because the shared features
  are scaled identically in both sweeps, this difference isolates the causal
  effect of the feature(s) present in one supernode but not the other.

The raw target *logit* is reported in the legend at the extremes, but the
probability is the scale-free quantity to read (softmax is shift-invariant, so a
flat logit can hide a collapsing probability when competing tokens move).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_sweep(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = sorted(data["results"], key=lambda r: r["magnitude"])
    mag = np.array([r["magnitude"] for r in results], dtype=float)
    prob = np.array([r["target"]["intervened_prob"] for r in results], dtype=float)
    return mag, prob, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_a", type=Path, help="first …__intervention-sweep__….json")
    parser.add_argument("sweep_b", type=Path, help="second …__intervention-sweep__….json")
    parser.add_argument("--label-a", default=None, help="legend label for sweep_a")
    parser.add_argument("--label-b", default=None, help="legend label for sweep_b")
    parser.add_argument("--title", default=None, help="figure title override")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    args = parser.parse_args()

    ma, pa, da = load_sweep(args.sweep_a)
    mb, pb, db = load_sweep(args.sweep_b)

    ta, tb = da["results"][0]["target"], db["results"][0]["target"]
    if ta["token_id"] != tb["token_id"]:
        raise SystemExit(
            f"target tokens differ: {ta['token']!r} vs {tb['token']!r}; "
            "the sweeps must steer the same target to be comparable."
        )
    token = ta["token"]
    clean_prob = float(ta["clean_prob"])
    prompt = da["metadata"]["prompt"]

    na = len(da["metadata"]["constituent_node_ids"])
    nb = len(db["metadata"]["constituent_node_ids"])
    label_a = args.label_a or f"{da['metadata']['supernode']} ({na} features)"
    label_b = args.label_b or f"{db['metadata']['supernode']} ({nb} features)"

    # Difference is only defined where both sweeps share a magnitude.
    common = np.array(sorted(set(ma.tolist()) & set(mb.tolist())), dtype=float)
    pa_map = dict(zip(ma.tolist(), pa.tolist(), strict=True))
    pb_map = dict(zip(mb.tolist(), pb.tolist(), strict=True))
    diff = np.array([pa_map[m] - pb_map[m] for m in common], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # --- left: probability overlay ---------------------------------------
    ax0.axhline(
        clean_prob, ls="--", color="k", lw=1.0, alpha=0.6, label=f"clean = {clean_prob:.3f}"
    )
    ax0.axvline(0.0, ls=":", color="0.6", lw=1.0, alpha=0.8)
    ax0.axvline(1.0, ls=":", color="0.6", lw=1.0, alpha=0.8)
    ax0.text(0.0, 0.02, "ablate", rotation=90, va="bottom", ha="right", fontsize=7, color="0.4")
    ax0.text(1.0, 0.02, "clean", rotation=90, va="bottom", ha="right", fontsize=7, color="0.4")
    ax0.plot(ma, pa, "s-", color="tab:red", ms=5, lw=1.9, zorder=4, label=label_a)
    ax0.plot(mb, pb, "o-", color="tab:blue", ms=5, lw=1.9, zorder=4, label=label_b)
    ax0.set_ylim(0.0, 1.02)
    ax0.set_xlabel("steering magnitude  m   (feature value = m × clean activation)")
    ax0.set_ylabel(f"probability of target token {token!r}")
    ax0.set_title("(a) target probability vs steering", fontsize=10)
    ax0.legend(fontsize=8, loc="lower left")
    ax0.grid(True, alpha=0.25)

    # --- right: isolated effect of the differing feature(s) --------------
    ax1.axhline(0.0, ls="--", color="k", lw=1.0, alpha=0.6)
    ax1.plot(common, diff, "D-", color="tab:purple", ms=4, lw=1.7, zorder=4)
    ax1.set_xlabel("steering magnitude  m")
    ax1.set_ylabel(f"P[{label_a}] − P[{label_b}]")
    ax1.set_title("(b) isolated effect of the differing feature(s)", fontsize=10)
    ax1.grid(True, alpha=0.25)
    lim = float(np.abs(diff).max()) * 1.15 or 0.1
    ax1.set_ylim(-lim, lim)

    title = args.title or f"Supernode composition vs causal effect\n{prompt!r} → {token!r}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
