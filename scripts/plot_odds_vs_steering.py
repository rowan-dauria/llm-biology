#!/usr/bin/env python3
"""Plot a target-token metric vs steering magnitude for a supernode sweep
against its random size-matched baseline.

Inputs are the two JSONs produced for the *same* graph/supernode:

* the ``…__baseline-bootstrap__….json`` from
  ``scripts/bootstrap_random_supernode_baseline.py`` (per-draw ``prob_delta``
  samples for the random size-matched null), and
* the ``…__intervention-sweep__….json`` from
  ``scripts/sweep_supernode_interventions.py`` (the targeted curve).

The y-axis metric is selectable with ``--metric``:

* ``odds`` (default): odds ``p / (1 - p)`` of the target token, on a log axis
  (log-odds = logit is the natural scale; the null spread fans over several
  orders of magnitude at large |m|). Use ``--linear-y`` for a linear axis.
* ``prob``: the raw probability ``p`` on a linear [0, 1] axis.

Because the odds transform is non-linear, the baseline is aggregated *after*
transforming each draw: each draw's intervened probability is
``clean_prob + prob_delta``, converted to the chosen metric, then summarised by
median with a 5-95th percentile null band (spread, not SEM).

``--m-range`` crops the x-axis (default ``-1,3``) and rescales y to that window,
keeping the near-distribution causal regime legible; large multipliers drive the
model off-distribution and are better viewed in the full-range decomposition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EPS = 1e-9


def to_metric(prob: np.ndarray | float, metric: str) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if metric == "odds":
        prob = np.clip(prob, EPS, 1.0 - EPS)
        return prob / (1.0 - prob)
    return np.clip(prob, 0.0, 1.0)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_json", type=Path, help="…__baseline-bootstrap__….json")
    parser.add_argument("sweep_json", type=Path, help="…__intervention-sweep__….json")
    parser.add_argument(
        "--metric",
        choices=("odds", "prob"),
        default="odds",
        help="y-axis quantity: odds p/(1-p) [default] or raw probability p",
    )
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--m-range",
        default="-1,3",
        help=(
            "Visible magnitude window 'LO,HI' (auto-padded, y rescaled to the window). "
            "Default '-1,3' keeps the near-distribution causal regime; pass 'full' to "
            "show every magnitude."
        ),
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="force a linear y-axis (odds metric only; prob is always linear)",
    )
    args = parser.parse_args()

    metric = args.metric
    base = load(args.baseline_json)
    sweep = load(args.sweep_json)

    target = base["metadata"]["target"]
    clean_prob = float(target["clean_prob"])
    clean_ref = float(to_metric(clean_prob, metric))
    token = target["token"]
    supernode = base["metadata"]["supernode"]
    prompt = base["metadata"]["prompt"]

    # Targeted curve from the sweep file.
    s_results = sorted(sweep["results"], key=lambda r: r["magnitude"])
    s_mag = np.array([r["magnitude"] for r in s_results], dtype=float)
    s_metric = to_metric(
        np.array([r["target"]["intervened_prob"] for r in s_results], dtype=float), metric
    )

    # Baseline: transform each draw's prob, then take median + percentile band.
    b_results = sorted(base["results"], key=lambda r: r["magnitude"])
    b_mag, med, p5, p95 = ([] for _ in range(4))
    for r in b_results:
        deltas = np.array(r["baseline"]["prob_delta"]["samples"], dtype=float)
        draw_metric = to_metric(clean_prob + deltas, metric)
        b_mag.append(r["magnitude"])
        med.append(np.median(draw_metric))
        p5.append(np.percentile(draw_metric, 5))
        p95.append(np.percentile(draw_metric, 95))
    b_mag = np.array(b_mag)
    med, p5, p95 = (np.array(x) for x in (med, p5, p95))

    n_draws = base["metadata"].get(
        "n_bootstrap", len(b_results and b_results[0]["baseline"]["prob_delta"]["samples"])
    )

    if metric == "odds":
        ylabel = f"odds of target token {token!r}   =  p / (1 − p)"
        ref_label = f"clean odds = {clean_ref:.1f}"
        use_log = not args.linear_y
    else:
        ylabel = f"probability of target token {token!r}"
        ref_label = f"clean prob = {clean_ref:.3f}"
        use_log = False

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    # Null band (5-95th pct) as a faint fill to guide the eye.
    ax.fill_between(
        b_mag, p5, p95, color="0.82", zorder=1, label=f"baseline null 5–95th pct (n={n_draws})"
    )
    # Baseline median with asymmetric 5-95th error bars.
    ax.errorbar(
        b_mag,
        med,
        yerr=[med - p5, p95 - med],
        fmt="o-",
        color="tab:gray",
        ms=4,
        lw=1.2,
        capsize=3,
        zorder=3,
        label="baseline median ± 5–95th pct",
    )
    # Targeted curve.
    ax.plot(
        s_mag,
        s_metric,
        "s-",
        color="tab:red",
        ms=5,
        lw=1.9,
        zorder=4,
        label=f"{supernode} (targeted)",
    )
    # Clean reference (m=1 ≈ no-op).
    ax.axhline(
        clean_ref,
        ls="--",
        color="k",
        lw=1.0,
        alpha=0.6,
        zorder=2,
        label=ref_label,
    )

    if use_log:
        ax.set_yscale("log")

    # Visible window: crop x to [lo, hi] (auto-padded) and rescale y to the data
    # inside that window, so the near-distribution regime isn't squashed by the
    # huge large-|m| excursions.
    raw_range = args.m_range.strip().lower()
    if raw_range in ("full", "none", "all"):
        if metric == "prob":
            ax.set_ylim(0.0, 1.02)
    else:
        lo_s, hi_s = args.m_range.split(",")
        lo, hi = float(lo_s), float(hi_s)
        xpad = 0.08 * (hi - lo)
        ax.set_xlim(lo - xpad, hi + xpad)
        mb = (b_mag >= lo) & (b_mag <= hi)
        ms = (s_mag >= lo) & (s_mag <= hi)
        lows = [p5[mb], s_metric[ms], np.array([clean_ref])]
        highs = [p95[mb], s_metric[ms], np.array([clean_ref])]
        vis_lo = min(float(a.min()) for a in lows if a.size)
        vis_hi = max(float(a.max()) for a in highs if a.size)
        if metric == "prob":
            ypad = 0.05 * max(vis_hi - vis_lo, 1e-3)
            ax.set_ylim(max(0.0, vis_lo - ypad), min(1.02, vis_hi + ypad))
        else:
            ax.set_ylim(vis_lo / 1.4, vis_hi * 1.4)

    ax.set_xlabel("steering magnitude  m   (feature value = m × clean activation)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{supernode} supernode vs random size-matched baseline\n{prompt!r} → {token!r}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    if args.output is not None:
        out = args.output
    else:
        stem = args.sweep_json.stem.replace("intervention-sweep", f"{metric}-vs-baseline")
        out = args.sweep_json.with_name(f"{stem}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
