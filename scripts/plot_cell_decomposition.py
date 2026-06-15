#!/usr/bin/env python3
"""Plot per-cell random-baseline decompositions for a supernode intervention.

The baseline JSONs are produced by ``scripts/bootstrap_random_supernode_baseline.py``.
Each file contains per-draw target-token probability deltas for a random
size-matched baseline, optionally restricted to a subset of the supernode's
``(layer, pos)`` cells. This script overlays the 5-95th percentile bands and
median curves for those baselines, with an optional targeted intervention sweep.

For the odds metric, the odds transform is applied per bootstrap draw before
aggregation. This matters because ``p / (1 - p)`` is non-linear; aggregating in
probability space first would understate the high-odds tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_from_inputs(baseline: dict[str, Any], sweep: dict[str, Any] | None) -> tuple[float, str]:
    if "target" in baseline["metadata"]:
        target = baseline["metadata"]["target"]
        return float(target["clean_prob"]), str(target["token"])

    if sweep is not None and sweep["results"]:
        target = sweep["results"][0]["target"]
        return float(target["clean_prob"]), str(target["token"])

    raise ValueError("could not find target clean probability/token in inputs")


def baseline_label(baseline: dict[str, Any]) -> str:
    restrict_cells = baseline["metadata"].get("restrict_cells")
    if restrict_cells is None:
        return "all cells (full footprint)"
    return "cells " + ", ".join(str(cell) for cell in restrict_cells)


def baseline_colour(baseline: dict[str, Any]) -> str:
    restrict_cells = baseline["metadata"].get("restrict_cells")
    if restrict_cells is None:
        return "tab:gray"
    if set(restrict_cells) == {"33_10"}:
        return "tab:orange"
    if set(restrict_cells) == {"12_9", "24_9", "24_10"}:
        return "tab:green"
    return "tab:blue"


def summarise_baseline(
    baseline: dict[str, Any], clean_prob: float, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    results = sorted(baseline["results"], key=lambda r: r["magnitude"])
    magnitudes, medians, p5s, p95s = ([] for _ in range(4))

    for result in results:
        deltas = np.array(result["baseline"]["prob_delta"]["samples"], dtype=float)
        draw_metric = to_metric(clean_prob + deltas, metric)
        magnitudes.append(result["magnitude"])
        medians.append(np.median(draw_metric))
        p5s.append(np.percentile(draw_metric, 5))
        p95s.append(np.percentile(draw_metric, 95))

    return (
        np.array(magnitudes, dtype=float),
        np.array(medians, dtype=float),
        np.array(p5s, dtype=float),
        np.array(p95s, dtype=float),
    )


def sweep_curve(sweep: dict[str, Any], metric: str) -> tuple[np.ndarray, np.ndarray]:
    results = sorted(sweep["results"], key=lambda r: r["magnitude"])
    magnitudes = np.array([result["magnitude"] for result in results], dtype=float)
    values = to_metric(
        np.array([result["target"]["intervened_prob"] for result in results], dtype=float),
        metric,
    )
    return magnitudes, values


def default_output_path(sweep_path: Path | None, baseline_path: Path, metric: str) -> Path:
    if sweep_path is not None:
        stem = sweep_path.stem.replace("intervention-sweep", f"{metric}-cell-decomposition")
        return sweep_path.with_name(f"{stem}.png")

    stem = baseline_path.stem.replace("baseline-bootstrap", f"{metric}-cell-decomposition")
    if stem == baseline_path.stem:
        stem = f"{baseline_path.stem}__{metric}-cell-decomposition"
    return baseline_path.with_name(f"{stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_jsons", type=Path, nargs="+", help="baseline bootstrap JSON(s)")
    parser.add_argument("--sweep", type=Path, default=None, help="optional intervention sweep JSON")
    parser.add_argument(
        "--metric",
        choices=("odds", "prob"),
        default="odds",
        help="y-axis quantity: odds p/(1-p) [default] or raw probability p",
    )
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="force a linear y-axis (odds metric only; prob is always linear)",
    )
    args = parser.parse_args()

    metric = args.metric
    baselines = [load(path) for path in args.baseline_jsons]
    sweep = load(args.sweep) if args.sweep is not None else None

    clean_prob, token = target_from_inputs(baselines[0], sweep)
    clean_ref = float(to_metric(clean_prob, metric))
    supernode = baselines[0]["metadata"].get("supernode")
    prompt = baselines[0]["metadata"].get("prompt")
    if sweep is not None:
        supernode = sweep["metadata"].get("supernode", supernode)
        prompt = sweep["metadata"].get("prompt", prompt)

    if metric == "odds":
        ylabel = f"odds of target token {token!r}   =  p / (1 − p)"
        ref_label = f"clean odds = {clean_ref:.1f}"
        use_log = not args.linear_y
    else:
        ylabel = f"probability of target token {token!r}"
        ref_label = f"clean prob = {clean_ref:.3f}"
        use_log = False

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for baseline in baselines:
        magnitudes, medians, p5s, p95s = summarise_baseline(baseline, clean_prob, metric)
        colour = baseline_colour(baseline)
        label = baseline_label(baseline)
        ax.fill_between(
            magnitudes,
            p5s,
            p95s,
            color=colour,
            alpha=0.18,
            lw=0,
            zorder=1,
            label=f"{label} 5-95th pct",
        )
        ax.plot(
            magnitudes,
            medians,
            "o-",
            color=colour,
            ms=4,
            lw=1.3,
            zorder=3,
            label=f"{label} median",
        )

    if sweep is not None:
        sweep_magnitudes, sweep_values = sweep_curve(sweep, metric)
        ax.plot(
            sweep_magnitudes,
            sweep_values,
            "s-",
            color="tab:red",
            ms=5,
            lw=1.9,
            zorder=5,
            label=f"{supernode} (targeted)",
        )

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
    if metric == "prob":
        ax.set_ylim(0.0, 1.02)

    ax.set_xlabel("steering magnitude  m   (feature value = m × clean activation)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{supernode}: per-cell decomposition of the random baseline\n{prompt!r} -> {token!r}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    out = (
        args.output
        if args.output is not None
        else default_output_path(args.sweep, args.baseline_jsons[0], metric)
    )
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
