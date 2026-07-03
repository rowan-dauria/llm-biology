#!/usr/bin/env python3
"""Plot all-node cross-model feature fates as an unsupervised dot map.

Inputs are the two all-node CSV files produced by
``compare_cross_model_feature_activations.py``. Each feature is drawn as an
unlabelled dot in its attribution-graph cell, with token position on the x-axis
and tracked transcoder layer on the y-axis. The script also prints and saves
summary statistics for the late decision region identified by the pinned-node
analysis: layers 24/33 at positions 11/18.

Example:

    llm-biology/venv/bin/python3 \
        llm-biology/scripts/figures/plot_cross_model_feature_fate_unsupervised.py \
        llm-biology/data/base_jailbreak_comparison/2026-07-03__base_to_jailbroken_allnodes__vs-qwen3-4b-heretic-trial114-merged.csv \
        llm-biology/data/base_jailbreak_comparison/2026-07-03__jailbroken_to_base_allnodes__vs-qwen3-4b.csv \
        --output-pdf report/figures/feature_fate_map_unsupervised.pdf \
        --output-png report/figures/feature_fate_map_unsupervised.png \
        --summary-output llm-biology/data/base_jailbreak_comparison/allnodes_feature_fate_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LAYERS = [2, 12, 24, 33]
REGION_LAYERS = {24, 33}
REGION_POSITIONS = {11, 18}
DEGRADED_OUTCOMES = {"absent", "reduced"}

OUTCOME_STYLES = {
    "shared_active": {
        "label": "shared active",
        "face": "#dff1df",
        "edge": "#2c6b38",
        "dot": "#2c6b38",
    },
    "reduced": {
        "label": "reduced",
        "face": "#fde7b6",
        "edge": "#b85e00",
        "dot": "#b85e00",
    },
    "absent": {
        "label": "absent",
        "face": "#f9d6d5",
        "edge": "#a53232",
        "dot": "#a53232",
    },
}

CALLOUT_KEYWORDS = {
    "base_to_jailbroken": [
        ({"shared_active"}, ("bomb", "explosive")),
        ({"absent", "reduced"}, ("negation",)),
        ({"absent", "reduced"}, ("ethics", "safety", "illicit")),
        ({"absent", "reduced"}, ("medical advice", "disclaimer")),
    ],
    "jailbroken_to_base": [
        ({"shared_active"}, ("bomb", "explosive")),
        ({"absent", "reduced"}, ("instruction", "answer", "beginning")),
        ({"absent", "reduced"}, ("safety",)),
        ({"absent", "reduced"}, ("greeting", "confirmation", "initialization")),
    ],
}


@dataclass(frozen=True)
class Feature:
    direction: str
    node_id: str
    layer: int
    pos: int
    feature: int
    label: str
    outcome: str
    source_activation: float
    comparison_activation: float
    activation_ratio_abs: float
    source_influence: float | None
    comparison_token: str

    @property
    def is_region(self) -> bool:
        return self.layer in REGION_LAYERS and self.pos in REGION_POSITIONS

    @property
    def is_degraded(self) -> bool:
        return self.outcome in DEGRADED_OUTCOMES

    @property
    def size_metric(self) -> float:
        if self.source_influence is not None and self.source_influence > 0:
            return self.source_influence
        return abs(self.source_activation)


def escape_token(token: str) -> str:
    """Render whitespace tokens as printable literals for axis labels."""
    if token == "\n":
        return r"\n"
    if token == "\n\n":
        return r"\n\n"
    if "\n" in token:
        return token.replace("\n", r"\n")
    stripped = token.strip()
    return stripped or repr(token)


def as_float(raw: str) -> float | None:
    if raw == "":
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def load_features(path: Path) -> list[Feature]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    features: list[Feature] = []
    for row in rows:
        features.append(
            Feature(
                direction=row["direction"],
                node_id=row["node_id"],
                layer=int(row["layer"]),
                pos=int(row["pos"]),
                feature=int(row["feature"]),
                label=row["label"],
                outcome=row["outcome"],
                source_activation=float(row["source_activation"]),
                comparison_activation=float(row["comparison_activation"]),
                activation_ratio_abs=float(row["activation_ratio_abs"]),
                source_influence=as_float(row.get("source_influence", "")),
                comparison_token=row["comparison_token"],
            )
        )
    return features


def position_tokens(feature_sets: list[list[Feature]]) -> dict[int, str]:
    tokens: dict[int, str] = {}
    for features in feature_sets:
        for feature in features:
            tokens.setdefault(feature.pos, feature.comparison_token)
    return tokens


def grouped_by_cell(features: list[Feature]) -> dict[tuple[int, int], list[Feature]]:
    grouped: dict[tuple[int, int], list[Feature]] = {}
    for feature in features:
        grouped.setdefault((feature.layer, feature.pos), []).append(feature)
    for cell_features in grouped.values():
        cell_features.sort(key=lambda item: (item.outcome == "shared_active", -item.size_metric))
    return grouped


def dot_sizes(features: list[Feature]) -> dict[str, float]:
    metrics = sorted(feature.size_metric for feature in features if feature.size_metric > 0)
    if not metrics:
        return {feature.node_id: 16.0 for feature in features}
    hi = metrics[min(len(metrics) - 1, int(0.95 * (len(metrics) - 1)))]
    lo = metrics[0]
    span = max(hi - lo, 1e-9)
    sizes: dict[str, float] = {}
    for feature in features:
        scaled = min(max((feature.size_metric - lo) / span, 0.0), 1.0)
        sizes[feature.node_id] = 8.0 + 58.0 * math.sqrt(scaled)
    return sizes


def cell_dot_positions(
    *,
    n_features: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    seed: int,
) -> list[tuple[float, float]]:
    x_margin = 0.12 * (right - left)
    y_margin = 0.18 * (top - bottom)
    inner_left = left + x_margin
    inner_right = right - x_margin
    inner_bottom = bottom + y_margin
    inner_top = top - y_margin

    if n_features > 12:
        aspect = max((inner_right - inner_left) / max(inner_top - inner_bottom, 1e-9), 0.5)
        n_cols = max(2, math.ceil(math.sqrt(n_features * aspect)))
        n_rows = math.ceil(n_features / n_cols)
        points: list[tuple[float, float]] = []
        for idx in range(n_features):
            row = idx // n_cols
            col = idx % n_cols
            x = inner_left + (col + 0.5) * (inner_right - inner_left) / n_cols
            y = inner_top - (row + 0.5) * (inner_top - inner_bottom) / n_rows
            points.append((x, y))
        return points

    rng = random.Random(seed)
    return [
        (
            rng.uniform(inner_left, inner_right),
            rng.uniform(inner_bottom, inner_top),
        )
        for _ in range(n_features)
    ]


def scatter_panel(
    ax: plt.Axes,
    *,
    features: list[Feature],
    positions: list[int],
    tokens: dict[int, str],
    title: str,
) -> dict[str, tuple[float, float]]:
    x_edges = [float(index) for index in range(len(positions) + 1)]
    y_edges = [float(index) for index in range(len(LAYERS) + 1)]
    pos_to_index = {pos: index for index, pos in enumerate(positions)}
    layer_to_index = {layer: index for index, layer in enumerate(LAYERS)}

    ax.set_xlim(x_edges[0] - 1.15, x_edges[-1] + 1.15)
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_axisbelow(True)

    for edge in x_edges:
        ax.axvline(edge, color="0.88", lw=0.55, zorder=0)
    for edge in y_edges:
        ax.axhline(edge, color="0.88", lw=0.55, zorder=0)

    ax.set_yticks([layer_to_index[layer] + 0.5 for layer in LAYERS])
    ax.set_yticklabels([f"L{layer}" for layer in LAYERS], fontsize=8)
    ax.set_xticks([pos_to_index[pos] + 0.5 for pos in positions])
    ax.set_xticklabels(
        [f"{escape_token(tokens[pos])}\n{pos}" for pos in positions],
        fontsize=6.3,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        linespacing=0.95,
    )
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    sizes = dot_sizes(features)
    locations: dict[str, tuple[float, float]] = {}
    for (layer, pos), cell_features in grouped_by_cell(features).items():
        col = pos_to_index[pos]
        row = layer_to_index[layer]
        points = cell_dot_positions(
            n_features=len(cell_features),
            left=x_edges[col],
            right=x_edges[col + 1],
            bottom=y_edges[row],
            top=y_edges[row + 1],
            seed=layer * 1000 + pos,
        )
        for feature, (x, y) in zip(cell_features, points, strict=True):
            style = OUTCOME_STYLES[feature.outcome]
            ax.scatter(
                x,
                y,
                s=sizes[feature.node_id],
                marker="o",
                facecolor=style["dot"],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.86,
                zorder=3 if feature.is_degraded else 2,
            )
            locations[feature.node_id] = (x, y)
    return locations


def choose_callouts(features: list[Feature]) -> list[Feature]:
    direction = features[0].direction if features else ""
    selected: list[Feature] = []
    used_ids: set[str] = set()

    for outcomes, keywords in CALLOUT_KEYWORDS.get(direction, []):
        candidates = [
            feature
            for feature in features
            if feature.outcome in outcomes
            and feature.node_id not in used_ids
            and any(keyword in feature.label.lower() for keyword in keywords)
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                not item.is_region,
                item.outcome == "shared_active",
                -item.size_metric,
            )
        )
        selected.append(candidates[0])
        used_ids.add(candidates[0].node_id)

    if len(selected) < 3:
        fallback = sorted(
            features,
            key=lambda item: (
                item.outcome == "shared_active",
                not item.is_region,
                -item.size_metric,
            ),
        )
        for feature in fallback:
            if feature.node_id in used_ids:
                continue
            selected.append(feature)
            used_ids.add(feature.node_id)
            if len(selected) >= 3:
                break
    return selected[:5]


def wrapped_callout(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=22, break_long_words=False)) or "(unlabelled)"


def add_callouts(
    ax: plt.Axes,
    *,
    features: list[Feature],
    locations: dict[str, tuple[float, float]],
    side: str,
) -> None:
    callouts = [feature for feature in choose_callouts(features) if feature.node_id in locations]
    if not callouts:
        return
    y_slots = [3.72, 3.18, 2.64, 2.10, 1.56]
    if side == "left":
        x_text = -0.42
        ha = "right"
        rad = -0.12
    else:
        x_text = len({feature.pos for feature in features}) + 0.42
        ha = "left"
        rad = 0.12

    for feature, y_text in zip(callouts, y_slots, strict=False):
        x, y = locations[feature.node_id]
        style = OUTCOME_STYLES[feature.outcome]
        ax.annotate(
            wrapped_callout(feature.label),
            xy=(x, y),
            xytext=(x_text, y_text),
            ha=ha,
            va="center",
            fontsize=6.8,
            color="0.18",
            arrowprops={
                "arrowstyle": "-",
                "color": style["edge"],
                "lw": 0.7,
                "connectionstyle": f"arc3,rad={rad}",
                "shrinkA": 1,
                "shrinkB": 2,
            },
            bbox={
                "boxstyle": "round,pad=0.16,rounding_size=0.08",
                "facecolor": "white",
                "edgecolor": style["edge"],
                "linewidth": 0.55,
                "alpha": 0.92,
            },
            zorder=6,
        )


def direction_title(direction: str, n_features: int) -> str:
    if direction == "base_to_jailbroken":
        return f"A  Base graph features measured in the abliterated model (n={n_features})"
    if direction == "jailbroken_to_base":
        return f"B  Abliterated graph features measured in the base model (n={n_features})"
    return f"{direction or 'comparison'} (n={n_features})"


def build_figure(
    base_to_jailbroken: list[Feature], jailbroken_to_base: list[Feature]
) -> plt.Figure:
    feature_sets = [base_to_jailbroken, jailbroken_to_base]
    positions = sorted({feature.pos for features in feature_sets for feature in features})
    tokens = position_tokens(feature_sets)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.8), sharex=True)
    for index, (ax, features) in enumerate(zip(axes, feature_sets, strict=True)):
        locations = scatter_panel(
            ax,
            features=features,
            positions=positions,
            tokens=tokens,
            title=direction_title(features[0].direction if features else "", len(features)),
        )
        add_callouts(
            ax,
            features=features,
            locations=locations,
            side="left" if index == 0 else "right",
        )
        ax.tick_params(axis="x", labelbottom=True, pad=2)

    axes[-1].set_xlabel("Prompt token text and position", fontsize=8, labelpad=8)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=style["dot"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.2,
            label=style["label"],
        )
        for style in OUTCOME_STYLES.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=8,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.text(0.03, 0.53, "Layer", rotation=90, ha="center", va="center", fontsize=8)
    fig.subplots_adjust(top=0.94, bottom=0.115, left=0.125, right=0.875, hspace=0.34)
    return fig


def fisher_p_value(table: list[list[int]]) -> dict[str, float] | None:
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return None
    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    return {"odds_ratio": float(odds_ratio), "p_value_greater": float(p_value)}


def summarise_direction(features: list[Feature]) -> dict[str, Any]:
    outcome_counts = dict(Counter(feature.outcome for feature in features))
    region_degraded = sum(feature.is_region and feature.is_degraded for feature in features)
    region_shared = sum(
        feature.is_region and feature.outcome == "shared_active" for feature in features
    )
    other_degraded = sum((not feature.is_region) and feature.is_degraded for feature in features)
    other_shared = sum(
        (not feature.is_region) and feature.outcome == "shared_active" for feature in features
    )
    table = [[region_degraded, region_shared], [other_degraded, other_shared]]
    region_total = region_degraded + region_shared
    other_total = other_degraded + other_shared
    other_outcomes = sum(
        feature.outcome not in DEGRADED_OUTCOMES | {"shared_active"} for feature in features
    )
    summary = {
        "n_features": len(features),
        "outcome_counts": outcome_counts,
        "region_definition": {
            "layers": sorted(REGION_LAYERS),
            "positions": sorted(REGION_POSITIONS),
        },
        "region_table": {
            "rows": ["late_region", "elsewhere"],
            "columns": ["absent_or_reduced", "shared_active"],
            "counts": table,
            "proportions_absent_or_reduced": {
                "late_region": region_degraded / region_total if region_total else None,
                "elsewhere": other_degraded / other_total if other_total else None,
            },
            "other_outcomes_excluded": other_outcomes,
        },
    }
    fisher = fisher_p_value(table)
    if fisher is not None:
        summary["region_table"]["fisher_exact"] = fisher
    return summary


def build_summary(
    base_to_jailbroken: list[Feature], jailbroken_to_base: list[Feature]
) -> dict[str, Any]:
    return {
        "base_to_jailbroken": summarise_direction(base_to_jailbroken),
        "jailbroken_to_base": summarise_direction(jailbroken_to_base),
        "notes": [
            "The tested region was derived from the pinned-feature analysis.",
            "Feature rows are not independent because features repeat across positions and related features are correlated.",
        ],
    }


def print_summary(summary: dict[str, Any]) -> None:
    for direction, item in summary.items():
        if direction == "notes":
            continue
        table = item["region_table"]
        proportions = table["proportions_absent_or_reduced"]
        fisher = table.get("fisher_exact")
        print(f"{direction}:")
        print(f"  outcome_counts={item['outcome_counts']}")
        print(
            "  late_region [absent_or_reduced, shared_active]="
            f"{table['counts'][0]} "
            f"p_degraded={proportions['late_region']:.3f}"
        )
        print(
            "  elsewhere   [absent_or_reduced, shared_active]="
            f"{table['counts'][1]} "
            f"p_degraded={proportions['elsewhere']:.3f}"
        )
        if fisher is not None:
            print(
                "  fisher_exact_greater="
                f"odds_ratio={fisher['odds_ratio']:.3g}, "
                f"p={fisher['p_value_greater']:.3g}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_to_jailbroken_csv", type=Path)
    parser.add_argument("jailbroken_to_base_csv", type=Path)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("report/figures/feature_fate_map_unsupervised.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("report/figures/feature_fate_map_unsupervised.png"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="JSON summary path. Defaults to the PNG path with '_summary.json' appended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_to_jailbroken = load_features(args.base_to_jailbroken_csv)
    jailbroken_to_base = load_features(args.jailbroken_to_base_csv)

    fig = build_figure(base_to_jailbroken, jailbroken_to_base)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    fig.savefig(args.output_png, dpi=300)
    print(f"wrote {args.output_pdf}")
    print(f"wrote {args.output_png}")

    summary = build_summary(base_to_jailbroken, jailbroken_to_base)
    summary_path = args.summary_output
    if summary_path is None:
        summary_path = args.output_png.with_name(f"{args.output_png.stem}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"wrote {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
