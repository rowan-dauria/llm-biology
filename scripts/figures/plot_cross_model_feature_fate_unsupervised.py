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
        --output-svg report/figures/feature_fate_map_unsupervised.svg \
        --summary-output llm-biology/data/base_jailbreak_comparison/allnodes_feature_fate_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

LAYERS = [2, 12, 24, 33]
LAYER_HEIGHTS = {2: 0.52, 12: 0.62, 24: 1.0, 33: 1.0}
REGION_LAYERS = {24, 33}
REGION_POSITIONS = {11, 18}
DEGRADED_OUTCOMES = {"absent", "reduced"}
RATIO_MIN = 0.0
RATIO_MAX = 2.0
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "feature_activation_ratio",
    [
        (0.00, "#4b0012"),  # absent
        (0.25, "#f59e00"),  # halved
        (0.50, "#007a3d"),  # unchanged
        (1.00, "#0052ff"),  # stronger
    ],
)
RATIO_NORM = Normalize(vmin=RATIO_MIN, vmax=RATIO_MAX)
DOT_SIZE = 6.0

CALLOUT_KEYWORDS = {
    "base_to_jailbroken": [
        ({"shared_active"}, ("bomb", "explosive")),
        ({"absent", "reduced"}, ("harmful content refusal",)),
        ({"absent", "reduced"}, ("negation",)),
        ({"absent", "reduced"}, ("ethics", "safety", "illicit")),
    ],
    "jailbroken_to_base": [
        ({"shared_active"}, ("bomb", "explosive")),
        ({"absent", "reduced"}, ("certainly", "sure")),
        ({"absent", "reduced"}, ("answer",)),
        ({"absent", "reduced"}, ("craft project", "instruction")),
    ],
}

PINNED_PANEL_NAMES = {
    "base_to_jailbroken": "refusal-bomb-base-to-jailbroken.json",
    "jailbroken_to_base": "refusal-bomb-jailbroken-to-base.json",
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
    human_label: str | None = None

    @property
    def callout_label(self) -> str:
        return self.human_label or self.label

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

    @property
    def activation_ratio(self) -> float:
        if self.source_activation == 0:
            return 0.0 if self.comparison_activation == 0 else RATIO_MAX
        return abs(self.comparison_activation) / abs(self.source_activation)

    @property
    def clipped_activation_ratio(self) -> float:
        return min(max(self.activation_ratio, RATIO_MIN), RATIO_MAX)

    @property
    def colour(self) -> tuple[float, float, float, float]:
        return RATIO_CMAP(RATIO_NORM(self.clipped_activation_ratio))


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


def load_human_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features", []) if isinstance(payload, dict) else []
    labels: dict[str, str] = {}
    for feature in features:
        if isinstance(feature, dict) and feature.get("node_id") and feature.get("label"):
            labels[str(feature["node_id"])] = str(feature["label"])
    return labels


def default_human_label_path(csv_path: Path, direction: str) -> Path | None:
    name = PINNED_PANEL_NAMES.get(direction)
    if name is None:
        return None
    return csv_path.parent / "panels" / name


def load_features(path: Path, human_labels: dict[str, str] | None = None) -> list[Feature]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    human_labels = human_labels or {}
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
                human_label=human_labels.get(row["node_id"]),
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


def layer_edges() -> list[float]:
    edges = [0.0]
    for layer in LAYERS:
        edges.append(edges[-1] + LAYER_HEIGHTS[layer])
    return edges


def dot_sizes(features: list[Feature]) -> dict[str, float]:
    return {feature.node_id: DOT_SIZE for feature in features}


def choose_grid_shape(
    *,
    n_features: int,
    width: float,
    height: float,
    x_to_y_display_scale: float,
) -> tuple[int, int]:
    best_cols = 1
    best_rows = n_features
    best_score = -1.0
    for n_cols in range(1, n_features + 1):
        n_rows = math.ceil(n_features / n_cols)
        x_spacing = (width * x_to_y_display_scale) / n_cols
        y_spacing = height / n_rows
        score = min(x_spacing, y_spacing)
        if score > best_score:
            best_cols = n_cols
            best_rows = n_rows
            best_score = score
    return best_cols, best_rows


def cell_dot_positions(
    *,
    n_features: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    x_to_y_display_scale: float,
) -> list[tuple[float, float]]:
    x_margin = 0.08 * (right - left)
    y_margin = 0.10 * (top - bottom)
    inner_left = left + x_margin
    inner_right = right - x_margin
    inner_bottom = bottom + y_margin
    inner_top = top - y_margin

    if n_features == 1:
        return [((left + right) / 2, (bottom + top) / 2)]

    n_cols, n_rows = choose_grid_shape(
        n_features=n_features,
        width=inner_right - inner_left,
        height=inner_top - inner_bottom,
        x_to_y_display_scale=x_to_y_display_scale,
    )
    points: list[tuple[float, float]] = []
    for idx in range(n_features):
        row = idx // n_cols
        col = idx % n_cols
        row_start = row * n_cols
        row_count = min(n_cols, n_features - row_start)
        row_width = row_count * (inner_right - inner_left) / n_cols
        row_left = (inner_left + inner_right - row_width) / 2
        x = row_left + (col + 0.5) * (inner_right - inner_left) / n_cols
        y = inner_top - (row + 0.5) * (inner_top - inner_bottom) / n_rows
        points.append((x, y))
    return points


def scatter_panel(
    ax: plt.Axes,
    *,
    features: list[Feature],
    positions: list[int],
    tokens: dict[int, str],
    title: str,
) -> dict[str, tuple[float, float]]:
    x_edges = [float(index) for index in range(len(positions) + 1)]
    y_edges = layer_edges()
    pos_to_index = {pos: index for index, pos in enumerate(positions)}
    layer_to_index = {layer: index for index, layer in enumerate(LAYERS)}

    ax.set_xlim(x_edges[0] - 1.15, x_edges[-1] + 1.15)
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_axisbelow(True)
    fig_width, fig_height = ax.figure.get_size_inches()
    bbox = ax.get_position()
    x_units = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_units = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_to_y_display_scale = (fig_width * bbox.width / x_units) / (fig_height * bbox.height / y_units)

    for edge in x_edges:
        ax.vlines(edge, y_edges[0], y_edges[-1], color="0.88", lw=0.55, zorder=0)
    for edge in y_edges:
        ax.hlines(edge, x_edges[0], x_edges[-1], color="0.88", lw=0.55, zorder=0)

    ax.set_yticks(
        [
            (y_edges[layer_to_index[layer]] + y_edges[layer_to_index[layer] + 1]) / 2
            for layer in LAYERS
        ]
    )
    ax.set_yticklabels([f"L{layer}" for layer in LAYERS], fontsize=8)
    ax.set_xticks([pos_to_index[pos] + 0.5 for pos in positions])
    ax.set_xticklabels(
        [escape_token(tokens[pos]) for pos in positions],
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
            x_to_y_display_scale=x_to_y_display_scale,
        )
        for feature, (x, y) in zip(cell_features, points, strict=True):
            ax.scatter(
                x,
                y,
                s=sizes[feature.node_id],
                marker="o",
                facecolor=feature.colour,
                edgecolor="white",
                linewidth=0.18,
                alpha=0.90,
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
            and any(keyword in feature.callout_label.lower() for keyword in keywords)
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
    return "\n".join(textwrap.wrap(label, width=18, break_long_words=False)) or "(unlabelled)"


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
    y_top = layer_edges()[-1]
    y_slots = [0.94 * y_top, 0.78 * y_top, 0.62 * y_top, 0.46 * y_top, 0.30 * y_top]
    if side == "left":
        x_text = -0.42
        ha = "right"
        rad = -0.12
    else:
        x_text = len({feature.pos for feature in features}) + 0.12
        ha = "left"
        rad = 0.12

    for feature, y_text in zip(callouts, y_slots, strict=False):
        x, y = locations[feature.node_id]
        colour = feature.colour
        ax.annotate(
            wrapped_callout(feature.callout_label),
            xy=(x, y),
            xytext=(x_text, y_text),
            ha=ha,
            va="center",
            fontsize=6.8,
            color="0.18",
            arrowprops={
                "arrowstyle": "-",
                "color": colour,
                "lw": 0.7,
                "connectionstyle": f"arc3,rad={rad}",
                "shrinkA": 1,
                "shrinkB": 2,
            },
            bbox={
                "boxstyle": "round,pad=0.16,rounding_size=0.08",
                "facecolor": "white",
                "edgecolor": colour,
                "linewidth": 0.55,
                "alpha": 0.92,
            },
            zorder=6,
        )


def direction_title(direction: str, n_features: int) -> str:
    if direction == "base_to_jailbroken":
        return f"A  Base graph features in the abliterated model (n={n_features})"
    if direction == "jailbroken_to_base":
        return f"B  Abliterated graph features in the base model (n={n_features})"
    return f"{direction or 'comparison'} (n={n_features})"


def build_figure(
    base_to_jailbroken: list[Feature], jailbroken_to_base: list[Feature]
) -> plt.Figure:
    feature_sets = [base_to_jailbroken, jailbroken_to_base]
    positions = sorted({feature.pos for features in feature_sets for feature in features})
    tokens = position_tokens(feature_sets)

    fig, axes = plt.subplots(2, 1, figsize=(7.3, 7.1), sharex=True)
    fig.subplots_adjust(top=0.955, bottom=0.115, left=0.125, right=0.76, hspace=0.32)
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

    colourbar = fig.colorbar(
        ScalarMappable(norm=RATIO_NORM, cmap=RATIO_CMAP),
        ax=axes,
        orientation="vertical",
        fraction=0.03,
        pad=0.15,
        aspect=28,
    )
    colourbar.set_ticks([0, 0.5, 1, 2])
    colourbar.set_ticklabels(["0\nabsent", "0.5\nhalved", "1\nunchanged", "2+\nstronger"])
    colourbar.ax.tick_params(labelsize=7, length=2)
    colourbar.ax.set_title(
        "|comparison|\n/ |source|",
        fontsize=7,
        pad=6,
    )
    fig.text(0.03, 0.53, "Layer", rotation=90, ha="center", va="center", fontsize=8)
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
        "--base-human-labels",
        type=Path,
        default=None,
        help="Pinned-panel JSON whose labels override base-to-jailbroken callout text.",
    )
    parser.add_argument(
        "--jailbroken-human-labels",
        type=Path,
        default=None,
        help="Pinned-panel JSON whose labels override jailbroken-to-base callout text.",
    )
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
        "--output-svg",
        type=Path,
        default=None,
        help="Optional SVG output path. Omit to avoid overwriting an existing editable SVG.",
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
    base_label_path = args.base_human_labels or default_human_label_path(
        args.base_to_jailbroken_csv, "base_to_jailbroken"
    )
    jailbroken_label_path = args.jailbroken_human_labels or default_human_label_path(
        args.jailbroken_to_base_csv, "jailbroken_to_base"
    )
    base_to_jailbroken = load_features(
        args.base_to_jailbroken_csv,
        load_human_labels(base_label_path) if base_label_path is not None else None,
    )
    jailbroken_to_base = load_features(
        args.jailbroken_to_base_csv,
        load_human_labels(jailbroken_label_path) if jailbroken_label_path is not None else None,
    )

    fig = build_figure(base_to_jailbroken, jailbroken_to_base)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    fig.savefig(args.output_png, dpi=300)
    print(f"wrote {args.output_pdf}")
    print(f"wrote {args.output_png}")
    if args.output_svg is not None:
        args.output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_svg)
        print(f"wrote {args.output_svg}")

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
