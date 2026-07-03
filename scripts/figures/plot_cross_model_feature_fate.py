#!/usr/bin/env python3
"""Plot cross-model fates for pinned attribution-graph features.

Inputs are the two CSV files produced by re-measuring human-pruned graph
features from one model in the other model. Each row must include the feature's
``layer``, token ``pos``, human ``label``, ``outcome``, source/comparison
activation values, absolute activation ratio, and comparison token text.

The output is a two-panel feature-fate map. Both panels use the same categorical
token-position geometry, with variable-width columns so that the rounded feature
chips remain legible in report-scale PDF output.

Command used for the report figure:

    llm-biology/venv/bin/python3 \
        llm-biology/scripts/figures/plot_cross_model_feature_fate.py \
        llm-biology/data/base_jailbreak_comparison/2026-07-01-20-56-31__base_to_jailbroken__vs-qwen3-4b-heretic-trial114-merged.csv \
        llm-biology/data/base_jailbreak_comparison/2026-07-01-20-57-21__jailbroken_to_base__vs-qwen3-4b.csv \
        --output-pdf report/figures/feature_fate_map.pdf \
        --output-png report/figures/feature_fate_map.png
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

LAYERS = [2, 12, 24, 33]

OUTCOME_STYLES = {
    "shared_active": {
        "label": "shared active",
        "face": "#dff1df",
        "edge": "#2c6b38",
        "text": "#1f5229",
    },
    "reduced": {
        "label": "reduced",
        "face": "#fde7b6",
        "edge": "#b85e00",
        "text": "#743c00",
    },
    "absent": {
        "label": "absent",
        "face": "#f9d6d5",
        "edge": "#a53232",
        "text": "#6f1f1f",
    },
}


@dataclass(frozen=True)
class Feature:
    layer: int
    pos: int
    label: str
    outcome: str
    source_activation: float
    comparison_activation: float
    activation_ratio_abs: float
    comparison_token: str
    chip_text: str


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


def wrapped_label(label: str, *, pos: int) -> str:
    """Wrap only labels that would otherwise force very wide columns."""
    width = 20 if pos in {8, 9, 11} else 28
    if len(label) <= width:
        return label
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def chip_text(row: dict[str, str]) -> str:
    label = wrapped_label(row["label"], pos=int(row["pos"]))
    outcome = row["outcome"]
    source = float(row["source_activation"])
    comparison = float(row["comparison_activation"])
    ratio = float(row["activation_ratio_abs"])

    if outcome == "absent":
        return f"{label} {source:.1f}→0"
    if outcome == "reduced":
        return f"{label} {source:.1f}→{comparison:.1f}"
    if ratio < 0.85 or ratio > 1.15:
        return f"{label} ×{ratio:.2f}"
    return label


def load_features(path: Path) -> list[Feature]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    features: list[Feature] = []
    for row in rows:
        features.append(
            Feature(
                layer=int(row["layer"]),
                pos=int(row["pos"]),
                label=row["label"],
                outcome=row["outcome"],
                source_activation=float(row["source_activation"]),
                comparison_activation=float(row["comparison_activation"]),
                activation_ratio_abs=float(row["activation_ratio_abs"]),
                comparison_token=row["comparison_token"],
                chip_text=chip_text(row),
            )
        )
    return features


def position_tokens(feature_sets: list[list[Feature]]) -> dict[int, str]:
    tokens: dict[int, str] = {}
    for features in feature_sets:
        for feature in features:
            tokens.setdefault(feature.pos, feature.comparison_token)
    return tokens


def column_widths(positions: list[int], feature_sets: list[list[Feature]]) -> dict[int, float]:
    max_chars = dict.fromkeys(positions, 8)
    for features in feature_sets:
        for feature in features:
            longest_line = max(len(line) for line in feature.chip_text.splitlines())
            max_chars[feature.pos] = max(max_chars[feature.pos], longest_line)

    widths: dict[int, float] = {}
    for pos in positions:
        # Character-driven widths keep request-span columns compact while giving
        # the late newline columns enough room for activation annotations.
        widths[pos] = min(3.45, max(1.05, 0.084 * max_chars[pos] + 0.42))
    return widths


def grouped_by_cell(features: list[Feature]) -> dict[tuple[int, int], list[Feature]]:
    grouped: dict[tuple[int, int], list[Feature]] = defaultdict(list)
    for feature in features:
        grouped[(feature.layer, feature.pos)].append(feature)
    for values in grouped.values():
        values.sort(key=lambda item: item.source_activation, reverse=True)
    return grouped


def draw_panel(
    ax: plt.Axes,
    *,
    features: list[Feature],
    positions: list[int],
    tokens: dict[int, str],
    widths: dict[int, float],
    title: str,
) -> None:
    edges = [0.0]
    for pos in positions:
        edges.append(edges[-1] + widths[pos])
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(positions))]

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, len(LAYERS))
    ax.set_axisbelow(True)

    for edge in edges:
        ax.axvline(edge, color="0.84", lw=0.8, zorder=0)
    for row in range(len(LAYERS) + 1):
        ax.axhline(row, color="0.84", lw=0.8, zorder=0)

    ax.set_yticks([idx + 0.5 for idx in range(len(LAYERS))])
    ax.set_yticklabels([f"L{layer}" for layer in LAYERS], fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(centres)
    ax.set_xticklabels(
        [f"{escape_token(tokens[pos])}\npos {pos}" for pos in positions],
        fontsize=9,
        linespacing=1.05,
    )
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    grouped = grouped_by_cell(features)
    layer_to_row = {layer: idx for idx, layer in enumerate(LAYERS)}
    pos_to_index = {pos: idx for idx, pos in enumerate(positions)}

    for (layer, pos), cell_features in grouped.items():
        row = layer_to_row[layer]
        col = pos_to_index[pos]
        left, right = edges[col], edges[col + 1]
        x = (left + right) / 2
        n_features = len(cell_features)
        slot = 0.86 / n_features
        font_size = 9.3 if n_features < 6 else 9.0

        for idx, feature in enumerate(cell_features):
            y = row + 0.93 - (idx + 0.5) * slot
            style = OUTCOME_STYLES[feature.outcome]
            ax.text(
                x,
                y,
                feature.chip_text,
                ha="center",
                va="center",
                fontsize=font_size,
                color=style["text"],
                linespacing=0.92,
                bbox={
                    "boxstyle": "round,pad=0.18,rounding_size=0.12",
                    "facecolor": style["face"],
                    "edgecolor": style["edge"],
                    "linewidth": 0.9,
                },
                zorder=3,
            )


def build_figure(
    base_to_jailbroken: list[Feature], jailbroken_to_base: list[Feature]
) -> plt.Figure:
    feature_sets = [base_to_jailbroken, jailbroken_to_base]
    positions = sorted({feature.pos for features in feature_sets for feature in features})
    tokens = position_tokens(feature_sets)
    widths = column_widths(positions, feature_sets)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 12.6), sharex=True)
    titles = [
        f"A  Base-model features measured in the abliterated model (n={len(base_to_jailbroken)})",
        f"B  Abliterated-model features measured in the base model (n={len(jailbroken_to_base)})",
    ]

    for ax, features, title in zip(axes, feature_sets, titles, strict=True):
        draw_panel(
            ax,
            features=features,
            positions=positions,
            tokens=tokens,
            widths=widths,
            title=title,
        )
        ax.tick_params(axis="x", labelbottom=True, pad=5)

    axes[-1].set_xlabel("Prompt token position, categorically spaced", fontsize=10, labelpad=10)

    handles = [
        Patch(
            facecolor=style["face"],
            edgecolor=style["edge"],
            label=style["label"],
            linewidth=1.0,
        )
        for style in OUTCOME_STYLES.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.997),
        fontsize=10,
    )
    fig.subplots_adjust(top=0.945, bottom=0.065, left=0.07, right=0.985, hspace=0.28)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_to_jailbroken_csv",
        type=Path,
        help="base-graph features re-measured in the abliterated model",
    )
    parser.add_argument(
        "jailbroken_to_base_csv",
        type=Path,
        help="abliterated-graph features re-measured in the base model",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("report/figures/feature_fate_map.pdf"),
        help="output vector PDF path",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("report/figures/feature_fate_map.png"),
        help="output PNG path for inspection",
    )
    args = parser.parse_args()

    base_to_jailbroken = load_features(args.base_to_jailbroken_csv)
    jailbroken_to_base = load_features(args.jailbroken_to_base_csv)
    fig = build_figure(base_to_jailbroken, jailbroken_to_base)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf)
    fig.savefig(args.output_png, dpi=240)
    print(f"wrote {args.output_pdf}")
    print(f"wrote {args.output_png}")


if __name__ == "__main__":
    main()
