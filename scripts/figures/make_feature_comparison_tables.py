#!/usr/bin/env python3
"""Generate LaTeX table fragments for cross-model feature comparisons.

Inputs are the two CSV files produced by re-measuring pinned attribution-graph
features from one model in the other model. The emitted fragments contain only
table body rows, grouped by outcome and sorted by source activation, so the
report appendix can keep captions and table layout in LaTeX while the numerical
values remain generated from the CSV source of truth.

Command used for the report tables:

    llm-biology/venv/bin/python3 \
        llm-biology/scripts/figures/make_feature_comparison_tables.py \
        llm-biology/data/base_jailbreak_comparison/2026-07-01-20-56-31__base_to_jailbroken__vs-qwen3-4b-heretic-trial114-merged.csv \
        llm-biology/data/base_jailbreak_comparison/2026-07-01-20-57-21__jailbroken_to_base__vs-qwen3-4b.csv \
        --output-dir report/tables
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

OUTCOME_ORDER = ("absent", "reduced", "shared_active")
OUTCOME_LABELS = {
    "absent": "absent",
    "reduced": "reduced",
    "shared_active": "shared",
}


@dataclass(frozen=True)
class FeatureRow:
    direction: str
    node_id: str
    feature_id: str
    layer: int
    pos: int
    label: str
    token: str
    source_activation: float
    comparison_activation: float
    ratio: float
    outcome: str


def latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = "".join(replacements.get(char, char) for char in value)
    return text.replace('"', "''").replace("''", "``", 1) if '"' in text else text


def latex_label(value: str) -> str:
    # Convert straight double quotes pairwise to LaTeX opening/closing quotes.
    parts = value.split('"')
    if len(parts) == 1:
        return latex_text(value)
    output: list[str] = []
    for index, part in enumerate(parts):
        output.append(latex_text(part))
        if index < len(parts) - 1:
            output.append("``" if index % 2 == 0 else "''")
    return "".join(output)


def latex_texttt(value: str) -> str:
    return r"\texttt{" + latex_text(value) + "}"


def token_display(token: str) -> str:
    if token == "\n":
        return latex_texttt(r"\n")
    if token == "\n\n":
        return latex_texttt(r"\n\n")
    stripped = token.strip()
    return latex_texttt(stripped) if stripped else latex_texttt("")


def fmt_number(value: float) -> str:
    if abs(value) < 0.005:
        return "0"
    return f"{value:.2f}"


def feature_id_from_node(node_id: str) -> str:
    parts = node_id.split("_")
    if len(parts) >= 2:
        return parts[1]
    return node_id


def load_rows(path: Path) -> list[FeatureRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        FeatureRow(
            direction=row["direction"],
            node_id=row["node_id"],
            feature_id=feature_id_from_node(row["node_id"]),
            layer=int(row["layer"]),
            pos=int(row["pos"]),
            label=row["label"],
            token=row["comparison_token"],
            source_activation=float(row["source_activation"]),
            comparison_activation=float(row["comparison_activation"]),
            ratio=float(row["activation_ratio_abs"]),
            outcome=row["outcome"],
        )
        for row in rows
    ]


def model_activations(row: FeatureRow) -> tuple[float, float]:
    if row.direction == "base_to_jailbroken":
        return row.source_activation, row.comparison_activation
    if row.direction == "jailbroken_to_base":
        return row.comparison_activation, row.source_activation
    raise ValueError(f"unknown direction: {row.direction}")


def sorted_rows(rows: list[FeatureRow]) -> list[FeatureRow]:
    outcome_rank = {outcome: rank for rank, outcome in enumerate(OUTCOME_ORDER)}
    return sorted(rows, key=lambda row: (outcome_rank[row.outcome], -row.source_activation))


def row_to_latex(row: FeatureRow) -> str:
    base_activation, ablit_activation = model_activations(row)
    columns = [
        latex_label(row.label),
        latex_texttt(row.feature_id),
        str(row.layer),
        str(row.pos),
        token_display(row.token),
        fmt_number(base_activation),
        fmt_number(ablit_activation),
        fmt_number(row.ratio),
        OUTCOME_LABELS[row.outcome],
    ]
    return " & ".join(columns) + r" \\"


def fragment(rows: list[FeatureRow]) -> str:
    lines: list[str] = []
    current_outcome: str | None = None
    for row in sorted_rows(rows):
        if current_outcome is not None and row.outcome != current_outcome:
            lines.append(r"\midrule")
        current_outcome = row.outcome
        lines.append(row_to_latex(row))
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def output_name(rows: list[FeatureRow]) -> str:
    directions = {row.direction for row in rows}
    if len(directions) != 1:
        raise ValueError(f"expected one direction per CSV, got {sorted(directions)}")
    direction = directions.pop()
    if direction == "base_to_jailbroken":
        return "feature_comparison_base_to_jailbroken.tex"
    if direction == "jailbroken_to_base":
        return "feature_comparison_jailbroken_to_base.tex"
    raise ValueError(f"unknown direction: {direction}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_to_jailbroken_csv", type=Path)
    parser.add_argument("jailbroken_to_base_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("report/tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.base_to_jailbroken_csv, args.jailbroken_to_base_csv):
        rows = load_rows(path)
        output_path = args.output_dir / output_name(rows)
        output_path.write_text(fragment(rows), encoding="utf-8")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
