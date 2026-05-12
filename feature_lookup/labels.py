"""Helpers for loading per-feature labels from JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LABEL_DIR = Path(__file__).parent.parent / "data" / "feature_labels"


@dataclass(frozen=True, slots=True)
class FeatureLabel:
    layer: int
    feature: int
    label: str
    rationale: str
    max_activation: float | None = None
    n_distinct_prompts: int | None = None
    source: str | None = None


FeatureLabelMap = dict[tuple[int, int], FeatureLabel]


def _parse_record(record: dict[str, Any], *, source: Path, line_no: int) -> FeatureLabel:
    try:
        layer = int(record["layer"])
        feature = int(record["feature"])
        label = str(record["label"]).strip()
        rationale = str(record["rationale"]).strip()
    except KeyError as exc:
        raise ValueError(f"{source}:{line_no}: missing key {exc.args[0]!r}") from exc

    if not label:
        raise ValueError(f"{source}:{line_no}: empty label")

    max_activation = record.get("max_activation")
    n_distinct_prompts = record.get("n_distinct_prompts")
    raw_source = record.get("source")
    record_source = str(raw_source) if isinstance(raw_source, str) and raw_source else None
    return FeatureLabel(
        layer=layer,
        feature=feature,
        label=label,
        rationale=rationale,
        max_activation=float(max_activation) if max_activation is not None else None,
        n_distinct_prompts=(int(n_distinct_prompts) if n_distinct_prompts is not None else None),
        source=record_source,
    )


def load_feature_labels(
    layers: list[int] | tuple[int, ...] | set[int] | None = None,
    labels_dir: Path | str = DEFAULT_LABEL_DIR,
) -> FeatureLabelMap:
    """Load feature labels keyed by ``(layer, feature)``.

    Missing layer files are skipped so attribution runs can still produce graphs
    with fallback labels.
    """

    labels_path = Path(labels_dir)
    if layers is None:
        paths = sorted(labels_path.glob("layer_*.jsonl"))
    else:
        paths = [labels_path / f"layer_{layer}.jsonl" for layer in sorted(layers)]

    out: FeatureLabelMap = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                label = _parse_record(json.loads(line), source=path, line_no=line_no)
                out[(label.layer, label.feature)] = label
    return out


UNLABELLED_PREFIX = "[?] "


def get_feature_label(
    labels: FeatureLabelMap,
    layer: int,
    feature: int,
    *,
    fallback: str | None = None,
) -> str:
    """Return a label for a feature.

    Unlabelled features get a ``[?] L{layer} F{feature}`` fallback so the UI
    can distinguish them from deliberate placeholders at a glance.
    """

    found = labels.get((layer, feature))
    if found is not None:
        return found.label
    return fallback or f"{UNLABELLED_PREFIX}L{layer} F{feature}"
