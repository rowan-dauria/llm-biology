"""Pick unlabelled ``(layer, feature)`` targets from an attribution graph.

Reads a frontend graph JSON written by ``biology_server.circuit_graph_export.export_circuit_graph``,
joins against the per-layer labels store, and ranks the transcoder feature nodes
that still need attention — those absent from the store, or whose baked-in
``clerp`` is still a ``[?] ...`` placeholder — by ``alpha * abs(target-logit
effect) + (1-alpha) * weighted-degree centrality`` (each component min-max
normalised within the candidate set).

Pure-logic module: no torch, no transformers, no network.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .labels import UNLABELLED_PREFIX, FeatureLabelMap, load_feature_labels
except ImportError:
    from labels import (  # type: ignore[no-redef]
        UNLABELLED_PREFIX,
        FeatureLabelMap,
        load_feature_labels,
    )

TRANSCODER_FEATURE_TYPE = "cross layer transcoder"


@dataclass(frozen=True, slots=True)
class GraphTarget:
    layer: int
    feature: int
    target_effect: float
    centrality: float
    score: float
    n_positions: int


def parse_feature_node_id(node_id: str) -> tuple[int, int, int]:
    """Parse ``"<layer>_<feature>_<pos>"`` into ``(layer, feature, pos)``.

    Raises ``ValueError`` for embedding ids (``E_*``) and malformed strings.
    Logit ids parse successfully — the caller is expected to filter by
    ``feature_type`` first.
    """

    if node_id.startswith("E_"):
        raise ValueError(f"embedding node_id: {node_id!r}")
    parts = node_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"unexpected node_id format: {node_id!r}")
    try:
        layer = int(parts[0])
        feature = int(parts[1])
        pos = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"non-integer parts in node_id: {node_id!r}") from exc
    if layer < 0 or feature < 0 or pos < 0:
        raise ValueError(f"negative parts in node_id: {node_id!r}")
    return layer, feature, pos


def load_graph(graph_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ``(transcoder_feature_nodes, non_self_loop_links)`` from graph JSON."""

    with graph_path.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    transcoder_nodes = [
        node
        for node in graph.get("nodes", [])
        if node.get("feature_type") == TRANSCODER_FEATURE_TYPE
    ]
    links = [link for link in graph.get("links", []) if link.get("source") != link.get("target")]
    return transcoder_nodes, links


def compute_centrality(
    feature_keys: Iterable[tuple[int, int]],
    links: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], float]:
    """Weighted degree centrality keyed by ``(layer, feature)``.

    Sums ``abs(link.weight)`` across every link whose source *or* target
    parses to a key in ``feature_keys``. Aggregates across context positions.
    """

    centrality: dict[tuple[int, int], float] = dict.fromkeys(set(feature_keys), 0.0)
    if not centrality:
        return centrality
    for link in links:
        weight_raw = link.get("weight")
        if weight_raw is None:
            continue
        weight = abs(float(weight_raw))
        if weight == 0.0:
            continue
        for endpoint in (link.get("source"), link.get("target")):
            if not isinstance(endpoint, str):
                continue
            try:
                layer, feature, _ = parse_feature_node_id(endpoint)
            except ValueError:
                continue
            key = (layer, feature)
            if key in centrality:
                centrality[key] += weight
    return centrality


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def select_unlabeled_targets(
    graph_path: Path,
    labels: FeatureLabelMap,
    *,
    top_n: int | None = None,
    alpha: float = 0.5,
) -> list[GraphTarget]:
    """Return targets sorted by descending combined score.

    A ``(layer, feature)`` is a target if it is absent from ``labels`` *or* if
    any of its graph nodes still carries a ``[?] ...`` placeholder ``clerp``.
    The latter covers features labelled after this graph was built: their stale
    placeholder clerp is selected so the downstream ``patch_graph`` rewrites it
    from the store (no model call needed, since it is skipped as already
    labelled).
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    nodes, links = load_graph(graph_path)

    grouped: dict[tuple[int, int], dict[str, float]] = {}
    placeholder_keys: set[tuple[int, int]] = set()
    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        try:
            layer, feature, _ = parse_feature_node_id(node_id)
        except ValueError:
            continue
        influence_raw = node.get("influence")
        if influence_raw is None:
            influence_raw = node.get("activation", 0.0)
        influence = abs(float(influence_raw))
        key = (layer, feature)
        clerp = node.get("clerp")
        if isinstance(clerp, str) and clerp.startswith(UNLABELLED_PREFIX):
            placeholder_keys.add(key)
        entry = grouped.setdefault(key, {"target_effect": 0.0, "n_positions": 0.0})
        if influence > entry["target_effect"]:
            entry["target_effect"] = influence
        entry["n_positions"] += 1.0

    centrality_map = compute_centrality(grouped.keys(), links)

    candidates = [
        (key, info) for key, info in grouped.items() if key not in labels or key in placeholder_keys
    ]
    if not candidates:
        return []

    te_values = [info["target_effect"] for _, info in candidates]
    ce_values = [centrality_map.get(key, 0.0) for key, _ in candidates]
    te_norm = _minmax(te_values)
    ce_norm = _minmax(ce_values)

    targets = [
        GraphTarget(
            layer=key[0],
            feature=key[1],
            target_effect=te,
            centrality=ce,
            score=alpha * te_n + (1.0 - alpha) * ce_n,
            n_positions=int(info["n_positions"]),
        )
        for ((key, info), te, ce, te_n, ce_n) in zip(
            candidates, te_values, ce_values, te_norm, ce_norm, strict=True
        )
    ]
    targets.sort(key=lambda target: (-target.score, target.layer, target.feature))
    if top_n is not None:
        if top_n < 0:
            raise ValueError(f"top_n must be non-negative, got {top_n}")
        targets = targets[:top_n]
    return targets


def _write_jsonl(path: Path, targets: list[GraphTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(asdict(target), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank unlabelled (layer, feature) nodes in an attribution graph.",
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSONL output path. If omitted, prints one record per line.",
    )
    args = parser.parse_args()

    labels = load_feature_labels()
    targets = select_unlabeled_targets(
        args.graph,
        labels,
        top_n=args.top_n,
        alpha=args.alpha,
    )
    print(f"[INFO] {len(targets)} unlabelled (layer, feature) targets")

    if args.out is not None:
        _write_jsonl(args.out, targets)
        print(f"[SAVE] wrote targets to {args.out}")
        return

    for target in targets:
        print(json.dumps(asdict(target), ensure_ascii=False))


if __name__ == "__main__":
    main()
