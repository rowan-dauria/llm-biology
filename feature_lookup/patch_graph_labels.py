"""Refresh ``clerp`` on a graph JSON.

Reads a frontend graph JSON, joins against the per-layer labels store, and
overwrites ``clerp`` for every transcoder feature node.

Pure-logic module: no torch, no transformers, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    from .graph_targets import (
        TRANSCODER_FEATURE_TYPE,
        parse_feature_node_id,
    )
    from .labels import FeatureLabelMap, get_feature_label, load_feature_labels
except ImportError:
    from graph_targets import (  # type: ignore[no-redef]
        TRANSCODER_FEATURE_TYPE,
        parse_feature_node_id,
    )
    from labels import (  # type: ignore[no-redef]
        FeatureLabelMap,
        get_feature_label,
        load_feature_labels,
    )

DEFAULT_SCAN_DIR = "qwen3-4b"
DEFAULT_GRAPH_DIR = Path(__file__).parent.parent / "data" / "ui_graphs"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _patch_graph_nodes(
    graph: dict[str, Any], labels: FeatureLabelMap
) -> tuple[int, set[tuple[int, int]]]:
    updated = 0
    touched: set[tuple[int, int]] = set()
    for node in graph.get("nodes", []):
        if node.get("feature_type") != TRANSCODER_FEATURE_TYPE:
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        try:
            layer, feature, _ = parse_feature_node_id(node_id)
        except ValueError:
            continue
        new_clerp = get_feature_label(labels, layer, feature)
        if node.get("clerp") != new_clerp:
            node["clerp"] = new_clerp
            updated += 1
        touched.add((layer, feature))
    return updated, touched


def patch_graph(
    graph_path: Path,
    labels: FeatureLabelMap,
    *,
    scan_dir: str = DEFAULT_SCAN_DIR,
) -> dict[str, int]:
    """Patch ``clerp`` in the graph JSON.

    Returns ``{clerp_updated, examples_updated, examples_missing}``. The sidecar
    counts are retained as zero-valued compatibility fields; feature-detail
    sidecars are schema-minimal and are no longer treated as label stores.
    """
    del scan_dir

    with graph_path.open(encoding="utf-8") as handle:
        graph = json.load(handle)

    clerp_updated, _ = _patch_graph_nodes(graph, labels)
    if clerp_updated:
        _atomic_write_json(graph_path, graph)

    return {
        "clerp_updated": clerp_updated,
        "examples_updated": 0,
        "examples_missing": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh clerp labels on an attribution graph JSON.",
    )
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--scan-dir", default=DEFAULT_SCAN_DIR)
    args = parser.parse_args()

    if args.graph is None and args.slug is None:
        parser.error("provide --graph or --slug")
    graph_path = args.graph if args.graph is not None else args.graph_dir / f"{args.slug}.json"
    if not graph_path.exists():
        parser.error(f"graph not found: {graph_path}")

    labels = load_feature_labels()
    counts = patch_graph(graph_path, labels, scan_dir=args.scan_dir)
    print(
        f"[INFO] patched {graph_path}: "
        f"clerp_updated={counts['clerp_updated']} "
        f"examples_updated={counts['examples_updated']} "
        f"examples_missing={counts['examples_missing']}"
    )


if __name__ == "__main__":
    main()
