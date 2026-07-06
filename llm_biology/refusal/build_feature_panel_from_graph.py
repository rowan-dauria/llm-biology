"""Build a cross-model feature panel from a pruned attribution graph.

This implements stages 1-3 of the base-vs-jailbroken comparison proposed in
supervisor meeting feedback (cf. Alessandro 2026-06-19):

1. a graph has already been constructed and pruned in one model (a human
   selected the important nodes interactively in the supernode graph editor,
   recorded as ``qParams.pinnedIds``/``qParams.supernodes`` on the exported
   frontend graph JSON);
2. take the important features selected by that graph;
3. hand them to ``compare_cross_model_feature_activations.py``, which measures
   those exact ``(layer, pos, feature)`` activations in the other model.

By default, only ``"cross layer transcoder"`` pinned nodes are emitted
(embedding, logit, and reconstruction-error pinned nodes aren't features with
a stable ``(layer, pos, feature)`` triple, so the comparison script can't
measure them in another model). Each pinned feature's label prefers the
human-edited ``qParams.clerps`` override (keyed by transcoder feature id) over
the graph's baked-in ``clerp``, and its category is the supernode group name if
the node belongs to one, else ``"ungrouped"``.

Pass ``--all-feature-nodes`` to emit every ``"cross layer transcoder"`` node in
the graph. In that mode labels use the baked graph ``clerp`` so the panel is
unsupervised apart from the algorithmic graph pruning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("build_feature_panel_from_graph")

FEATURE_TYPE = "cross layer transcoder"


def setup_logging() -> None:
    """Configure root logging to stream INFO-and-above to stdout with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_supernode_labels(raw: str | None) -> dict[str, str]:
    """Map pinned node_id -> supernode group name from ``qParams.supernodes``."""
    if not raw:
        return {}
    groups = json.loads(raw)
    labels: dict[str, str] = {}
    for group in groups:
        name, *members = group
        for member in members:
            labels[member] = name
    return labels


def parse_clerp_overrides(raw: str | None) -> dict[str, str]:
    """Map transcoder feature id (as string) -> human-edited label from ``qParams.clerps``."""
    if not raw:
        return {}
    pairs = json.loads(raw)
    return {str(feature_id): label for feature_id, label in pairs}


def panel_row_from_node(
    node: dict[str, Any],
    *,
    prompt_tokens: list[str],
    label: str,
    category: str,
) -> dict[str, Any]:
    """Build one feature-panel row from a graph node, its resolved label, and its group category."""
    ctx_idx = node["ctx_idx"]
    prompt_token = prompt_tokens[ctx_idx] if 0 <= ctx_idx < len(prompt_tokens) else None
    return {
        "node_id": node["node_id"],
        "label": label,
        "category": category,
        "source_activation": node.get("activation"),
        "source_influence": node.get("influence"),
        "prompt_token": prompt_token,
    }


def build_panel(
    graph: dict[str, Any], *, direction: str, all_feature_nodes: bool = False
) -> dict[str, Any]:
    """Build a feature panel from either the graph's pinned nodes or all its feature nodes.

    Raises ``ValueError`` if no ``"cross layer transcoder"`` nodes are found
    among the candidates.
    """
    metadata = graph["metadata"]
    q_params = graph.get("qParams", {})
    pinned_ids = [node_id for node_id in q_params.get("pinnedIds", "").split(",") if node_id]
    supernode_labels = parse_supernode_labels(q_params.get("supernodes"))
    clerp_overrides = parse_clerp_overrides(q_params.get("clerps"))
    prompt_tokens = metadata["prompt_tokens"]

    nodes_by_id = {node["node_id"]: node for node in graph["nodes"]}

    features: list[dict[str, Any]] = []
    skipped_other_type = 0
    if all_feature_nodes:
        candidate_ids = [node["node_id"] for node in graph["nodes"]]
    else:
        candidate_ids = pinned_ids

    for node_id in candidate_ids:
        node = nodes_by_id.get(node_id)
        if node is None:
            LOGGER.warning("pinned node_id=%s not found in graph nodes; skipping", node_id)
            continue
        if node.get("feature_type") != FEATURE_TYPE:
            skipped_other_type += 1
            continue
        if all_feature_nodes:
            label = node.get("clerp") or ""
            category = "all_feature_nodes"
        else:
            label = clerp_overrides.get(str(node["feature"])) or node.get("clerp") or ""
            category = supernode_labels.get(node_id, "ungrouped")
        features.append(
            panel_row_from_node(
                node,
                prompt_tokens=prompt_tokens,
                label=label,
                category=category,
            )
        )

    if not features:
        raise ValueError(
            f"no {FEATURE_TYPE!r} nodes found in graph "
            f"(candidate_ids={len(candidate_ids)}, non-feature={skipped_other_type})"
        )

    LOGGER.info(
        "graph=%s mode=%s candidates=%d pinned=%d feature_nodes=%d skipped_non_feature=%d",
        metadata.get("slug"),
        "all-feature-nodes" if all_feature_nodes else "pinned",
        len(candidate_ids),
        len(pinned_ids),
        len(features),
        skipped_other_type,
    )

    panel_metadata = {
        "prompt": metadata["prompt"],
        "prompt_format": "chat",
        "direction": direction,
        "prompt_id": metadata.get("slug", "prompt_0"),
        "source_graph_slug": metadata.get("slug"),
    }
    if all_feature_nodes:
        panel_metadata["selection"] = "all_feature_nodes"

    return {
        "metadata": panel_metadata,
        "features": features,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building a feature panel from a graph JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph",
        type=Path,
        help="Frontend attribution graph JSON with qParams.pinnedIds (a pruned, human-curated graph).",
    )
    parser.add_argument(
        "--direction",
        required=True,
        help="e.g. base_to_jailbroken or jailbroken_to_base.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-feature-nodes",
        action="store_true",
        help=(
            f"Emit every {FEATURE_TYPE!r} node in the graph instead of only "
            "qParams.pinnedIds. Labels use baked graph clerps in this mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: build a feature panel from a graph JSON and write it as JSON."""
    setup_logging()
    args = parse_args()
    with args.graph.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    panel = build_panel(graph, direction=args.direction, all_feature_nodes=args.all_feature_nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(panel, handle, indent=2)
        handle.write("\n")
    LOGGER.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
