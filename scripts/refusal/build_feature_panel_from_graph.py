"""Build a curated cross-model feature panel from a pruned attribution graph.

This implements stages 1-3 of the base-vs-jailbroken comparison from
`notes/meeting-notes/19-06-alessandro/alessandro-meeting-summary.md`:

1. a graph has already been constructed and pruned in one model (a human
   selected the important nodes interactively in the supernode graph editor,
   recorded as ``qParams.pinnedIds``/``qParams.supernodes`` on the exported
   frontend graph JSON);
2. take the important features selected by that graph;
3. hand them to ``compare_cross_model_feature_activations.py``, which measures
   those exact ``(layer, pos, feature)`` activations in the other model.

Only ``"cross layer transcoder"`` pinned nodes are emitted (embedding, logit,
and reconstruction-error pinned nodes aren't features with a stable
``(layer, pos, feature)`` triple, so the comparison script can't measure them
in another model). Each feature's label prefers the human-edited
``qParams.clerps`` override (keyed by transcoder feature id) over the graph's
baked-in ``clerp``, and its category is the supernode group name if the node
belongs to one, else ``"ungrouped"``.
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


def build_panel(graph: dict[str, Any], *, direction: str) -> dict[str, Any]:
    metadata = graph["metadata"]
    q_params = graph.get("qParams", {})
    pinned_ids = [node_id for node_id in q_params.get("pinnedIds", "").split(",") if node_id]
    supernode_labels = parse_supernode_labels(q_params.get("supernodes"))
    clerp_overrides = parse_clerp_overrides(q_params.get("clerps"))
    prompt_tokens = metadata["prompt_tokens"]

    nodes_by_id = {node["node_id"]: node for node in graph["nodes"]}

    features: list[dict[str, Any]] = []
    skipped_other_type = 0
    for node_id in pinned_ids:
        node = nodes_by_id.get(node_id)
        if node is None:
            LOGGER.warning("pinned node_id=%s not found in graph nodes; skipping", node_id)
            continue
        if node.get("feature_type") != FEATURE_TYPE:
            skipped_other_type += 1
            continue
        ctx_idx = node["ctx_idx"]
        prompt_token = prompt_tokens[ctx_idx] if 0 <= ctx_idx < len(prompt_tokens) else None
        label = clerp_overrides.get(str(node["feature"])) or node.get("clerp") or ""
        category = supernode_labels.get(node_id, "ungrouped")
        features.append(
            {
                "node_id": node_id,
                "label": label,
                "category": category,
                "source_activation": node.get("activation"),
                "source_influence": node.get("influence"),
                "prompt_token": prompt_token,
            }
        )

    if not features:
        raise ValueError(
            f"no pinned {FEATURE_TYPE!r} nodes found in graph "
            f"(pinnedIds={len(pinned_ids)}, non-feature pinned={skipped_other_type})"
        )

    LOGGER.info(
        "graph=%s pinned=%d feature_nodes=%d skipped_non_feature=%d",
        metadata.get("slug"),
        len(pinned_ids),
        len(features),
        skipped_other_type,
    )

    return {
        "metadata": {
            "prompt": metadata["prompt"],
            "prompt_format": "chat",
            "direction": direction,
            "prompt_id": metadata.get("slug", "prompt_0"),
            "source_graph_slug": metadata.get("slug"),
        },
        "features": features,
    }


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    with args.graph.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    panel = build_panel(graph, direction=args.direction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(panel, handle, indent=2)
        handle.write("\n")
    LOGGER.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
