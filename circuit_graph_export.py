"""Export custom attribution results to the circuit-tracer frontend format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_SCAN = "./data/features/qwen3-4b-transcoders"
LOCAL_FEATURE_DIR = "qwen3-4b-transcoders"


@dataclass(frozen=True, slots=True)
class FeatureNode:
    layer: int
    pos: int
    feature: int
    activation: float
    clerp: str
    influence: float | None = None


@dataclass(frozen=True, slots=True)
class GraphLink:
    source: str
    target: str
    weight: float


def cantor_pair(x: int, y: int) -> int:
    """Pair two non-negative ints using the frontend's expected Cantor map."""

    if x < 0 or y < 0:
        raise ValueError("cantor_pair expects non-negative integers")
    return (x + y) * (x + y + 1) // 2 + y


def paired_feature_index(layer: int, feature: int) -> int:
    return cantor_pair(layer, feature)


def feature_node_id(layer: int, feature: int, pos: int) -> str:
    return f"{layer}_{feature}_{pos}"


def embedding_node_id(vocab_idx: int, pos: int) -> str:
    return f"E_{vocab_idx}_{pos}"


def logit_node_id(num_layers: int, vocab_idx: int, pos: int) -> str:
    return f"{num_layers + 1}_{vocab_idx}_{pos}"


def _feature_node_to_json(node: FeatureNode) -> dict[str, Any]:
    paired = paired_feature_index(node.layer, node.feature)
    data: dict[str, Any] = {
        "node_id": feature_node_id(node.layer, node.feature, node.pos),
        "feature": paired,
        "layer": str(node.layer),
        "ctx_idx": node.pos,
        "feature_type": "cross layer transcoder",
        "token_prob": 0.0,
        "is_target_logit": False,
        "run_idx": 0,
        "reverse_ctx_idx": 0,
        "jsNodeId": f"{node.layer}_{node.feature}-0",
        "clerp": node.clerp,
        "activation": node.activation,
        "vis_link": "",
    }
    if node.influence is not None:
        data["influence"] = node.influence
    return data


def _embedding_node_to_json(pos: int, vocab_idx: int) -> dict[str, Any]:
    return {
        "node_id": embedding_node_id(vocab_idx, pos),
        "feature": pos,
        "layer": "E",
        "ctx_idx": pos,
        "feature_type": "embedding",
        "token_prob": 0.0,
        "is_target_logit": False,
        "run_idx": 0,
        "reverse_ctx_idx": 0,
        "jsNodeId": f"E_{vocab_idx}-{pos}",
        "clerp": "",
    }


def _logit_node_to_json(
    *,
    num_layers: int,
    pos: int,
    vocab_idx: int,
    token: str,
    token_prob: float,
) -> dict[str, Any]:
    layer = str(num_layers + 1)
    return {
        "node_id": logit_node_id(num_layers, vocab_idx, pos),
        "feature": vocab_idx,
        "layer": layer,
        "ctx_idx": pos,
        "feature_type": "logit",
        "token_prob": token_prob,
        "is_target_logit": True,
        "run_idx": 0,
        "reverse_ctx_idx": 0,
        "jsNodeId": f"L_{vocab_idx}-{pos}",
        "clerp": f'Output "{token}" (p={token_prob:.3f})',
    }


def _read_graph_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"graphs": []}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_graph_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    """Add or replace a graph metadata entry by slug."""

    metadata_path = output_dir / "graph-metadata.json"
    payload = _read_graph_metadata(metadata_path)
    payload["graphs"] = [
        graph for graph in payload.get("graphs", []) if graph.get("slug") != metadata["slug"]
    ]
    payload["graphs"].append(metadata)
    _write_json(metadata_path, payload)


def write_feature_examples(
    output_dir: Path,
    feature_examples: dict[int, dict[str, Any]],
    *,
    feature_dir_name: str = LOCAL_FEATURE_DIR,
) -> None:
    feature_dir = output_dir / "features" / feature_dir_name
    feature_dir.mkdir(parents=True, exist_ok=True)
    for feature_index, payload in feature_examples.items():
        _write_json(feature_dir / f"{feature_index}.json", payload)


def export_circuit_graph(
    *,
    output_dir: Path | str,
    slug: str,
    prompt: str,
    prompt_tokens: list[str],
    input_token_ids: list[int],
    num_layers: int,
    feature_nodes: list[FeatureNode],
    links: list[GraphLink],
    target_token_id: int,
    target_token_str: str,
    target_token_prob: float,
    feature_examples: dict[int, dict[str, Any]] | None = None,
    scan: str = LOCAL_SCAN,
) -> Path:
    """Write a graph JSON and metadata file for circuit-tracer's local server."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_id = logit_node_id(num_layers, target_token_id, len(prompt_tokens) - 1)
    nodes = [_feature_node_to_json(node) for node in feature_nodes]
    nodes.extend(
        _embedding_node_to_json(pos, vocab_idx) for pos, vocab_idx in enumerate(input_token_ids)
    )
    nodes.append(
        _logit_node_to_json(
            num_layers=num_layers,
            pos=len(prompt_tokens) - 1,
            vocab_idx=target_token_id,
            token=target_token_str,
            token_prob=target_token_prob,
        )
    )

    node_ids = {node["node_id"] for node in nodes}
    dangling = [
        link for link in links if link.source not in node_ids or link.target not in node_ids
    ]
    if dangling:
        first = dangling[0]
        raise ValueError(f"Dangling graph link: {first.source!r} -> {first.target!r}")

    metadata = {
        "slug": slug,
        "scan": scan,
        "transcoder_list": [],
        "prompt_tokens": prompt_tokens,
        "prompt": prompt,
        "node_threshold": None,
        "schema_version": 1,
    }
    graph = {
        "metadata": metadata,
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
            "clickedId": logit_id,
            "sg_pos": "",
        },
        "nodes": nodes,
        "links": [
            {"source": link.source, "target": link.target, "weight": link.weight}
            for link in links
            if link.weight != 0.0
        ],
    }

    graph_path = out_dir / f"{slug}.json"
    _write_json(graph_path, graph)
    write_graph_metadata(out_dir, metadata)
    if feature_examples:
        write_feature_examples(out_dir, feature_examples)
    return graph_path


def make_feature_example_payload(
    *,
    feature_index: int,
    label: str,
    windows: list[dict[str, Any]],
    act_max: float | None = None,
) -> dict[str, Any]:
    """Create the minimal feature-example JSON consumed by the bundled frontend."""

    max_window_act = max((float(window.get("value", 0.0)) for window in windows), default=0.0)
    return {
        "feature": feature_index,
        "featureIndex": feature_index,
        "label": label,
        "act_min": 0,
        "act_max": float(act_max if act_max is not None else max(max_window_act, 1.0)),
        "top_logits": [],
        "bottom_logits": [],
        "examples_quantiles": [
            {
                "quantile_name": "Top activating windows",
                "examples": windows,
            }
        ],
    }
