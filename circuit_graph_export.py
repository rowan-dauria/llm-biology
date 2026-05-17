"""Export custom attribution results to the circuit-tracer frontend format."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_SCAN = "qwen3-4b"
LOCAL_FEATURE_DIR = LOCAL_SCAN
LOCAL_EXTRA_QPARAMS = {
    "clerps",
    "densityThreshold",
    "gridsnap",
    "hiddenIds",
    "pruningThreshold",
}


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


def _read_existing_qparams(path: Path) -> dict[str, Any]:
    """Return the ``qParams`` block from an existing graph JSON, or ``{}``."""

    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    existing = payload.get("qParams")
    return existing if isinstance(existing, dict) else {}


def merge_qparams(
    existing: dict[str, Any],
    node_ids: set[str],
    default_logit_id: str,
) -> dict[str, Any]:
    """Carry forward UI state across regenerations, dropping references to nodes
    that no longer exist.

    Supernodes use the ``[label, ...node_ids]`` convention; groups left with no
    members after filtering are discarded.
    """

    raw_pinned = existing.get("pinnedIds", [])
    if isinstance(raw_pinned, str):
        raw_pinned = [item for item in raw_pinned.split(",") if item]
    pinned = (
        [pid for pid in raw_pinned if isinstance(pid, str) and pid in node_ids]
        if isinstance(raw_pinned, list)
        else []
    )

    supernodes: list[list[str]] = []
    raw_supernodes = existing.get("supernodes", [])
    if isinstance(raw_supernodes, str) and raw_supernodes:
        try:
            raw_supernodes = json.loads(raw_supernodes)
        except json.JSONDecodeError:
            raw_supernodes = []
    if isinstance(raw_supernodes, list):
        for group in raw_supernodes:
            if not isinstance(group, list) or not group:
                continue
            label = group[0]
            if not isinstance(label, str):
                continue
            kept = [nid for nid in group[1:] if isinstance(nid, str) and nid in node_ids]
            if kept:
                supernodes.append([label, *kept])

    clicked_raw = existing.get("clickedId")
    clicked = (
        clicked_raw
        if isinstance(clicked_raw, str) and clicked_raw in node_ids
        else default_logit_id
    )

    link_type_raw = existing.get("linkType", "both")
    link_type = link_type_raw if isinstance(link_type_raw, str) else "both"

    sg_pos_raw = existing.get("sg_pos", "")
    sg_pos = sg_pos_raw if isinstance(sg_pos_raw, str) else ""

    merged: dict[str, Any] = {
        "pinnedIds": pinned,
        "supernodes": supernodes,
        "linkType": link_type,
        "clickedId": clicked,
        "sg_pos": sg_pos,
    }
    for key in LOCAL_EXTRA_QPARAMS:
        value = existing.get(key)
        if value is not None:
            merged[key] = value
    return merged


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


def build_feature_details_metadata(
    *,
    feature_json_base_url: str | None = None,
    neuronpedia_source_set: str | None = None,
    neuronpedia_lorsa_source_set: str | None = None,
) -> dict[str, str]:
    details: dict[str, str] = {}
    if feature_json_base_url:
        details["feature_json_base_url"] = feature_json_base_url
    if neuronpedia_source_set:
        details["neuronpedia_source_set"] = neuronpedia_source_set
    if neuronpedia_lorsa_source_set:
        details["neuronpedia_lorsa_source_set"] = neuronpedia_lorsa_source_set
    return details


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
    feature_dir_name: str | None = None,
    feature_json_base_url: str | None = None,
    neuronpedia_source_set: str | None = None,
    neuronpedia_lorsa_source_set: str | None = None,
    node_threshold: float | None = None,
) -> Path:
    """Write a graph JSON and metadata file for circuit-tracer's local server."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / f"{slug}.json"
    existing_qparams = _read_existing_qparams(graph_path)

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
        "schema_version": 1,
    }
    if node_threshold is not None:
        threshold = float(node_threshold)
        if math.isfinite(threshold):
            metadata["node_threshold"] = threshold

    feature_details = build_feature_details_metadata(
        feature_json_base_url=feature_json_base_url,
        neuronpedia_source_set=neuronpedia_source_set,
        neuronpedia_lorsa_source_set=neuronpedia_lorsa_source_set,
    )
    if feature_details:
        metadata["feature_details"] = feature_details

    graph = {
        "metadata": metadata,
        "qParams": merge_qparams(existing_qparams, node_ids, logit_id),
        "nodes": nodes,
        "links": [
            {"source": link.source, "target": link.target, "weight": link.weight}
            for link in links
            if link.weight != 0.0
        ],
    }

    _write_json(graph_path, graph)
    write_graph_metadata(out_dir, metadata)
    if feature_examples:
        write_feature_examples(out_dir, feature_examples, feature_dir_name=feature_dir_name or scan)
    return graph_path


def make_feature_example_payload(
    *,
    feature_index: int,
    label: str | None = None,
    windows: list[dict[str, Any]],
    act_max: float | None = None,
    top_logits: list[str] | None = None,
    bottom_logits: list[str] | None = None,
) -> dict[str, Any]:
    """Create the minimal Neuronpedia feature-details JSON.

    ``top_logits`` / ``bottom_logits`` are rendered as token chips in the UI's
    "Token Predictions" panel; the frontend treats them as opaque strings.
    """

    del label, act_max
    schema_windows = [
        {
            "tokens": list(window.get("tokens", [])),
            "tokens_acts_list": list(window.get("tokens_acts_list", [])),
        }
        for window in windows
    ]
    return {
        "index": feature_index,
        "top_logits": list(top_logits or []),
        "bottom_logits": list(bottom_logits or []),
        "examples_quantiles": [
            {
                "quantile_name": "Top activating windows",
                "examples": schema_windows,
            }
        ],
    }
