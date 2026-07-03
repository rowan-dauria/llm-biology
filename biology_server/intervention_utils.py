"""Shared helpers for graph-based intervention scripts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated magnitude")
    return values


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated layer")
    return values


def parse_feature_node_id(node_id: str) -> tuple[int, int, int]:
    parts = node_id.split("_")
    if len(parts) != 3 or parts[0] == "E":
        raise ValueError(f"not a feature node id: {node_id!r}")
    layer, feature, pos = (int(part) for part in parts)
    if layer < 0 or feature < 0 or pos < 0:
        raise ValueError(f"negative feature node id part: {node_id!r}")
    return layer, feature, pos


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"graph JSON must be an object: {path}")
    return payload


def normalized_supernodes(qparams: dict[str, Any]) -> list[list[str]]:
    raw = qparams.get("supernodes", [])
    if isinstance(raw, str) and raw.strip():
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []

    out: list[list[str]] = []
    for group in raw:
        if not isinstance(group, list) or len(group) < 2:
            continue
        label = group[0]
        if not isinstance(label, str):
            continue
        node_ids = [item for item in group[1:] if isinstance(item, str)]
        if node_ids:
            out.append([label, *node_ids])
    return out


def find_supernode(graph: dict[str, Any], name: str) -> tuple[str, list[str]]:
    qparams = graph.get("qParams", {})
    if not isinstance(qparams, dict):
        qparams = {}
    groups = normalized_supernodes(qparams)
    target = name.casefold()
    for group in groups:
        if group[0].casefold() == target:
            return group[0], group[1:]

    available = ", ".join(group[0] for group in groups) or "(none)"
    raise ValueError(f"supernode {name!r} not found. Available supernodes: {available}")


def graph_feature_keys(graph: dict[str, Any]) -> list[tuple[int, int, int]]:
    keys: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("feature_type") != "cross layer transcoder":
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        try:
            layer, feature, pos = parse_feature_node_id(node_id)
        except (TypeError, ValueError):
            continue
        key = (layer, pos, feature)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def graph_nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        node["node_id"]: node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("node_id"), str)
    }


def is_graph_feature_node(node: dict[str, Any] | None) -> bool:
    return bool(node and node.get("feature_type") == "cross layer transcoder")


def primary_logit_target(graph: dict[str, Any]) -> tuple[int | None, int | None]:
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or not node.get("is_target_logit"):
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        parts = node_id.split("_")
        if len(parts) != 3:
            continue
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            continue
    return None, None


def tensor_probs(logits):
    import torch

    return torch.softmax(logits.float(), dim=-1)


def top_token_rows(tokenizer, values, *, k: int, largest_abs: bool = False) -> list[dict[str, Any]]:
    import torch

    if k <= 0:
        return []
    cap = min(k, values.shape[-1])
    order_values = values.abs() if largest_abs else values
    indices = torch.argsort(order_values, descending=True)[:cap]
    ids = [int(idx) for idx in indices.detach().cpu().tolist()]
    decoded = tokenizer.batch_decode([[idx] for idx in ids])
    return [
        {
            "token_id": token_id,
            "token": token,
            "value": float(values[token_id].item()),
        }
        for token_id, token in zip(ids, decoded, strict=True)
    ]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
