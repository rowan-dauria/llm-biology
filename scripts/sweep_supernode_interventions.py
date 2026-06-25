"""Sweep local-replacement steering magnitudes for one graph supernode.

The input graph is the Neuronpedia-compatible attribution JSON exported by this
project. The script looks up a named qParams supernode, parses its constituent
feature node ids, jointly steers those features with
``biology_server_t_lens.tl_intervention.run_feature_intervention``, and writes a
JSON summary of the logit and feature-activation effects at each magnitude.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("sweep_supernode_interventions")
DEFAULT_MAGNITUDES = "-2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8"


def _apply_circuit_tracer_shim() -> None:
    """Alias older circuit-tracer factory names before project imports."""

    slt = importlib.import_module("circuit_tracer.transcoder.single_layer_transcoder")
    if not hasattr(slt, "load_transcoder"):
        if not hasattr(slt, "load_relu_transcoder"):
            raise ImportError(
                "circuit_tracer.transcoder.single_layer_transcoder has neither "
                "load_transcoder nor load_relu_transcoder"
            )
        slt.load_transcoder = slt.load_relu_transcoder  # type: ignore[attr-defined]


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


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def default_output_path(graph_path: Path, supernode_name: str) -> Path:
    from biology_server.attribution import slugify

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    stem = graph_path.stem
    name = slugify(supernode_name)
    return graph_path.parent / f"{stem}__{name}__intervention-sweep__{stamp}.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_json", type=Path, help="Attribution graph JSON.")
    parser.add_argument("supernode", help="Exact qParams supernode label to steer.")
    parser.add_argument(
        "--magnitudes",
        default=DEFAULT_MAGNITUDES,
        help=(
            "Comma-separated multiplicative steering factors applied to each "
            f"clean feature activation. Default: {DEFAULT_MAGNITUDES}"
        ),
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated tracked layers. Defaults to all feature layers in the graph.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--target-token-id", type=int, default=None)
    parser.add_argument(
        "--target-pos",
        type=int,
        default=None,
        help="Logit position to score. Defaults to the graph target-logit position, else last.",
    )
    parser.add_argument(
        "--measure",
        choices=("all-graph", "supernode", "none"),
        default="all-graph",
        help="Which feature activations to summarize alongside logits.",
    )
    parser.add_argument("--top-logit-changes", type=int, default=10)
    parser.add_argument(
        "--top-prob-tokens",
        type=int,
        default=10,
        help=(
            "How many top-probability tokens to record per magnitude in "
            "top_clean_tokens / top_intervened_tokens (the top-L logit "
            "distribution). Default 10; raise to track a wider distribution."
        ),
    )
    parser.add_argument("--top-feature-changes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply Qwen chat template before tokenization. Off for factual-completion graphs.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    _apply_circuit_tracer_shim()

    from transformers import AutoTokenizer

    from biology_server.attribution import (
        CACHE_DIR,
        load_transcoders,
        pick_device_dtype,
        prepend_special_prefix,
    )
    from biology_server_t_lens.tl_intervention import (
        FeatureIntervention,
        run_feature_intervention,
    )
    from biology_server_t_lens.tl_model import load_replacement_model

    graph_path = args.graph_json.expanduser().resolve()
    graph = load_graph(graph_path)
    metadata = graph.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("graph metadata.prompt is required")

    supernode_label, raw_node_ids = find_supernode(graph, args.supernode)
    nodes_by_id = graph_nodes_by_id(graph)

    constituent_keys: list[tuple[int, int, int]] = []
    skipped_node_ids: list[str] = []
    for node_id in raw_node_ids:
        if not is_graph_feature_node(nodes_by_id.get(node_id)):
            skipped_node_ids.append(node_id)
            continue
        try:
            layer, feature, pos = parse_feature_node_id(node_id)
        except (TypeError, ValueError):
            skipped_node_ids.append(node_id)
            continue
        constituent_keys.append((layer, pos, feature))
    if not constituent_keys:
        raise ValueError(f"supernode {supernode_label!r} contains no feature nodes")

    graph_keys = graph_feature_keys(graph)
    if args.layers is None:
        layers = sorted({layer for layer, _pos, _feature in graph_keys})
    else:
        layers = sorted(parse_csv_ints(args.layers))
    if not layers:
        raise ValueError("no tracked feature layers found; pass --layers explicitly")
    missing_layers = sorted({layer for layer, _pos, _feature in constituent_keys} - set(layers))
    if missing_layers:
        raise ValueError(
            f"supernode contains layers not in --layers: {missing_layers}; tracked layers={layers}"
        )

    magnitudes = parse_csv_floats(args.magnitudes)
    graph_target_token_id, graph_target_pos = primary_logit_target(graph)
    target_token_id = (
        args.target_token_id if args.target_token_id is not None else graph_target_token_id
    )
    target_pos = args.target_pos if args.target_pos is not None else graph_target_pos

    LOGGER.info("graph=%s", graph_path)
    LOGGER.info("prompt=%r", prompt)
    LOGGER.info(
        "supernode=%r feature_nodes=%d skipped=%d",
        supernode_label,
        len(constituent_keys),
        len(skipped_node_ids),
    )
    LOGGER.info("layers=%s magnitudes=%s", layers, magnitudes)

    device, dtype = pick_device_dtype()
    LOGGER.info("device=%s dtype=%s", device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text = prompt
    if args.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    input_ids = prepend_special_prefix(tokenizer, input_ids)
    decoded_prompt_tokens = tokenizer.batch_decode(
        [[int(token_id)] for token_id in input_ids[0].detach().cpu().tolist()]
    )
    graph_prompt_tokens = metadata.get("prompt_tokens")
    if isinstance(graph_prompt_tokens, list) and len(graph_prompt_tokens) != len(
        decoded_prompt_tokens
    ):
        LOGGER.warning(
            "tokenized prompt length (%d) differs from graph metadata.prompt_tokens (%d)",
            len(decoded_prompt_tokens),
            len(graph_prompt_tokens),
        )
    if target_pos is None:
        target_pos = -1

    model = load_replacement_model(args.model_id, device=device, dtype=dtype, cache_dir=CACHE_DIR)
    transcoders = load_transcoders(layers, device=device, dtype=dtype)

    if args.measure == "all-graph":
        measure_features = graph_keys
    elif args.measure == "supernode":
        measure_features = constituent_keys
    else:
        measure_features = []

    rows: list[dict[str, Any]] = []
    constituent_set = set(constituent_keys)
    start = time.time()
    for magnitude in magnitudes:
        LOGGER.info("running magnitude=%s", magnitude)
        interventions = [
            FeatureIntervention(layer=layer, pos=pos, feature=feature, factor=magnitude)
            for layer, pos, feature in constituent_keys
        ]
        result = run_feature_intervention(
            model,
            transcoders,
            input_ids,
            interventions=interventions,
            layers=layers,
            measure_features=measure_features,
        )

        pos = target_pos
        clean_logits = result.clean_logits[pos]
        intervened_logits = result.intervened_logits[pos]
        logit_diff = intervened_logits - clean_logits
        clean_probs = tensor_probs(clean_logits)
        intervened_probs = tensor_probs(intervened_logits)
        prob_diff = intervened_probs - clean_probs

        if target_token_id is None:
            target_token_id_for_row = int(clean_logits.argmax().item())
        else:
            target_token_id_for_row = int(target_token_id)
        target_token = tokenizer.decode([target_token_id_for_row])

        feature_changes: list[dict[str, Any]] = []
        for key, clean_act in result.clean_feature_acts.items():
            intervened_act = result.intervened_feature_acts.get(key)
            if intervened_act is None:
                continue
            layer, pos_key, feature = key
            node_id = f"{layer}_{feature}_{pos_key}"
            delta = intervened_act - clean_act
            frac = float("nan") if clean_act == 0 else intervened_act / clean_act
            node = nodes_by_id.get(node_id, {})
            feature_changes.append(
                {
                    "node_id": node_id,
                    "layer": layer,
                    "pos": pos_key,
                    "feature": feature,
                    "label": node.get("clerp", ""),
                    "clean_activation": clean_act,
                    "intervened_activation": intervened_act,
                    "delta": delta,
                    "fraction": finite_or_none(frac),
                    "is_constituent": key in constituent_set,
                }
            )
        feature_changes.sort(key=lambda item: abs(float(item["delta"])), reverse=True)

        rows.append(
            {
                "magnitude": magnitude,
                "target": {
                    "token_id": target_token_id_for_row,
                    "token": target_token,
                    "pos": target_pos,
                    "clean_logit": float(clean_logits[target_token_id_for_row].item()),
                    "intervened_logit": float(intervened_logits[target_token_id_for_row].item()),
                    "logit_delta": float(logit_diff[target_token_id_for_row].item()),
                    "clean_prob": float(clean_probs[target_token_id_for_row].item()),
                    "intervened_prob": float(intervened_probs[target_token_id_for_row].item()),
                    "prob_delta": float(prob_diff[target_token_id_for_row].item()),
                },
                "max_abs_logit_delta": float(logit_diff.abs().max().item()),
                "top_logit_changes": top_token_rows(
                    tokenizer,
                    logit_diff,
                    k=args.top_logit_changes,
                    largest_abs=True,
                ),
                "top_clean_tokens": top_token_rows(
                    tokenizer,
                    clean_probs,
                    k=args.top_prob_tokens,
                ),
                "top_intervened_tokens": top_token_rows(
                    tokenizer,
                    intervened_probs,
                    k=args.top_prob_tokens,
                ),
                "top_feature_changes": feature_changes[: args.top_feature_changes],
            }
        )

    out_path = (
        (args.output or default_output_path(graph_path, supernode_label)).expanduser().resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "graph": str(graph_path),
            "graph_slug": metadata.get("slug"),
            "prompt": prompt,
            "prompt_tokens": decoded_prompt_tokens,
            "supernode": supernode_label,
            "constituent_node_ids": raw_node_ids,
            "skipped_node_ids": skipped_node_ids,
            "layers": layers,
            "magnitudes": magnitudes,
            "steering_mode": "factor_times_clean_activation",
            "measure": args.measure,
            "top_prob_tokens": args.top_prob_tokens,
            "model_id": args.model_id,
            "device": str(device),
            "dtype": str(dtype),
            "elapsed_seconds": time.time() - start,
        },
        "constituents": [
            {
                "node_id": f"{layer}_{feature}_{pos}",
                "layer": layer,
                "pos": pos,
                "feature": feature,
                "label": nodes_by_id.get(f"{layer}_{feature}_{pos}", {}).get("clerp", ""),
                "graph_activation": nodes_by_id.get(f"{layer}_{feature}_{pos}", {}).get(
                    "activation"
                ),
                "graph_influence": nodes_by_id.get(f"{layer}_{feature}_{pos}", {}).get("influence"),
            }
            for layer, pos, feature in constituent_keys
        ],
        "results": rows,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    LOGGER.info("wrote %s", out_path)
    LOGGER.info(
        "target token=%r id=%s pos=%s",
        rows[0]["target"]["token"],
        rows[0]["target"]["token_id"],
        target_pos,
    )
    for row in rows:
        target = row["target"]
        LOGGER.info(
            "m=%s target_logit_delta=%+.4f target_prob_delta=%+.6f max_abs_logit_delta=%.4f",
            row["magnitude"],
            target["logit_delta"],
            target["prob_delta"],
            row["max_abs_logit_delta"],
        )


if __name__ == "__main__":
    main()
