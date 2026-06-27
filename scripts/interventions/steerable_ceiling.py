"""Diagnose whether weak interventions are a layer-coverage artefact.

For one exported attribution graph + named supernode, this computes two
complementary quantities on the local replacement model and prints a verdict:

#1 **Steerable ceiling (read-off, no perturbation).** With attention patterns and
   normalization scales frozen, the target logit is an affine function of the
   transcoder feature activations, ``z_t = const + Σ a·g``. A single linearised
   backward pass gives each feature's direct coefficient ``g``; we report the
   total direct logit mass carried by *all* tracked features (the ceiling on what
   any feature steering at this layer scope can move), the share at the target
   position (direct-write features), and the supernode's own share. ``const`` is
   the frozen background steering cannot touch.

#2 **Max destruction (intervention).** Zero the features and read the target
   probability: ablating *all* tracked features gives the background prediction
   (the empirical realisation of ``const`` — and the confirmation of #1, since the
   gap between ``background`` and ``clean`` is exactly the feature-controlled
   mass); ablating just the supernode gives how much this supernode controls.

Read together: if removing every tracked feature barely dents the target
probability, no supernode at this coverage can steer it strongly — the
underwhelming result is a coverage artefact, and graphs on more layers are
warranted. If the supernode already owns most of the tracked control, the weak
steering is about leverage/direction, not coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sweep_supernode_interventions as sweep  # noqa: E402

LOGGER = logging.getLogger("steerable_ceiling")

EPS = 1e-9


def odds(prob: float) -> float:
    p = min(max(prob, EPS), 1.0 - EPS)
    return p / (1.0 - p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_json", type=Path, help="Attribution graph JSON.")
    parser.add_argument("supernode", help="Exact qParams supernode label.")
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
        "--top-features",
        type=int,
        default=20,
        help="How many top direct-logit contributors among graph nodes to list.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply Qwen chat template before tokenization. Off for factual-completion graphs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def default_output_path(graph_path: Path, supernode_name: str) -> Path:
    from biology_server.attribution import slugify

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = slugify(supernode_name)
    return graph_path.parent / f"{graph_path.stem}__{name}__steerable-ceiling__{stamp}.json"


def main() -> None:
    sweep.setup_logging()
    args = parse_args()
    sweep._apply_circuit_tracer_shim()

    from transformers import AutoTokenizer

    from biology_server.attribution import (
        CACHE_DIR,
        load_transcoders,
        pick_device_dtype,
        prepend_special_prefix,
    )
    from biology_server.tl_intervention import (
        FeatureIntervention,
        compute_direct_logit_contributions,
        run_feature_intervention,
    )
    from biology_server.tl_model import load_replacement_model

    graph_path = args.graph_json.expanduser().resolve()
    graph = sweep.load_graph(graph_path)
    metadata = graph.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("graph metadata.prompt is required")

    supernode_label, raw_node_ids = sweep.find_supernode(graph, args.supernode)
    nodes_by_id = sweep.graph_nodes_by_id(graph)

    constituent_keys: list[tuple[int, int, int]] = []
    skipped_node_ids: list[str] = []
    for node_id in raw_node_ids:
        if not sweep.is_graph_feature_node(nodes_by_id.get(node_id)):
            skipped_node_ids.append(node_id)
            continue
        try:
            layer, feature, pos = sweep.parse_feature_node_id(node_id)
        except (TypeError, ValueError):
            skipped_node_ids.append(node_id)
            continue
        constituent_keys.append((layer, pos, feature))
    if not constituent_keys:
        raise ValueError(f"supernode {supernode_label!r} contains no feature nodes")

    graph_keys = sweep.graph_feature_keys(graph)
    if args.layers is None:
        layers = sorted({layer for layer, _pos, _feature in graph_keys})
    else:
        layers = sorted(sweep.parse_csv_ints(args.layers))
    if not layers:
        raise ValueError("no tracked feature layers found; pass --layers explicitly")
    missing = sorted({layer for layer, _pos, _feature in constituent_keys} - set(layers))
    if missing:
        raise ValueError(f"supernode contains layers not in --layers: {missing}; layers={layers}")

    graph_target_token_id, graph_target_pos = sweep.primary_logit_target(graph)
    target_token_id = (
        args.target_token_id if args.target_token_id is not None else graph_target_token_id
    )
    target_pos = args.target_pos if args.target_pos is not None else graph_target_pos

    cells = sorted({(layer, pos) for layer, pos, _feature in constituent_keys})

    device, dtype = pick_device_dtype()
    LOGGER.info("graph=%s", graph_path)
    LOGGER.info("prompt=%r", prompt)
    LOGGER.info(
        "supernode=%r constituents=%d cells=%s layers=%s",
        supernode_label,
        len(constituent_keys),
        [f"{layer}_{pos}" for layer, pos in cells],
        layers,
    )
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
    n_pos = input_ids.shape[1]
    if target_pos is None:
        target_pos = n_pos - 1
    tpos = int(target_pos) % n_pos

    model = load_replacement_model(args.model_id, device=device, dtype=dtype, cache_dir=CACHE_DIR)
    transcoders = load_transcoders(layers, device=device, dtype=dtype)

    start = time.time()

    # --- clean forward (full logit vector) + no-op freeze sanity check ---
    clean = run_feature_intervention(model, transcoders, input_ids, interventions=[], layers=layers)
    clean_logits = clean.clean_logits[tpos]
    noop_logits = clean.intervened_logits[tpos]
    noop_drift = float((noop_logits - clean_logits).abs().max().item())
    clean_probs = sweep.tensor_probs(clean_logits)
    if target_token_id is None:
        target_token_id = int(clean_logits.argmax().item())
    target_token_id = int(target_token_id)
    target_token = tokenizer.decode([target_token_id])
    clean_logit = float(clean_logits[target_token_id].item())
    clean_prob = float(clean_probs[target_token_id].item())

    # --- #1 steerable ceiling: linearised direct-logit read-off ---
    contrib = compute_direct_logit_contributions(
        model,
        transcoders,
        input_ids,
        target_token_id=target_token_id,
        target_pos=tpos,
        layers=layers,
        feature_keys=graph_keys,
    )
    supernode_mass = sum(contrib.contributions.get(k, 0.0) for k in constituent_keys)

    # --- #2 ablations on the same replacement model ---
    def ablation_prob_logit(result) -> tuple[float, float]:
        logits = result.intervened_logits[tpos]
        prob = float(sweep.tensor_probs(logits)[target_token_id].item())
        return prob, float(logits[target_token_id].item())

    all_pos = {layer: None for layer in layers}
    background = run_feature_intervention(
        model,
        transcoders,
        input_ids,
        interventions=[],
        layers=layers,
        ablate_all_features_at=all_pos,
    )
    background_prob, background_logit = ablation_prob_logit(background)

    tgt_pos_only = {layer: [tpos] for layer in layers}
    background_tpos = run_feature_intervention(
        model,
        transcoders,
        input_ids,
        interventions=[],
        layers=layers,
        ablate_all_features_at=tgt_pos_only,
    )
    background_tpos_prob, background_tpos_logit = ablation_prob_logit(background_tpos)

    supernode_ablate = run_feature_intervention(
        model,
        transcoders,
        input_ids,
        layers=layers,
        interventions=[
            FeatureIntervention(layer=layer, pos=pos, feature=feature, factor=0.0)
            for layer, pos, feature in constituent_keys
        ],
    )
    supernode_ablate_prob, supernode_ablate_logit = ablation_prob_logit(supernode_ablate)

    # --- derived diagnostics ---
    tracked_control = clean_prob - background_prob  # prob the tracked features add
    supernode_control = clean_prob - supernode_ablate_prob
    verification_residual = background_logit - contrib.const  # ~0 confirms the read-off

    def share(numer: float, denom: float) -> float | None:
        return numer / denom if abs(denom) > EPS else None

    supernode_mass_share = share(supernode_mass, contrib.total_mass_all)
    supernode_control_share = share(supernode_control, tracked_control)
    target_pos_mass_share = share(contrib.total_mass_target_pos, contrib.total_mass_all)

    # --- verdict heuristic ---
    if abs(tracked_control) < 0.05 * max(clean_prob, EPS):
        verdict = (
            "COVERAGE-LIMITED: removing every tracked feature barely moves the target "
            "probability, so no supernode at this layer scope can steer it strongly. The weak "
            "intervention is consistent with limited coverage — generate graphs on more layers."
        )
    elif supernode_control_share is not None and supernode_control_share < 0.33:
        verdict = (
            "SUPERNODE-INCOMPLETE: the tracked features control the prediction, but this "
            "supernode owns only a small share of that control. Weakness is about the supernode "
            "definition, not (only) coverage — more layers may add the missing constituents."
        )
    else:
        verdict = (
            "NOT COVERAGE-LIMITED: the supernode already owns most of what the covered features "
            "control. Weak steering is about leverage/direction, not coverage — more layers are "
            "unlikely to change the result."
        )

    # --- top direct contributors among graph nodes ---
    ranked = sorted(contrib.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[
        : args.top_features
    ]
    top_contributors = []
    for (layer, pos, feature), mass in ranked:
        node_id = f"{layer}_{feature}_{pos}"
        node = nodes_by_id.get(node_id, {})
        top_contributors.append(
            {
                "node_id": node_id,
                "layer": layer,
                "pos": pos,
                "feature": feature,
                "label": node.get("clerp", ""),
                "clean_activation": contrib.clean_acts.get((layer, pos, feature)),
                "coeff_g": contrib.coeffs.get((layer, pos, feature)),
                "direct_logit_mass": mass,
                "is_constituent": (layer, pos, feature) in set(constituent_keys),
            }
        )

    output = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "graph": str(graph_path),
            "graph_slug": metadata.get("slug"),
            "prompt": prompt,
            "supernode": supernode_label,
            "constituent_node_ids": raw_node_ids,
            "skipped_node_ids": skipped_node_ids,
            "cells": [f"{layer}_{pos}" for layer, pos in cells],
            "layers": layers,
            "model_id": args.model_id,
            "device": str(device),
            "dtype": str(dtype),
            "noop_freeze_drift": noop_drift,
            "elapsed_seconds": time.time() - start,
        },
        "target": {
            "token_id": target_token_id,
            "token": target_token,
            "pos": tpos,
            "clean_logit": clean_logit,
            "clean_prob": clean_prob,
            "clean_odds": odds(clean_prob),
        },
        "steerable_ceiling": {
            "_doc": "read-off: direct logit mass a*g of tracked features (no perturbation)",
            "const_background_logit": contrib.const,
            "total_mass_all_tracked": contrib.total_mass_all,
            "total_mass_target_pos": contrib.total_mass_target_pos,
            "target_pos_mass_share": target_pos_mass_share,
            "supernode_direct_mass": supernode_mass,
            "supernode_mass_share": supernode_mass_share,
            "verification_residual_logit": verification_residual,
        },
        "max_destruction": {
            "_doc": "intervention: target prob after zeroing features",
            "background_all_tracked": {
                "prob": background_prob,
                "logit": background_logit,
                "odds": odds(background_prob),
            },
            "background_target_pos_only": {
                "prob": background_tpos_prob,
                "logit": background_tpos_logit,
                "odds": odds(background_tpos_prob),
            },
            "supernode_ablated": {
                "prob": supernode_ablate_prob,
                "logit": supernode_ablate_logit,
                "odds": odds(supernode_ablate_prob),
            },
            "tracked_control_prob": tracked_control,
            "supernode_control_prob": supernode_control,
            "supernode_control_share": supernode_control_share,
        },
        "verdict": verdict,
        "top_direct_contributors": top_contributors,
    }

    out_path = (
        (args.output or default_output_path(graph_path, supernode_label)).expanduser().resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    LOGGER.info("wrote %s", out_path)
    LOGGER.info("target token=%r id=%d pos=%d", target_token, target_token_id, tpos)
    LOGGER.info("no-op freeze drift (should be ~0): %.3e", noop_drift)
    LOGGER.info("clean: logit=%.4f prob=%.6f odds=%.3f", clean_logit, clean_prob, odds(clean_prob))
    LOGGER.info(
        "#1 read-off: tracked direct mass=%.4f (target-pos %.4f, %.0f%%); supernode mass=%.4f "
        "(%.0f%% of tracked)",
        contrib.total_mass_all,
        contrib.total_mass_target_pos,
        100.0 * (target_pos_mass_share or 0.0),
        supernode_mass,
        100.0 * (supernode_mass_share or 0.0),
    )
    LOGGER.info(
        "    verification: ablate-all logit=%.4f vs read-off const=%.4f (residual %.3e)",
        background_logit,
        contrib.const,
        verification_residual,
    )
    LOGGER.info(
        "#2 ablations: background(all tracked) prob=%.6f | target-pos-only prob=%.6f | "
        "supernode prob=%.6f",
        background_prob,
        background_tpos_prob,
        supernode_ablate_prob,
    )
    LOGGER.info(
        "    tracked control=%.6f prob; supernode control=%.6f prob (%.0f%% of tracked)",
        tracked_control,
        supernode_control,
        100.0 * (supernode_control_share or 0.0),
    )
    LOGGER.info("VERDICT: %s", verdict)


if __name__ == "__main__":
    main()
