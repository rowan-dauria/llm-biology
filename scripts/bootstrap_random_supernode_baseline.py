"""Random size-matched baseline for a supernode intervention sweep.

Companion to ``sweep_supernode_interventions.py``. That script steers the
*real* constituents of a named supernode across magnitudes; this one asks
whether the observed effect is special by comparing it to ``N`` random
size-matched feature sets drawn from the same population.

For a supernode with constituents at cells ``(layer, pos)`` it builds a pool of
features that are **active at those same cells** on the clean forward (so the
control is "other features firing at the same tokens/layers", not "dead
features"), then draws ``N`` bootstrap sets that match the supernode's
``(layer, pos)`` footprint exactly (``--sampling matched-cell``, default) or
that draw ``K`` features from the pooled active cells (``--sampling
global-active``). Each draw is swept over the same magnitudes with
``run_feature_intervention``; per magnitude we report the targeted effect, the
baseline distribution (mean/std, percentiles) and a one-sided empirical
p-value.

Plot the targeted ``logit_delta(m)`` curve against the baseline percentile band
(use the spread, not SEM): if Texas sits outside the band, it is doing more
than a random same-size set at the same tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sweep_supernode_interventions import (  # noqa: E402
    _apply_circuit_tracer_shim,
    find_supernode,
    graph_nodes_by_id,
    is_graph_feature_node,
    load_graph,
    parse_csv_floats,
    parse_csv_ints,
    parse_feature_node_id,
    primary_logit_target,
    setup_logging,
    tensor_probs,
)

LOGGER = logging.getLogger("bootstrap_random_supernode_baseline")
DEFAULT_MAGNITUDES = "-2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8"
PERCENTILES = (5, 25, 50, 75, 95)

# A feature cell is (layer, pos, feature) — matches FeatureIntervention.
FeatureKey = tuple[int, int, int]


def capture_clean_features(
    model: Any,
    transcoders: dict[int, Any],
    input_ids: Any,
    layers: list[int],
) -> dict[int, Any]:
    """Clean forward; return ``{layer: feature_acts (n_pos, d_transcoder)}``.

    Mirrors the clean encode in ``tl_intervention._capture_clean`` (no position
    zeroing) so the activations read here match what the intervention resolves
    its multiplicative factors against.
    """
    import torch

    from biology_server_t_lens.tl_forward import ensure_replacement_mlp_hooks

    ensure_replacement_mlp_hooks(model, layers)
    captured: dict[int, Any] = {}

    def _in_hook(layer: int):
        transcoder = transcoders[layer]

        def hook(acts, hook):  # noqa: ARG001, ANN001
            with torch.no_grad():
                captured[layer] = transcoder.encode(acts).detach()[0].float().cpu()
            return acts

        return hook

    fwd_hooks = [(f"blocks.{layer}.mlp.hook_in", _in_hook(layer)) for layer in layers]
    with torch.no_grad():
        model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)
    return captured


def build_candidate_pool(
    clean_features: dict[int, Any],
    cells: set[tuple[int, int]],
    exclude: set[FeatureKey],
    min_activation: float,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """For each ``(layer, pos)`` cell, list active ``(feature, activation)``.

    Features in ``exclude`` (the supernode's own constituents) are removed so a
    random draw never re-selects them.
    """
    pool: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for layer, pos in sorted(cells):
        feats = clean_features.get(layer)
        if feats is None:
            pool[(layer, pos)] = []
            continue
        row = feats[pos]
        active_idx = (row > min_activation).nonzero().flatten().tolist()
        candidates = [
            (int(f), float(row[f])) for f in active_idx if (layer, pos, int(f)) not in exclude
        ]
        pool[(layer, pos)] = candidates
    return pool


def _magnitude_ok(cand_act: float, target_act: float, tol: float | None) -> bool:
    if tol is None:
        return True
    if target_act <= 0:
        return True
    lo, hi = target_act / (1.0 + tol), target_act * (1.0 + tol)
    return lo <= cand_act <= hi


def sample_draws(
    constituent_keys: list[FeatureKey],
    clean_features: dict[int, Any],
    pool: dict[tuple[int, int], list[tuple[int, float]]],
    *,
    n: int,
    sampling: str,
    match_magnitude_tol: float | None,
    rng: np.random.Generator,
) -> tuple[list[list[FeatureKey]], dict[str, Any]]:
    """Draw ``n`` random size-matched feature sets.

    Returns the draws plus an ``info`` dict (combination-space size, whether
    draws are guaranteed distinct, etc.) for the output metadata.
    """
    clean_act = {
        (layer, pos, feature): float(clean_features[layer][pos, feature])
        for layer, pos, feature in constituent_keys
    }
    k = len(constituent_keys)

    if sampling == "matched-cell":
        # Group constituents by cell, then resolve each cell's candidate set
        # once (it is deterministic: magnitude matching depends only on the
        # cell's constituents, not on the draw).
        by_cell: dict[tuple[int, int], list[FeatureKey]] = {}
        for key in constituent_keys:
            by_cell.setdefault((key[0], key[1]), []).append(key)

        cell_plan: list[tuple[int, int, list[int], int]] = []
        n_combos = 1
        replacement_forced = False
        for (layer, pos), members in by_cell.items():
            cands = [
                f
                for f, act in pool[(layer, pos)]
                if all(_magnitude_ok(act, clean_act[m], match_magnitude_tol) for m in members)
            ]
            need = len(members)
            if not cands:
                raise ValueError(
                    f"cell {(layer, pos)} has no candidate features; "
                    "lower --min-activation or relax --match-magnitude-tol"
                )
            if len(cands) < need:
                replacement_forced = True
            else:
                n_combos *= math.comb(len(cands), need)
            cell_plan.append((layer, pos, cands, need))

        # A draw is the joint selection across all cells, so the distinct-draw
        # space is the product of per-cell binomials — typically enormous even
        # when individual cells are small.
        want_distinct = (not replacement_forced) and n_combos >= n
        if replacement_forced:
            LOGGER.warning(
                "a cell has fewer candidates than constituents; sampling with replacement"
            )
        elif n_combos < n:
            LOGGER.warning("only %d distinct draws exist (< n=%d); draws will repeat", n_combos, n)
        else:
            LOGGER.info("matched-cell distinct-draw space ~= %d", n_combos)

        draws: list[list[FeatureKey]] = []
        seen: set[tuple[FeatureKey, ...]] = set()
        attempts = 0
        max_attempts = max(n * 50, 1000)
        while len(draws) < n and attempts < max_attempts:
            attempts += 1
            draw: list[FeatureKey] = []
            for layer, pos, cands, need in cell_plan:
                chosen = rng.choice(cands, size=need, replace=len(cands) < need)
                draw.extend((layer, pos, int(f)) for f in chosen)
            if want_distinct:
                signature = tuple(sorted(draw))
                if signature in seen:
                    continue
                seen.add(signature)
            draws.append(draw)
        info = {
            "n_combinations": int(n_combos),
            "replacement_forced": replacement_forced,
            "distinct_draws": want_distinct,
            "n_distinct_features": sum(len(cands) for _l, _p, cands, _n in cell_plan),
        }
        return draws, info

    if sampling == "global-active":
        flat = [(layer, pos, f) for (layer, pos), cands in pool.items() for f, _act in cands]
        if len(flat) < k:
            raise ValueError(f"global-active pool has {len(flat)} features < supernode size {k}")
        idx = np.arange(len(flat))
        draws = []
        for _ in range(n):
            chosen = rng.choice(idx, size=k, replace=False)
            draws.append([flat[i] for i in chosen])
        info = {
            "n_combinations": int(math.comb(len(flat), k)),
            "replacement_forced": False,
            "distinct_draws": False,
            "n_distinct_features": len(flat),
        }
        return draws, info

    raise ValueError(f"unknown sampling strategy: {sampling!r}")


def run_one_sweep(
    model: Any,
    transcoders: dict[int, Any],
    input_ids: Any,
    feature_keys: list[FeatureKey],
    magnitudes: list[float],
    layers: list[int],
    target_token_id: int | None,
    target_pos: int,
) -> tuple[list[dict[str, float]], int, float]:
    """Sweep one feature set over magnitudes; return per-magnitude target stats."""
    from biology_server_t_lens.tl_intervention import (
        FeatureIntervention,
        run_feature_intervention,
    )

    rows: list[dict[str, float]] = []
    resolved_token_id: int | None = target_token_id
    clean_prob = float("nan")
    for magnitude in magnitudes:
        interventions = [
            FeatureIntervention(layer=layer, pos=pos, feature=feature, factor=magnitude)
            for layer, pos, feature in feature_keys
        ]
        result = run_feature_intervention(
            model, transcoders, input_ids, interventions=interventions, layers=layers
        )
        clean_logits = result.clean_logits[target_pos]
        intervened_logits = result.intervened_logits[target_pos]
        logit_diff = intervened_logits - clean_logits
        clean_probs = tensor_probs(clean_logits)
        intervened_probs = tensor_probs(intervened_logits)
        if resolved_token_id is None:
            resolved_token_id = int(clean_logits.argmax().item())
        clean_prob = float(clean_probs[resolved_token_id].item())
        rows.append(
            {
                "magnitude": magnitude,
                "logit_delta": float(logit_diff[resolved_token_id].item()),
                "prob_delta": float(
                    (intervened_probs[resolved_token_id] - clean_probs[resolved_token_id]).item()
                ),
                "max_abs_logit_delta": float(logit_diff.abs().max().item()),
            }
        )
    if resolved_token_id is None:
        raise RuntimeError("no magnitudes swept; cannot resolve target token")
    return rows, resolved_token_id, clean_prob


def _empirical_p(targeted: float, samples: np.ndarray) -> float:
    """One-sided empirical p: fraction of baseline at least as extreme, same sign."""
    if samples.size == 0:
        return float("nan")
    if targeted >= 0:
        return float(np.mean(samples >= targeted))
    return float(np.mean(samples <= targeted))


def aggregate(targeted_value: float, samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    pct = np.percentile(arr, PERCENTILES) if arr.size else [float("nan")] * len(PERCENTILES)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    mean = float(arr.mean()) if arr.size else float("nan")
    z = (targeted_value - mean) / std if std > 0 else float("nan")
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "percentiles": {str(p): float(v) for p, v in zip(PERCENTILES, pct, strict=True)},
        "targeted": float(targeted_value),
        "z_score": float(z),
        "empirical_p": _empirical_p(targeted_value, arr),
    }


def default_output_path(graph_path: Path, supernode_name: str) -> Path:
    from biology_server.attribution import slugify

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = slugify(supernode_name)
    return graph_path.parent / f"{graph_path.stem}__{name}__baseline-bootstrap__{stamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_json", type=Path, help="Attribution graph JSON.")
    parser.add_argument("supernode", help="Exact qParams supernode label to baseline.")
    parser.add_argument("--magnitudes", default=DEFAULT_MAGNITUDES)
    parser.add_argument("--layers", default=None, help="Comma-separated tracked layers.")
    parser.add_argument(
        "-n",
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of random size-matched draws. Default 100.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=("matched-cell", "global-active"),
        default="matched-cell",
        help="matched-cell: same (layer,pos) footprint; global-active: K from pooled cells.",
    )
    parser.add_argument(
        "--min-activation",
        type=float,
        default=1e-6,
        help="Activation threshold for a feature to enter the candidate pool.",
    )
    parser.add_argument(
        "--match-magnitude-tol",
        type=float,
        default=None,
        help="If set, restrict candidates to clean activation within this fractional band.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--target-token-id", type=int, default=None)
    parser.add_argument("--target-pos", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply Qwen chat template before tokenization.",
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

    constituent_keys: list[FeatureKey] = []
    for node_id in raw_node_ids:
        if not is_graph_feature_node(nodes_by_id.get(node_id)):
            continue
        try:
            layer, feature, pos = parse_feature_node_id(node_id)
        except (TypeError, ValueError):
            continue
        constituent_keys.append((layer, pos, feature))
    if not constituent_keys:
        raise ValueError(f"supernode {supernode_label!r} contains no feature nodes")

    cells = {(layer, pos) for layer, pos, _feature in constituent_keys}
    if args.layers is None:
        layers = sorted({layer for layer, _pos, _feature in constituent_keys})
    else:
        layers = sorted(parse_csv_ints(args.layers))
    missing = sorted({layer for layer, _pos, _feature in constituent_keys} - set(layers))
    if missing:
        raise ValueError(f"supernode contains layers not in --layers: {missing}")

    magnitudes = parse_csv_floats(args.magnitudes)
    graph_target_token_id, graph_target_pos = primary_logit_target(graph)
    target_token_id = (
        args.target_token_id if args.target_token_id is not None else graph_target_token_id
    )
    target_pos = args.target_pos if args.target_pos is not None else graph_target_pos
    if target_pos is None:
        target_pos = -1

    LOGGER.info("graph=%s", graph_path)
    LOGGER.info("prompt=%r", prompt)
    LOGGER.info(
        "supernode=%r size=%d cells=%d sampling=%s n_bootstrap=%d",
        supernode_label,
        len(constituent_keys),
        len(cells),
        args.sampling,
        args.n_bootstrap,
    )

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

    model = load_replacement_model(args.model_id, device=device, dtype=dtype, cache_dir=CACHE_DIR)
    transcoders = load_transcoders(layers, device=device, dtype=dtype)

    # Build candidate pool from a single clean forward, then draw bootstraps.
    clean_features = capture_clean_features(model, transcoders, input_ids, layers)
    pool = build_candidate_pool(clean_features, cells, set(constituent_keys), args.min_activation)
    pool_sizes = {f"{layer}_{pos}": len(c) for (layer, pos), c in pool.items()}
    LOGGER.info("candidate pool per cell: %s", pool_sizes)
    thin = [cell for cell, c in pool.items() if len(c) < 2]
    if thin:
        LOGGER.warning("cells with <2 candidates (will sample with replacement): %s", thin)

    rng = np.random.default_rng(args.seed)
    draws, draw_info = sample_draws(
        constituent_keys,
        clean_features,
        pool,
        n=args.n_bootstrap,
        sampling=args.sampling,
        match_magnitude_tol=args.match_magnitude_tol,
        rng=rng,
    )

    start = time.time()
    LOGGER.info("running targeted sweep (%d features)", len(constituent_keys))
    targeted_rows, resolved_token_id, clean_prob = run_one_sweep(
        model,
        transcoders,
        input_ids,
        constituent_keys,
        magnitudes,
        layers,
        target_token_id,
        target_pos,
    )

    # Collect baseline draws: per-magnitude lists of each metric.
    metrics = ("logit_delta", "prob_delta", "max_abs_logit_delta")
    baseline: dict[float, dict[str, list[float]]] = {
        m: {metric: [] for metric in metrics} for m in magnitudes
    }
    for i, draw in enumerate(draws):
        if i % 10 == 0:
            LOGGER.info("baseline draw %d/%d", i, len(draws))
        rows, _tok, _cp = run_one_sweep(
            model, transcoders, input_ids, draw, magnitudes, layers, resolved_token_id, target_pos
        )
        for row in rows:
            for metric in metrics:
                baseline[row["magnitude"]][metric].append(row[metric])

    targeted_by_m = {row["magnitude"]: row for row in targeted_rows}
    results: list[dict[str, Any]] = []
    for m in magnitudes:
        targ = targeted_by_m[m]
        results.append(
            {
                "magnitude": m,
                "targeted": {metric: targ[metric] for metric in metrics},
                "baseline": {
                    metric: {
                        **aggregate(targ[metric], baseline[m][metric]),
                        "samples": baseline[m][metric],
                    }
                    for metric in metrics
                },
            }
        )

    target_token = tokenizer.decode([resolved_token_id])
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
            "supernode": supernode_label,
            "supernode_size": len(constituent_keys),
            "constituent_node_ids": raw_node_ids,
            "cells": sorted(f"{layer}_{pos}" for layer, pos in cells),
            "layers": layers,
            "magnitudes": magnitudes,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "sampling": args.sampling,
            "min_activation": args.min_activation,
            "match_magnitude_tol": args.match_magnitude_tol,
            "candidate_pool_sizes": pool_sizes,
            "draw_space": draw_info,
            "steering_mode": "factor_times_clean_activation",
            "target": {
                "token_id": resolved_token_id,
                "token": target_token,
                "pos": target_pos,
                "clean_prob": clean_prob,
            },
            "model_id": args.model_id,
            "device": str(device),
            "dtype": str(dtype),
            "elapsed_seconds": time.time() - start,
        },
        "draws": [[f"{layer}_{feature}_{pos}" for layer, pos, feature in draw] for draw in draws],
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    LOGGER.info("wrote %s", out_path)
    LOGGER.info(
        "target token=%r id=%s clean_prob=%.6f", target_token, resolved_token_id, clean_prob
    )
    for row in results:
        ld = row["baseline"]["logit_delta"]
        LOGGER.info(
            "m=%s targeted_logit_delta=%+.4f baseline_mean=%+.4f std=%.4f "
            "p5..p95=[%+.4f,%+.4f] empirical_p=%.3f z=%.2f",
            row["magnitude"],
            row["targeted"]["logit_delta"],
            ld["mean"],
            ld["std"],
            ld["percentiles"]["5"],
            ld["percentiles"]["95"],
            ld["empirical_p"],
            ld["z_score"],
        )


if __name__ == "__main__":
    main()
