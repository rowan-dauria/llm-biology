"""Random size-matched baseline for a supernode intervention sweep.

Companion to ``llm_biology.interventions.sweep``. That script steers the
*real* constituents of a named supernode across magnitudes; this one asks
whether the observed effect is special by comparing it to ``N`` random
size-matched feature sets drawn from the same population.

Two families of baseline are supported.

*Matched random-direction baselines* (``--sampling gaussian-direction``, the
default, or ``gaussian-rotation``) test the semantic *direction* itself rather than other
features (cf. Alessandro 2026-06-19). The real supernode perturbation at each
cell is the additive residual write ``u_cell = Σ_f a_f·W_dec[f]`` (clean
activation × decoder vector), swept as ``(m-1)·u_cell``; both the targeted and
baseline arms inject through the *same* additive ``residual_writes`` path so they
propagate identically. Each random draw replaces ``u_cell`` with a norm-matched
random vector:

- ``gaussian-direction``: isotropic Gaussian unit vector(s) scaled to match the
  real norm — per cell (``--match per-cell``, default; one draw matched to the
  cell's summed norm) or per feature (``--match per-feature``; sum of per-feature
  norm-matched draws, which under-shoots for correlated cells).
- ``gaussian-rotation``: a random rotation of the supernode's constituent vectors
  into a random subspace, preserving every per-cell norm *and* the internal
  angles between constituents exactly (the geometry-preserving control).

The *other-feature baselines* (``--sampling matched-cell`` or ``global-active``)
are kept as a secondary control: they draw ``N`` size-matched sets of features
**active at the supernode's cells** and clamp them like the real supernode,
testing "is the supernode special vs *other learned features*" rather than vs an
arbitrary direction.

For the random-direction modes we also record a per-cell norm check comparing
each cell's real summed norm ``‖Σ a_f W_dec[f]‖`` with the orthogonal
expectation ``√(Σ‖a_f W_dec[f]‖²)``; a ratio near 1 means per-feature matching
already matches the summed perturbation.

Each draw is swept over the same magnitudes; per magnitude we report the
targeted effect, the baseline distribution (mean/std, percentiles) and a
one-sided empirical p-value. Plot the targeted ``logit_delta(m)`` curve against
the baseline percentile band (use the spread, not SEM): if the supernode sits
outside the band, it is doing more than a norm-matched random perturbation at
the same tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from llm_biology.interventions.common import (
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


def parse_cell_tokens(raw: str) -> set[tuple[int, int]]:
    """Parse comma-separated ``layer_pos`` cell tokens (e.g. ``"33_10,24_9"``)."""
    cells: set[tuple[int, int]] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        parts = token.split("_")
        if len(parts) != 2:
            raise ValueError(f"not a layer_pos cell token: {token!r}")
        layer, pos = (int(part) for part in parts)
        cells.add((layer, pos))
    if not cells:
        raise ValueError("expected at least one layer_pos cell token")
    return cells


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

    from llm_biology.model.tl_forward import ensure_replacement_mlp_hooks

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


# --- Matched random-direction baselines ------------------------------------
# A cell write is one (layer, pos) -> d_model vector. The real supernode writes
# u_cell = Σ_f a_f W_dec[f]; random draws replace u_cell with a norm-matched
# random vector. ``run_one_sweep_writes`` then sweeps (m-1)*u_cell additively.
CellKey = tuple[int, int]


def constituent_decoder_vectors(
    transcoders: dict[int, Any],
    clean_features: dict[int, Any],
    constituent_keys: list[FeatureKey],
) -> dict[FeatureKey, tuple[float, np.ndarray]]:
    """Map each constituent to ``(clean_activation, decoder_vector)``.

    ``decoder_vector`` is the transcoder row ``W_dec[feature]`` (a ``d_model``
    direction); ``a_f · decoder_vector`` is the feature's clean residual write.
    """
    per_feature: dict[FeatureKey, tuple[float, np.ndarray]] = {}
    w_dec_cache: dict[int, np.ndarray] = {}
    for layer, pos, feature in constituent_keys:
        if layer not in w_dec_cache:
            w_dec_cache[layer] = transcoders[layer].W_dec.detach().float().cpu().numpy()
        act = float(clean_features[layer][pos, feature])
        per_feature[(layer, pos, feature)] = (act, w_dec_cache[layer][feature])
    return per_feature


def _cells_from_per_feature(
    per_feature: dict[FeatureKey, tuple[float, np.ndarray]],
) -> dict[CellKey, list[tuple[float, np.ndarray]]]:
    cells: dict[CellKey, list[tuple[float, np.ndarray]]] = {}
    for (layer, pos, _feature), member in per_feature.items():
        cells.setdefault((layer, pos), []).append(member)
    return cells


def real_cell_writes(
    per_feature: dict[FeatureKey, tuple[float, np.ndarray]],
) -> dict[CellKey, np.ndarray]:
    """Per-cell real perturbation base vector ``u_cell = Σ_f a_f W_dec[f]``."""
    return {
        cell: np.sum([act * vec for act, vec in members], axis=0)
        for cell, members in _cells_from_per_feature(per_feature).items()
    }


def per_cell_norm_check(
    per_feature: dict[FeatureKey, tuple[float, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """Compare each cell's real summed norm with the orthogonal expectation.

    ``real_summed_norm = ‖Σ a_f W_dec[f]‖`` vs ``orthogonal_expected_norm =
    √(Σ‖a_f W_dec[f]‖²)`` (what near-orthogonal per-feature random draws sum to).
    ``ratio`` near 1 means per-feature norm matching already matches the summed
    perturbation; far from 1 flags correlated constituents in that cell.
    """
    out: dict[str, dict[str, Any]] = {}
    for (layer, pos), members in _cells_from_per_feature(per_feature).items():
        feat_vecs = [act * vec for act, vec in members]
        real = float(np.linalg.norm(np.sum(feat_vecs, axis=0)))
        sqrt_sum_sq = float(np.sqrt(sum(float(np.dot(v, v)) for v in feat_vecs)))
        out[f"{layer}_{pos}"] = {
            "n_features": len(members),
            "real_summed_norm": real,
            "orthogonal_expected_norm": sqrt_sum_sq,
            "ratio": (real / sqrt_sum_sq) if sqrt_sum_sq > 0 else float("nan"),
        }
    return out


def _random_unit(d_model: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.standard_normal(d_model)
    norm = float(np.linalg.norm(g))
    return g / norm if norm > 0 else g


def _rotated_cell_writes(
    per_feature: dict[FeatureKey, tuple[float, np.ndarray]],
    rng: np.random.Generator,
) -> dict[CellKey, np.ndarray]:
    """Rotate all constituent vectors into a random subspace, preserving Gram.

    Stack ``v_f = a_f W_dec[f]`` as rows of ``V``; with ``V.T = Q R`` we have
    ``V V^T = R^T R``. Mapping the coordinates ``R^T`` onto a fresh random
    orthonormal frame ``Qn`` gives ``V' = R^T Qn^T`` with ``V' V'^T = R^T R`` —
    identical norms and inter-vector angles, random orientation/subspace.
    """
    keys = list(per_feature)
    rows = np.stack([per_feature[k][0] * per_feature[k][1] for k in keys])  # (k, d_model)
    d_model = rows.shape[1]
    _q, r = np.linalg.qr(rows.T)  # r: (k, k)
    frame, _ = np.linalg.qr(rng.standard_normal((d_model, r.shape[0])))  # (d_model, k)
    rotated = r.T @ frame.T  # (k, d_model)
    draw: dict[CellKey, np.ndarray] = {}
    for (layer, pos, _feature), row in zip(keys, rotated, strict=True):
        cell = (layer, pos)
        draw[cell] = draw[cell] + row if cell in draw else row.copy()
    return draw


def sample_gaussian_writes(
    per_feature: dict[FeatureKey, tuple[float, np.ndarray]],
    *,
    n: int,
    match: str,
    rotation: bool,
    rng: np.random.Generator,
) -> list[dict[CellKey, np.ndarray]]:
    """Draw ``n`` random per-cell base writes (m-independent, scaled by (m-1))."""
    cells = _cells_from_per_feature(per_feature)
    d_model = next(iter(per_feature.values()))[1].shape[0]
    draws: list[dict[CellKey, np.ndarray]] = []
    for _ in range(n):
        if rotation:
            draws.append(_rotated_cell_writes(per_feature, rng))
            continue
        draw: dict[CellKey, np.ndarray] = {}
        for cell, members in cells.items():
            if match == "per-cell":
                target_norm = float(np.linalg.norm(np.sum([a * v for a, v in members], axis=0)))
                draw[cell] = _random_unit(d_model, rng) * target_norm
            else:  # per-feature: norm-match each constituent, then sum
                acc = np.zeros(d_model)
                for act, vec in members:
                    acc = acc + _random_unit(d_model, rng) * float(np.linalg.norm(act * vec))
                draw[cell] = acc
        draws.append(draw)
    return draws


def run_one_sweep_writes(
    model: Any,
    transcoders: dict[int, Any],
    input_ids: Any,
    base_writes: dict[CellKey, np.ndarray],
    magnitudes: list[float],
    layers: list[int],
    target_token_id: int | None,
    target_pos: int,
) -> tuple[list[dict[str, float]], int, float]:
    """Sweep one set of additive cell writes over magnitudes.

    At magnitude ``m`` the write at each cell is ``(m-1)·base_write`` (so ``m=1``
    reproduces clean, matching the multiplicative-factor x-axis of the other
    baselines). Mirrors ``run_one_sweep``'s per-magnitude target stats.
    """
    import torch

    from llm_biology.interventions.tl_intervention import run_feature_intervention

    rows: list[dict[str, float]] = []
    resolved_token_id: int | None = target_token_id
    clean_prob = float("nan")
    for magnitude in magnitudes:
        scale = float(magnitude) - 1.0
        residual_writes = {
            cell: torch.from_numpy(np.ascontiguousarray(scale * base, dtype=np.float32))
            for cell, base in base_writes.items()
        }
        result = run_feature_intervention(
            model,
            transcoders,
            input_ids,
            interventions=[],
            layers=layers,
            residual_writes=residual_writes,
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
    from llm_biology.interventions.tl_intervention import (
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


def default_output_path(graph_path: Path, supernode_name: str, tag: str | None = None) -> Path:
    from llm_biology.attribution.attribution import slugify

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = slugify(supernode_name)
    suffix = f"__{tag}" if tag else ""
    return (
        graph_path.parent / f"{graph_path.stem}__{name}__baseline-bootstrap{suffix}__{stamp}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_json", type=Path, help="Attribution graph JSON.")
    parser.add_argument("supernode", help="Exact qParams supernode label to baseline.")
    parser.add_argument("--magnitudes", default=DEFAULT_MAGNITUDES)
    parser.add_argument("--layers", default=None, help="Comma-separated tracked layers.")
    parser.add_argument(
        "--restrict-cells",
        default=None,
        help=(
            "Comma-separated layer_pos cells (e.g. '33_10' or '12_9,24_9,24_10') to "
            "isolate for a per-cell decomposition: only the supernode constituents in "
            "these cells are steered, and baseline draws match only that sub-footprint."
        ),
    )
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
        choices=("matched-cell", "global-active", "gaussian-direction", "gaussian-rotation"),
        default="gaussian-direction",
        help=(
            "Baseline family (default: gaussian-direction). gaussian-direction/"
            "gaussian-rotation: norm-matched random residual-write directions controlling "
            "for the semantic direction itself. matched-cell/global-active: other learned "
            "features (same footprint / K from pooled cells)."
        ),
    )
    parser.add_argument(
        "--match",
        choices=("per-feature", "per-cell"),
        default="per-cell",
        help=(
            "Norm-matching granularity for gaussian-direction (default: per-cell). per-cell "
            "matches each cell's summed norm ‖Σ a_f W_dec[f]‖ exactly; per-feature matches "
            "each constituent's ‖a_f W_dec[f]‖ then sums (under-shoots for correlated cells). "
            "Ignored for other --sampling modes."
        ),
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

    from transformers import AutoTokenizer

    from llm_biology.attribution.attribution import (
        CACHE_DIR,
        load_transcoders,
        pick_device_dtype,
        prepend_special_prefix,
    )
    from llm_biology.model.tl_model import load_replacement_model

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

    restrict_cells: list[str] | None = None
    if args.restrict_cells is not None:
        keep = parse_cell_tokens(args.restrict_cells)
        all_cells = {(layer, pos) for layer, pos, _feature in constituent_keys}
        unknown = sorted(keep - all_cells)
        if unknown:
            available = sorted(f"{c[0]}_{c[1]}" for c in all_cells)
            raise ValueError(
                f"--restrict-cells contains cells not in supernode {supernode_label!r}: "
                f"{[f'{c[0]}_{c[1]}' for c in unknown]}; available: {available}"
            )
        n_full = len(constituent_keys)
        constituent_keys = [k for k in constituent_keys if (k[0], k[1]) in keep]
        restrict_cells = sorted(f"{c[0]}_{c[1]}" for c in keep)
        LOGGER.info(
            "restrict-cells %s: keeping %d/%d constituents",
            restrict_cells,
            len(constituent_keys),
            n_full,
        )

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

    # One clean forward feeds both baseline families.
    clean_features = capture_clean_features(model, transcoders, input_ids, layers)
    rng = np.random.default_rng(args.seed)

    gaussian = args.sampling in ("gaussian-direction", "gaussian-rotation")
    pool_sizes: dict[str, int] = {}
    draw_info: dict[str, Any] = {}
    norm_check: dict[str, dict[str, Any]] = {}
    real_cell_norms: dict[str, float] = {}
    draws_serialized: list[list[str]] | None

    if gaussian:
        # Matched random-direction baseline: real perturbation is the additive
        # residual write u_cell = Σ a_f W_dec[f]; random draws norm-match it.
        per_feature = constituent_decoder_vectors(transcoders, clean_features, constituent_keys)
        base_writes_real = real_cell_writes(per_feature)
        real_cell_norms = {
            f"{layer}_{pos}": float(np.linalg.norm(v))
            for (layer, pos), v in base_writes_real.items()
        }
        norm_check = per_cell_norm_check(per_feature)
        LOGGER.info("per-cell summed-norm check (real vs orthogonal expectation):")
        for cell_name, info in norm_check.items():
            LOGGER.info(
                "  cell %s n=%d real=%.4f orthog=%.4f ratio=%.3f",
                cell_name,
                info["n_features"],
                info["real_summed_norm"],
                info["orthogonal_expected_norm"],
                info["ratio"],
            )
        baseline_items: list[Any] = sample_gaussian_writes(
            per_feature,
            n=args.n_bootstrap,
            match=args.match,
            rotation=(args.sampling == "gaussian-rotation"),
            rng=rng,
        )
        targeted_item: Any = base_writes_real
        draw_info = {
            "kind": args.sampling,
            "match": args.match if args.sampling == "gaussian-direction" else None,
            "n_cells": len(base_writes_real),
        }
        draws_serialized = None

        def sweep(item: Any, tok: int | None) -> tuple[list[dict[str, float]], int, float]:
            return run_one_sweep_writes(
                model, transcoders, input_ids, item, magnitudes, layers, tok, target_pos
            )
    else:
        pool = build_candidate_pool(
            clean_features, cells, set(constituent_keys), args.min_activation
        )
        pool_sizes = {f"{layer}_{pos}": len(c) for (layer, pos), c in pool.items()}
        LOGGER.info("candidate pool per cell: %s", pool_sizes)
        thin = [cell for cell, c in pool.items() if len(c) < 2]
        if thin:
            LOGGER.warning("cells with <2 candidates (will sample with replacement): %s", thin)
        baseline_items, draw_info = sample_draws(
            constituent_keys,
            clean_features,
            pool,
            n=args.n_bootstrap,
            sampling=args.sampling,
            match_magnitude_tol=args.match_magnitude_tol,
            rng=rng,
        )
        targeted_item = constituent_keys
        draws_serialized = [
            [f"{layer}_{feature}_{pos}" for layer, pos, feature in draw] for draw in baseline_items
        ]

        def sweep(item: Any, tok: int | None) -> tuple[list[dict[str, float]], int, float]:
            return run_one_sweep(
                model, transcoders, input_ids, item, magnitudes, layers, tok, target_pos
            )

    start = time.time()
    LOGGER.info("running targeted sweep (%d features)", len(constituent_keys))
    targeted_rows, resolved_token_id, clean_prob = sweep(targeted_item, target_token_id)

    # Collect baseline draws: per-magnitude lists of each metric.
    metrics = ("logit_delta", "prob_delta", "max_abs_logit_delta")
    baseline: dict[float, dict[str, list[float]]] = {
        m: {metric: [] for metric in metrics} for m in magnitudes
    }
    for i, item in enumerate(baseline_items):
        if i % 10 == 0:
            LOGGER.info("baseline draw %d/%d", i, len(baseline_items))
        rows, _tok, _cp = sweep(item, resolved_token_id)
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
    tag = ("cells-" + "-".join(restrict_cells)) if restrict_cells else None
    out_path = (
        (args.output or default_output_path(graph_path, supernode_label, tag))
        .expanduser()
        .resolve()
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
            "restrict_cells": restrict_cells,
            "layers": layers,
            "magnitudes": magnitudes,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "sampling": args.sampling,
            "match": args.match if args.sampling == "gaussian-direction" else None,
            "min_activation": args.min_activation,
            "match_magnitude_tol": args.match_magnitude_tol,
            "candidate_pool_sizes": pool_sizes,
            "draw_space": draw_info,
            "steering_mode": (
                "residual_write_additive" if gaussian else "factor_times_clean_activation"
            ),
            "norm_check": norm_check or None,
            "real_cell_norms": real_cell_norms or None,
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
        "draws": draws_serialized,
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
