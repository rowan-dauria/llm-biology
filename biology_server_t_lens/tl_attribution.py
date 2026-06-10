"""Attribution rows on a linearised TransformerLens model.

Three layers of API, all consuming a populated :class:`HookState`:

- :func:`attribute_logit_row` — one backward from ``state.final_hidden`` with a
  demeaned-unembed gradient at a single ``(0, pos, :)`` slot. ``final_hidden``
  is the post-final-norm, pre-unembed residual. (Chunk 4.)
- :func:`attribute_feature_row` — one backward from ``state.mlp_inputs[layer]``
  with an encoder vector at a single ``(0, pos, :)`` slot. Source attribution
  for a single feature target. (Chunk 5.) Intentionally slow; correctness
  oracle for the batched version.
- :class:`AttributionContext` / :func:`setup_attribution` — batched backward
  via ``input_ids.expand(batch_size, -1)`` plus per-slot ``register_hook``
  injection. ``batch_size`` rows in one ``.backward()``. (Chunk 6.)

All three return ``(feature_scores, error_scores, embedding_scores)`` over the
active source nodes captured by the forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer

from biology_server_t_lens.tl_forward import (
    ActiveFeature,
    HookState,
    finalize_active_features,
    install_transcoder_hooks,
)

DEFAULT_SCORE_CHUNK_SIZE = 32_768

__all__ = [
    "AttributionContext",
    "TargetSpec",
    "attribute_feature_row",
    "attribute_logit_row",
    "build_dense_edge_matrix_serial",
    "collect_embedding_scores",
    "collect_error_scores",
    "collect_feature_scores",
    "compute_partial_influences",
    "demeaned_unembed_vector",
    "run_attribution",
    "run_attribution_from_context",
    "setup_attribution",
    "total_error_nodes",
    "total_active_features",
]


def total_active_features(state: HookState) -> int:
    return sum(state.layer_features[layer].feature_ids.numel() for layer in state.layers)


def _n_pos_from_state(state: HookState) -> int:
    if state.token_vectors is not None:
        return int(state.token_vectors.shape[0])
    if state.final_hidden is not None:
        return int(state.final_hidden.shape[1])
    raise RuntimeError("Position count is unavailable; run forward first")


def total_error_nodes(state: HookState) -> int:
    return len(state.layers) * _n_pos_from_state(state)


def demeaned_unembed_vector(W_U: torch.Tensor, token_id: int, dtype: torch.dtype) -> torch.Tensor:
    """Demeaned unembed direction for ``token_id``.

    TL's ``W_U`` is ``(d_model, d_vocab)``: column ``t`` is the unembed vec for
    that token. We subtract the mean over the vocab dim so attribution is for
    the logit *relative to* the average vocab token (Methods §3.3).
    """
    col = W_U[:, token_id]
    return (col - W_U.mean(dim=1)).to(dtype)


def collect_feature_scores(
    state: HookState,
    n_features: int,
    *,
    layers: Sequence[int] | None = None,
    batch_index: int = 0,
    chunk_size: int = DEFAULT_SCORE_CHUNK_SIZE,
) -> torch.Tensor:
    """Contract each tracked layer's MLP-out grad with its active decoder vecs.

    ``decoder_vectors`` are already activation-scaled (see ``layer_feature_data``),
    so the result is the eqn-7 / eqn-8 source score for each active feature.
    Returns zeros for layers that didn't receive a backward (legitimate when
    the target is *upstream* of that layer).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    score_layers = list(state.layers) if layers is None else list(layers)
    scores: torch.Tensor | None = None

    for layer in score_layers:
        data = state.layer_features[layer]
        grad = state.output_grads.get(layer)
        if grad is None or data.feature_ids.numel() == 0:
            continue
        if scores is None:
            scores = torch.zeros(
                n_features,
                dtype=data.decoder_vectors.dtype,
                device=grad.device,
            )
        assert scores is not None
        n_layer_features = int(data.feature_ids.numel())
        layer_slice = scores[data.start : data.end]
        for start in range(0, n_layer_features, chunk_size):
            end = min(start + chunk_size, n_layer_features)
            positions = data.positions[start:end].to(grad.device)
            grad_at_positions = grad[batch_index].index_select(0, positions)
            layer_slice[start:end] = (
                grad_at_positions.to(data.decoder_vectors.dtype) * data.decoder_vectors[start:end]
            ).sum(dim=-1)

    if scores is None:
        device = state.embedding.device if state.embedding is not None else torch.device("cpu")
        return torch.zeros(n_features, device=device)
    return scores


def collect_error_scores(
    state: HookState,
    *,
    layers: Sequence[int] | None = None,
    batch_index: int = 0,
) -> torch.Tensor:
    """Contract tracked-layer MLP-out grads with reconstruction-error vectors."""

    n_pos = _n_pos_from_state(state)
    n_errors = len(state.layers) * n_pos
    score_layers = list(state.layers) if layers is None else list(layers)
    layer_to_offset = {layer: idx * n_pos for idx, layer in enumerate(state.layers)}
    scores: torch.Tensor | None = None

    for layer in score_layers:
        offset = layer_to_offset.get(layer)
        grad = state.output_grads.get(layer)
        error = state.error_vectors.get(layer)
        if offset is None or grad is None or error is None:
            continue
        if scores is None:
            scores = torch.zeros(
                n_errors,
                dtype=error.dtype,
                device=grad.device,
            )
        assert scores is not None
        error_at_positions = error.to(device=grad.device, dtype=scores.dtype)
        scores[offset : offset + n_pos] = (
            grad[batch_index].to(scores.dtype) * error_at_positions
        ).sum(dim=-1)

    if scores is None:
        device = state.embedding.device if state.embedding is not None else torch.device("cpu")
        return torch.zeros(n_errors, device=device)
    return scores


def collect_embedding_scores(state: HookState, *, batch_index: int = 0) -> torch.Tensor:
    if state.embedding_grad is None or state.token_vectors is None:
        raise RuntimeError("Embedding gradients were not captured")
    return (
        state.embedding_grad[batch_index].to(state.token_vectors.dtype) * state.token_vectors
    ).sum(dim=-1)


def detached_logits(model: HookedTransformer, input_ids: torch.Tensor) -> torch.Tensor:
    """Do a forward pass of the transcoder-subbed model and return the output logits"""

    with torch.no_grad():
        return model(input_ids).detach()


def attribute_logit_row(
    model: HookedTransformer,
    state: HookState,
    *,
    token_id: int,
    pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-target logit attribution from post-final-norm residual."""
    fh = state.final_hidden
    if fh is None:
        raise RuntimeError("Final hidden state was not captured; run forward first")

    state.clear_grads()
    unembed_vec = demeaned_unembed_vector(model.W_U, token_id, fh.dtype).to(fh.device)
    gradient = torch.zeros_like(fh)
    gradient[0, pos] = unembed_vec
    fh.backward(gradient=gradient, retain_graph=True)

    n_features = total_active_features(state)
    feature_scores = collect_feature_scores(state, n_features)
    error_scores = collect_error_scores(state)
    embedding_scores = collect_embedding_scores(state)
    return feature_scores, error_scores, embedding_scores


def attribute_feature_row(
    model: HookedTransformer,  # noqa: ARG001 - kept for symmetry with logit row
    state: HookState,
    *,
    target_layer: int,
    target_pos: int,
    encoder_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-target feature attribution.

    Injects ``encoder_vector`` as the synthetic gradient at
    ``state.mlp_inputs[target_layer][0, target_pos, :]`` and calls
    ``mlp_input.backward(...)``. Because the model is linearised by
    :func:`install_freezes`, this is exactly the eqn-7 / eqn-8 source row.

    Only source layers strictly upstream of ``target_layer`` receive gradient;
    same-layer or downstream layers' entries in the returned vector are zero.
    """
    mlp_in = state.mlp_inputs.get(target_layer)
    if mlp_in is None:
        raise KeyError(
            f"mlp_input not captured for layer {target_layer}; "
            "target_layer must be one of state.layers"
        )

    state.clear_grads()
    gradient = torch.zeros_like(mlp_in)
    gradient[0, target_pos] = encoder_vector.to(device=mlp_in.device, dtype=mlp_in.dtype)
    mlp_in.backward(gradient=gradient, retain_graph=True)

    n_features = total_active_features(state)
    # Restrict source layers to those strictly upstream of the target.
    upstream = [layer for layer in state.layers if layer < target_layer]
    feature_scores = collect_feature_scores(state, n_features, layers=upstream)
    error_scores = collect_error_scores(state, layers=upstream)
    embedding_scores = collect_embedding_scores(state)
    return feature_scores, error_scores, embedding_scores


# ---------------------------------------------------------------------------
# Chunk 5: serial oracle for the batched version.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One row to attribute.

    Either a logit target (``kind='logit'``) — inject a demeaned unembed vec at
    ``state.final_hidden[0, pos, :]`` — or a feature target (``kind='feature'``)
    — inject ``encoder_vector`` at ``state.mlp_inputs[layer][0, pos, :]``.
    """

    kind: str  # "logit" | "feature"
    pos: int
    layer: int | None = None
    token_id: int | None = None
    encoder_vector: torch.Tensor | None = None
    prob: float = 1.0


def build_dense_edge_matrix_serial(
    model: HookedTransformer,
    state: HookState,
    targets: Sequence[TargetSpec],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loop ``attribute_*_row`` once per target. Slow; used as oracle."""
    n_features = total_active_features(state)
    n_errors = total_error_nodes(state)
    n_pos = _n_pos_from_state(state)
    feature_matrix = torch.zeros(len(targets), n_features)
    error_matrix = torch.zeros(len(targets), n_errors)
    embedding_matrix = torch.zeros(len(targets), n_pos)

    for row_idx, target in enumerate(targets):
        if target.kind == "logit":
            assert target.token_id is not None
            fs, errs, es = attribute_logit_row(
                model, state, token_id=target.token_id, pos=target.pos
            )
        elif target.kind == "feature":
            assert target.layer is not None and target.encoder_vector is not None
            fs, errs, es = attribute_feature_row(
                model,
                state,
                target_layer=target.layer,
                target_pos=target.pos,
                encoder_vector=target.encoder_vector,
            )
        else:
            raise ValueError(f"Unknown target kind {target.kind!r}")
        feature_matrix[row_idx] = fs.detach().to(feature_matrix.dtype)
        error_matrix[row_idx] = errs.detach().to(error_matrix.dtype)
        embedding_matrix[row_idx] = es.detach().to(embedding_matrix.dtype)

    return feature_matrix, error_matrix, embedding_matrix


# ---------------------------------------------------------------------------
# Chunk 6: batched backward.
# ---------------------------------------------------------------------------


class AttributionContext:
    """One forward pass with ``batch_size`` copies of the prompt, ready to
    serve many attribution rows from a single backward each.

    The forward expands ``input_ids`` to ``(batch_size, n_pos)`` so every cached
    residual lives on the leading batch axis. To attribute a batch of targets,
    :meth:`compute_batch` installs one transient ``register_hook`` per layer
    that — when backward reaches that ``mlp_input`` — overrides slot ``k``'s
    gradient with the injection vector for the target landing on that slot.
    A single ``.backward()`` from ``final_hidden`` (with a per-slot
    demeaned-unembed seed for logit-typed slots) then populates all rows.
    """

    def __init__(
        self,
        model: HookedTransformer,
        state: HookState,
        batch_size: int,
        logits: torch.Tensor,
    ) -> None:
        if state.final_hidden is None:
            raise RuntimeError("setup_attribution must be called before compute_batch")
        self.model = model
        self.state = state
        self.batch_size = batch_size
        self.logits = logits.detach()
        self.n_features = total_active_features(state)
        self.n_errors = total_error_nodes(state)
        self.n_pos = state.token_vectors.shape[0] if state.token_vectors is not None else 0

    def compute_batch(
        self,
        targets: Sequence[TargetSpec],
        *,
        retain_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attribute up to ``batch_size`` targets in one backward.

        Single-backward mechanic:

        - ``_make_mlp_out_hook`` wraps each tracked layer's MLP output as
          ``mlp_input * 0 + reconstruction.detach()``. The ``* 0`` keeps
          ``state.mlp_inputs[layer]`` reachable from a backward sweep started
          at ``state.final_hidden`` — natural grad propagated to it is 0
          everywhere, but PyTorch *does* visit the tensor.
        - For each feature target we ``register_hook`` on the cached
          ``mlp_input`` and **override** the grad at ``(slot_k, pos_k)`` with
          the encoder vector. For each logit target we do the same on
          ``final_hidden`` with the demeaned unembed vector. Slots without an
          override keep grad 0 on that residual.
        - One ``final_hidden.backward(zeros)`` then propagates per-slot grads
          upstream through the linearised model. Every batch slot is
          independent of the others except via frozen parameters — so
          ``batch_size`` independent rows are computed in one sweep.
        """
        if len(targets) > self.batch_size:
            raise ValueError(f"Got {len(targets)} targets, batch_size={self.batch_size}")

        state = self.state
        fh = state.final_hidden
        assert fh is not None
        state.clear_grads()

        # Group by injection site (just to factor one register_hook per
        # cached residual; behaviour is the same as one hook per target).
        logit_slots: list[tuple[int, TargetSpec]] = []
        feature_slots_by_layer: dict[int, list[tuple[int, TargetSpec]]] = {}
        for k, target in enumerate(targets):
            if target.kind == "logit":
                logit_slots.append((k, target))
            elif target.kind == "feature":
                assert target.layer is not None
                feature_slots_by_layer.setdefault(target.layer, []).append((k, target))
            else:
                raise ValueError(f"Unknown target kind {target.kind!r}")

        def _make_override(slots: list[tuple[int, TargetSpec]], inject_kind: str):
            def override(grad: torch.Tensor) -> torch.Tensor:
                new_grad = grad.clone()
                for k, target in slots:
                    if inject_kind == "logit":
                        assert target.token_id is not None
                        vec = demeaned_unembed_vector(self.model.W_U, target.token_id, grad.dtype)
                    else:
                        assert target.encoder_vector is not None
                        vec = target.encoder_vector
                    new_grad[k, target.pos] = vec.to(device=grad.device, dtype=grad.dtype)
                return new_grad

            return override

        handles = []
        if logit_slots:
            handles.append(fh.register_hook(_make_override(logit_slots, "logit")))
        for layer, slots in feature_slots_by_layer.items():
            mlp_in = state.mlp_inputs[layer]
            handles.append(mlp_in.register_hook(_make_override(slots, "feature")))

        try:
            fh.backward(
                gradient=torch.zeros_like(fh),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        feature_matrix = torch.zeros(
            len(targets), self.n_features, device=fh.device, dtype=fh.dtype
        )
        error_matrix = torch.zeros(len(targets), self.n_errors, device=fh.device, dtype=fh.dtype)
        embedding_matrix = torch.zeros(len(targets), self.n_pos, device=fh.device, dtype=fh.dtype)
        for k, target in enumerate(targets):
            if target.kind == "feature":
                assert target.layer is not None
                upstream = [layer_idx for layer_idx in state.layers if layer_idx < target.layer]
            else:
                upstream = list(state.layers)
            fs = collect_feature_scores(state, self.n_features, layers=upstream, batch_index=k)
            errs = collect_error_scores(state, layers=upstream, batch_index=k)
            es = collect_embedding_scores(state, batch_index=k)
            feature_matrix[k] = fs.to(feature_matrix.dtype)
            error_matrix[k] = errs.to(error_matrix.dtype)
            embedding_matrix[k] = es.to(embedding_matrix.dtype)
        return feature_matrix, error_matrix, embedding_matrix


def setup_attribution(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    *,
    batch_size: int,
    transcoders: dict[int, SingleLayerTranscoder],
    layers: Sequence[int],
    zero_positions: slice | None = slice(0, 1),
    base_logits: torch.Tensor | None = None,
) -> AttributionContext:
    """Run one retained forward over an expanded prompt, stopping before unembed.

    ``input_ids`` may be ``(n_pos,)`` or ``(1, n_pos)`` — both are expanded.
    Logits are computed first under ``no_grad`` for target selection, then the
    attribution forward is expanded to ``(batch_size, n_pos)`` and stopped after
    the final block. ``ln_final`` is applied manually so ``state.final_hidden``
    is the post-final-norm, pre-unembed residual.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError(f"Expected (1, n_pos) or (n_pos,), got {tuple(input_ids.shape)}")
    logits = detached_logits(model, input_ids) if base_logits is None else base_logits.detach()
    expanded = input_ids.expand(batch_size, -1).contiguous()

    state = HookState(
        layers=list(layers),
        transcoders=transcoders,
        zero_positions=zero_positions,
    )
    fwd_hooks = install_transcoder_hooks(model, transcoders, state)
    residual = model.run_with_hooks(
        expanded,
        fwd_hooks=fwd_hooks,
        stop_at_layer=model.cfg.n_layers,
    )
    state.final_hidden = model.ln_final(residual)
    finalize_active_features(state)
    return AttributionContext(model, state, batch_size, logits)


# ---------------------------------------------------------------------------
# Chunk 7: partial-graph top-K feature selection.
# ---------------------------------------------------------------------------


def compute_partial_influences(
    edge_matrix: torch.Tensor,
    logit_probabilities: torch.Tensor,
    row_to_node_index: torch.Tensor,
    *,
    max_iter: int = 128,
) -> torch.Tensor:
    """Estimate source-node influence from a partially filled adjacency matrix.

    ``edge_matrix`` is oriented as ``A[target_row, source_col]``. Only rows
    listed in ``row_to_node_index`` are currently known; each entry maps that
    row to the corresponding node index in the column space. Starting from the
    logit nodes, this walks known rows backwards through absolute, row-normalised
    edge weights and accumulates one influence score per source node.
    """
    if edge_matrix.dim() != 2:
        raise ValueError("edge_matrix must be rank-2")
    if row_to_node_index.numel() != edge_matrix.shape[0]:
        raise ValueError("row_to_node_index must have one entry per matrix row")
    if logit_probabilities.numel() > edge_matrix.shape[1]:
        raise ValueError("too many logit probabilities for edge_matrix columns")

    matrix = edge_matrix.detach().abs().to(dtype=torch.float32)
    matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
    row_to_node_index = row_to_node_index.to(device=matrix.device, dtype=torch.long)
    logit_probabilities = logit_probabilities.to(device=matrix.device, dtype=matrix.dtype)

    influences = torch.zeros(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    prod = torch.zeros_like(influences)
    prod[-logit_probabilities.numel() :] = logit_probabilities

    for _ in range(max_iter):
        prod = prod.index_select(0, row_to_node_index) @ matrix
        if not bool(prod.any().item()):
            break
        influences += prod
    else:
        raise RuntimeError("partial influence computation failed to converge")

    return influences


def _active_features_from_state(state: HookState) -> list[ActiveFeature]:
    active: list[ActiveFeature] = []
    for layer in state.layers:
        data = state.layer_features[layer]
        for local_idx in range(data.feature_ids.numel()):
            active.append(
                ActiveFeature(
                    layer=layer,
                    pos=int(data.positions[local_idx].item()),
                    feature=int(data.feature_ids[local_idx].item()),
                    activation=float(data.activations[local_idx].item()),
                    encoder_vector=data.encoder_vectors[local_idx],
                    score_index=data.start + local_idx,
                )
            )
    return active


def _as_logit_target_specs(
    logit_targets: Sequence[TargetSpec | object],
    *,
    default_pos: int,
) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for target in logit_targets:
        if isinstance(target, TargetSpec):
            if target.kind != "logit":
                raise ValueError("logit_targets must contain only logit targets")
            if target.token_id is None:
                raise ValueError("logit TargetSpec requires token_id")
            specs.append(
                TargetSpec(
                    kind="logit",
                    pos=target.pos,
                    token_id=target.token_id,
                    prob=target.prob,
                )
            )
            continue

        token_id = getattr(target, "token_id", None)
        if token_id is None:
            raise TypeError("logit target objects must expose token_id")
        specs.append(
            TargetSpec(
                kind="logit",
                pos=int(getattr(target, "pos", default_pos)),
                token_id=int(token_id),
                prob=float(getattr(target, "prob", 1.0)),
            )
        )
    return specs


def _feature_target_from_active(feature: ActiveFeature) -> TargetSpec:
    return TargetSpec(
        kind="feature",
        layer=feature.layer,
        pos=feature.pos,
        encoder_vector=feature.encoder_vector,
    )


def _pack_selected_edge_matrix(
    *,
    edge_matrix: torch.Tensor,
    selected_indices: list[int],
    n_features: int,
    n_errors: int,
    n_pos: int,
    n_logits: int,
) -> torch.Tensor:
    """Pack the temporary all-feature columns into a square selected graph.

    Final node order is ``selected features``, error nodes, token embeddings,
    then logits. Matrix orientation remains ``A[target, source]``.
    """
    n_selected = len(selected_indices)
    n_nodes = n_selected + n_errors + n_pos + n_logits
    packed = torch.zeros(n_nodes, n_nodes, dtype=edge_matrix.dtype)
    if n_nodes == 0:
        return packed

    selected = torch.tensor(selected_indices, dtype=torch.long)
    error_start = n_selected
    token_start = error_start + n_errors
    logit_start = token_start + n_pos
    all_error_start = n_features
    all_token_start = all_error_start + n_errors

    # Feature target rows were appended after the logit rows in the same order
    # as ``selected_indices``.
    for target_row in range(n_selected):
        source_row = n_logits + target_row
        if n_selected:
            packed[target_row, :n_selected] = edge_matrix[source_row].index_select(0, selected)
        packed[target_row, error_start:token_start] = edge_matrix[
            source_row, all_error_start:all_token_start
        ]
        packed[target_row, token_start:logit_start] = edge_matrix[
            source_row, all_token_start : all_token_start + n_pos
        ]

    # Logit target rows occupy the tail of the packed matrix.
    for logit_idx in range(n_logits):
        target_row = logit_start + logit_idx
        if n_selected:
            packed[target_row, :n_selected] = edge_matrix[logit_idx].index_select(0, selected)
        packed[target_row, error_start:token_start] = edge_matrix[
            logit_idx, all_error_start:all_token_start
        ]
        packed[target_row, token_start:logit_start] = edge_matrix[
            logit_idx, all_token_start : all_token_start + n_pos
        ]

    return packed


def run_attribution_from_context(
    ctx: AttributionContext,
    *,
    logit_targets: Sequence[TargetSpec | object],
    max_feature_nodes: int,
    update_interval: int = 1,
) -> tuple[list[ActiveFeature], torch.Tensor]:
    """Run logit rows plus iterative top-K feature rows from a prepared context.

    Returns ``(selected_active_features, dense_edge_matrix)``. The dense matrix
    is square with node order ``selected features``, error nodes, embeddings,
    logits and orientation ``matrix[target, source]``.
    """
    if max_feature_nodes < 0:
        raise ValueError("max_feature_nodes must be non-negative")
    if update_interval <= 0:
        raise ValueError("update_interval must be positive")

    n_features = ctx.n_features
    n_errors = ctx.n_errors
    n_pos = ctx.n_pos
    cap = min(max_feature_nodes, n_features)
    default_pos = n_pos - 1
    logit_specs = _as_logit_target_specs(logit_targets, default_pos=default_pos)
    if not logit_specs:
        raise ValueError("at least one logit target is required")
    if len(logit_specs) > ctx.batch_size:
        raise ValueError(f"{len(logit_specs)} logit targets exceed batch_size={ctx.batch_size}")

    n_logits = len(logit_specs)
    n_cols = n_features + n_errors + n_pos + n_logits
    edge_matrix = torch.zeros(n_logits + cap, n_cols, dtype=torch.float32)
    row_to_node_index = torch.empty(n_logits + cap, dtype=torch.long)

    logit_feature, logit_error, logit_embedding = ctx.compute_batch(
        logit_specs,
        retain_graph=cap > 0,
    )
    edge_matrix[:n_logits, :n_features] = logit_feature.detach().cpu().float()
    edge_matrix[:n_logits, n_features : n_features + n_errors] = logit_error.detach().cpu().float()
    edge_matrix[:n_logits, n_features + n_errors : n_features + n_errors + n_pos] = (
        logit_embedding.detach().cpu().float()
    )
    row_to_node_index[:n_logits] = torch.arange(n_logits) + n_features + n_errors + n_pos

    active_features = _active_features_from_state(ctx.state)
    visited = torch.zeros(n_features, dtype=torch.bool)
    selected_indices: list[int] = []
    next_row = n_logits
    logit_probs = torch.tensor([target.prob for target in logit_specs], dtype=torch.float32)

    while len(selected_indices) < cap:
        remaining = cap - len(selected_indices)
        if cap == n_features:
            pending = torch.arange(n_features)[~visited]
        else:
            influences = compute_partial_influences(
                edge_matrix[:next_row],
                logit_probs,
                row_to_node_index[:next_row],
            )
            feature_rank = torch.argsort(influences[:n_features], descending=True)
            queue_size = min(update_interval * ctx.batch_size, remaining)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        if pending.numel() == 0:
            break

        for start in range(0, pending.numel(), ctx.batch_size):
            batch_indices = pending[start : start + ctx.batch_size].tolist()
            targets = [
                _feature_target_from_active(active_features[int(idx)]) for idx in batch_indices
            ]
            will_finish = len(selected_indices) + len(batch_indices) >= cap
            feature_rows, error_rows, embedding_rows = ctx.compute_batch(
                targets, retain_graph=not will_finish
            )
            rows = len(batch_indices)
            edge_matrix[next_row : next_row + rows, :n_features] = (
                feature_rows.detach().cpu().float()
            )
            edge_matrix[next_row : next_row + rows, n_features : n_features + n_errors] = (
                error_rows.detach().cpu().float()
            )
            edge_matrix[
                next_row : next_row + rows,
                n_features + n_errors : n_features + n_errors + n_pos,
            ] = embedding_rows.detach().cpu().float()
            row_to_node_index[next_row : next_row + rows] = torch.tensor(
                batch_indices, dtype=torch.long
            )
            for idx in batch_indices:
                visited[int(idx)] = True
                selected_indices.append(int(idx))
            next_row += rows

    selected = [active_features[idx] for idx in selected_indices]
    dense = _pack_selected_edge_matrix(
        edge_matrix=edge_matrix[:next_row],
        selected_indices=selected_indices,
        n_features=n_features,
        n_errors=n_errors,
        n_pos=n_pos,
        n_logits=n_logits,
    )
    return selected, dense


def run_attribution(
    model: HookedTransformer,
    transcoders: dict[int, SingleLayerTranscoder],
    prompt: torch.Tensor | Sequence[int] | str,
    *,
    batch_size: int,
    max_feature_nodes: int,
    update_interval: int = 1,
    logit_targets: Sequence[TargetSpec | object],
    layers: Sequence[int] | None = None,
    zero_positions: slice | None = slice(0, 1),
) -> tuple[list[ActiveFeature], torch.Tensor]:
    """Phase 0 + Phase 3 + Phase 4 attribution orchestration.

    ``prompt`` may be a token tensor/list or a string accepted by
    ``HookedTransformer.to_tokens``. The returned dense matrix is already
    packed to the selected node set, ready for GraphLink translation.
    """
    if isinstance(prompt, str):
        input_ids = model.to_tokens(prompt, prepend_bos=False)
    else:
        input_ids = torch.as_tensor(prompt, dtype=torch.long, device=model.W_E.device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    ctx = setup_attribution(
        model,
        input_ids,
        batch_size=batch_size,
        transcoders=transcoders,
        layers=list(layers if layers is not None else sorted(transcoders)),
        zero_positions=zero_positions,
    )
    return run_attribution_from_context(
        ctx,
        logit_targets=logit_targets,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
    )
