"""Attribution rows on a linearised TransformerLens model.

Three layers of API, all consuming a populated :class:`HookState`:

- :func:`attribute_logit_row` — one backward from ``state.final_hidden`` with a
  demeaned-unembed gradient at a single ``(0, pos, :)`` slot. Source attribution
  for a single logit target. (Chunk 4.)
- :func:`attribute_feature_row` — one backward from ``state.mlp_inputs[layer]``
  with an encoder vector at a single ``(0, pos, :)`` slot. Source attribution
  for a single feature target. (Chunk 5.) Intentionally slow; correctness
  oracle for the batched version.
- :class:`AttributionContext` / :func:`setup_attribution` — batched backward
  via ``input_ids.expand(batch_size, -1)`` plus per-slot ``register_hook``
  injection. ``batch_size`` rows in one ``.backward()``. (Chunk 6.)

All three return ``(feature_scores, embedding_scores)`` over the active source
nodes captured by the forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer

from biology_server_t_lens.tl_forward import (
    HookState,
    finalize_active_features,
    install_transcoder_hooks,
)

__all__ = [
    "AttributionContext",
    "TargetSpec",
    "attribute_feature_row",
    "attribute_logit_row",
    "build_dense_edge_matrix_serial",
    "collect_embedding_scores",
    "collect_feature_scores",
    "demeaned_unembed_vector",
    "setup_attribution",
    "total_active_features",
]


def total_active_features(state: HookState) -> int:
    return sum(state.layer_features[layer].feature_ids.numel() for layer in state.layers)


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
) -> torch.Tensor:
    """Contract each tracked layer's MLP-out grad with its active decoder vecs.

    ``decoder_vectors`` are already activation-scaled (see ``layer_feature_data``),
    so the result is the eqn-7 / eqn-8 source score for each active feature.
    Returns zeros for layers that didn't receive a backward (legitimate when
    the target is *upstream* of that layer).
    """
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
        grad_at_positions = grad[batch_index].index_select(0, data.positions.to(grad.device))
        layer_scores = (
            grad_at_positions.to(data.decoder_vectors.dtype) * data.decoder_vectors
        ).sum(dim=-1)
        scores[data.start : data.end] = layer_scores

    if scores is None:
        device = state.embedding.device if state.embedding is not None else torch.device("cpu")
        return torch.zeros(n_features, device=device)
    return scores


def collect_embedding_scores(state: HookState, *, batch_index: int = 0) -> torch.Tensor:
    if state.embedding_grad is None or state.token_vectors is None:
        raise RuntimeError("Embedding gradients were not captured")
    return (
        state.embedding_grad[batch_index].to(state.token_vectors.dtype) * state.token_vectors
    ).sum(dim=-1)


def attribute_logit_row(
    model: HookedTransformer,
    state: HookState,
    *,
    token_id: int,
    pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-target logit attribution. One backward from ``state.final_hidden``."""
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
    embedding_scores = collect_embedding_scores(state)
    return feature_scores, embedding_scores


def attribute_feature_row(
    model: HookedTransformer,  # noqa: ARG001 - kept for symmetry with logit row
    state: HookState,
    *,
    target_layer: int,
    target_pos: int,
    encoder_vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    embedding_scores = collect_embedding_scores(state)
    return feature_scores, embedding_scores


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


def build_dense_edge_matrix_serial(
    model: HookedTransformer,
    state: HookState,
    targets: Sequence[TargetSpec],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loop ``attribute_*_row`` once per target. Slow; used as oracle."""
    n_features = total_active_features(state)
    n_pos = (
        state.token_vectors.shape[0]
        if state.token_vectors is not None
        else state.final_hidden.shape[1]  # type: ignore[union-attr]
    )
    feature_matrix = torch.zeros(len(targets), n_features)
    embedding_matrix = torch.zeros(len(targets), n_pos)

    for row_idx, target in enumerate(targets):
        if target.kind == "logit":
            assert target.token_id is not None
            fs, es = attribute_logit_row(model, state, token_id=target.token_id, pos=target.pos)
        elif target.kind == "feature":
            assert target.layer is not None and target.encoder_vector is not None
            fs, es = attribute_feature_row(
                model,
                state,
                target_layer=target.layer,
                target_pos=target.pos,
                encoder_vector=target.encoder_vector,
            )
        else:
            raise ValueError(f"Unknown target kind {target.kind!r}")
        feature_matrix[row_idx] = fs.detach().to(feature_matrix.dtype)
        embedding_matrix[row_idx] = es.detach().to(embedding_matrix.dtype)

    return feature_matrix, embedding_matrix


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
    ) -> None:
        if state.final_hidden is None:
            raise RuntimeError("setup_attribution must be called before compute_batch")
        self.model = model
        self.state = state
        self.batch_size = batch_size
        self.n_features = total_active_features(state)
        self.n_pos = state.token_vectors.shape[0] if state.token_vectors is not None else 0

    def compute_batch(
        self,
        targets: Sequence[TargetSpec],
        *,
        retain_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attribute up to ``batch_size`` targets in one backward.

        Single-backward mechanic (matches circuit-tracer's
        ``AttributionContext.compute_batch``):

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
        embedding_matrix = torch.zeros(len(targets), self.n_pos, device=fh.device, dtype=fh.dtype)
        for k, target in enumerate(targets):
            if target.kind == "feature":
                assert target.layer is not None
                upstream = [layer_idx for layer_idx in state.layers if layer_idx < target.layer]
            else:
                upstream = list(state.layers)
            fs = collect_feature_scores(state, self.n_features, layers=upstream, batch_index=k)
            es = collect_embedding_scores(state, batch_index=k)
            feature_matrix[k] = fs.to(feature_matrix.dtype)
            embedding_matrix[k] = es.to(embedding_matrix.dtype)
        return feature_matrix, embedding_matrix


def setup_attribution(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    *,
    batch_size: int,
    transcoders: dict[int, SingleLayerTranscoder],
    layers: Sequence[int],
) -> AttributionContext:
    """Run one forward over an ``(batch_size, n_pos)``-expanded prompt.

    ``input_ids`` may be ``(n_pos,)`` or ``(1, n_pos)`` — both are expanded.
    Returns a populated :class:`AttributionContext`. The forward pass holds
    onto the autograd graph (``retain_graph`` is the caller's job via
    :meth:`AttributionContext.compute_batch`).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError(f"Expected (1, n_pos) or (n_pos,), got {tuple(input_ids.shape)}")
    expanded = input_ids.expand(batch_size, -1).contiguous()

    state = HookState(layers=list(layers), transcoders=transcoders)
    fwd_hooks = install_transcoder_hooks(model, transcoders, state)
    # We use model.hooks(...) as a context — but we need the graph to live past
    # the context exit. Use run_with_hooks (which removes hooks after forward
    # but keeps the autograd graph alive on the cached tensors).
    model.run_with_hooks(expanded, fwd_hooks=fwd_hooks)
    finalize_active_features(state)
    return AttributionContext(model, state, batch_size)
