"""Forward-pass machinery: transcoder substitution + active-feature cache.

Mirrors the legacy ``attribution.py`` MLP / embedding / final-hidden hooks, but
uses TransformerLens ``HookPoint``\\ s instead of raw module forward hooks. The
freezes (:func:`biology_server_t_lens.tl_freeze.install_freezes`) must already
be installed on the model before these hooks fire.

Public surface:

- :class:`HookState` — per-forward-pass capture buffer.
- :func:`install_transcoder_hooks` — registers the MLP/embedding/final hooks
  that populate the state and expose backward injection sites.
- :func:`layer_feature_data` / :class:`LayerFeatureData` — re-exported from the
  legacy attribution module so downstream code can reuse the same dataclass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from biology_server.attribution import ActiveFeature, LayerFeatureData, layer_feature_data

FwdHook = tuple[str, Callable[[torch.Tensor, HookPoint], torch.Tensor]]

__all__ = [
    "ActiveFeature",
    "FwdHook",
    "HookState",
    "LayerFeatureData",
    "finalize_active_features",
    "install_transcoder_hooks",
    "layer_feature_data",
]


@dataclass(slots=True)
class HookState:
    """Per-forward capture buffer.

    Mirrors the legacy ``biology_server.attribution.HookState`` but keyed off
    TransformerLens hook points. ``mlp_inputs`` / ``feature_values`` /
    ``layer_features`` are populated on the forward pass; ``output_grads`` and
    ``embedding_grad`` are populated by backward hooks during ``.backward()``.
    """

    layers: list[int]
    transcoders: dict[int, SingleLayerTranscoder]
    mlp_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    feature_values: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_features: dict[int, LayerFeatureData] = field(default_factory=dict)
    output_grads: dict[int, torch.Tensor] = field(default_factory=dict)
    embedding: torch.Tensor | None = None
    token_vectors: torch.Tensor | None = None
    embedding_grad: torch.Tensor | None = None
    final_hidden: torch.Tensor | None = None

    def clear(self) -> None:
        self.mlp_inputs.clear()
        self.feature_values.clear()
        self.layer_features.clear()
        self.output_grads.clear()
        self.embedding = None
        self.token_vectors = None
        self.embedding_grad = None
        self.final_hidden = None

    def clear_grads(self) -> None:
        """Clear backward-captured grads, leave forward caches intact.

        Call before each backward when running multiple attribution rows from
        the same forward pass.
        """
        self.output_grads.clear()
        self.embedding_grad = None
        if self.embedding is not None and self.embedding.grad is not None:
            self.embedding.grad = None


def finalize_active_features(state: HookState) -> list[ActiveFeature]:
    """Assign per-layer score offsets and flatten the active-feature list.

    Mirrors ``biology_server.attribution.finalize_active_features``. Mutates
    ``state.layer_features`` in place to set ``start`` on each ``LayerFeatureData``
    so that ``data.start:data.end`` slices into the dense feature-score vector.
    """
    active: list[ActiveFeature] = []
    offset = 0
    for layer in state.layers:
        data = state.layer_features[layer]
        state.layer_features[layer] = LayerFeatureData(
            positions=data.positions,
            feature_ids=data.feature_ids,
            activations=data.activations,
            encoder_vectors=data.encoder_vectors,
            decoder_vectors=data.decoder_vectors,
            start=offset,
        )
        for local_idx in range(data.feature_ids.numel()):
            active.append(
                ActiveFeature(
                    layer=layer,
                    pos=int(data.positions[local_idx].item()),
                    feature=int(data.feature_ids[local_idx].item()),
                    activation=float(data.activations[local_idx].item()),
                    encoder_vector=data.encoder_vectors[local_idx],
                    score_index=offset + local_idx,
                )
            )
        offset += int(data.feature_ids.numel())
    return active


def _make_mlp_in_hook(layer: int, state: HookState):
    transcoder = state.transcoders[layer]

    def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        state.mlp_inputs[layer] = acts
        with torch.no_grad():
            features = transcoder.encode(acts).detach()
            # All batch slots are identical (input was expanded), so the
            # per-slot active-feature set is the same. Take slot 0.
            data = layer_feature_data(transcoder, features[0])
        state.feature_values[layer] = features
        state.layer_features[layer] = data
        return acts

    return hook


def _make_mlp_out_hook(layer: int, state: HookState):
    transcoder = state.transcoders[layer]

    def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        features = state.feature_values[layer]
        mlp_input = state.mlp_inputs[layer]
        with torch.no_grad():
            reconstruction = transcoder.decode(
                features.to(transcoder.W_dec.dtype),
                mlp_input if transcoder.W_skip is not None else None,
            ).to(acts.dtype)
        # Ghost-skip trick (matches circuit-tracer's replacement model):
        # numerically ``replacement == reconstruction`` because ``skip`` is the
        # transcoder skip (or zero, scaled by mlp_input * 0) and the residual
        # ``(reconstruction - skip)`` is detached. But the ``skip`` term has
        # ``grad_fn`` pointing back to ``mlp_input``, so backward from
        # ``final_hidden`` reaches ``mlp_input`` (with grad 0 from the *0
        # multiplier) — letting a caller-installed ``register_hook`` on
        # ``mlp_input`` override that gradient at chosen ``(slot, pos)`` slots
        # for batched feature-target injection (see :class:`AttributionContext`).
        if transcoder.W_skip is not None:
            skip = transcoder.compute_skip(mlp_input).to(acts.dtype)
        else:
            skip = mlp_input * 0
        replacement = skip + (reconstruction - skip).detach()

        def grab(grad: torch.Tensor) -> None:
            state.output_grads[layer] = grad.detach()

        replacement.register_hook(grab)
        return replacement

    return hook


def _make_embed_hook(state: HookState):
    def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        # `install_freezes` already set requires_grad on acts via its permanent
        # hook; make sure of it in case freezes weren't installed.
        acts.requires_grad_(True)
        state.embedding = acts
        # All batch slots see the same input (we expand), so a single
        # ``(n_pos, d_model)`` copy is the right "per-prompt" token vector.
        state.token_vectors = acts[0].detach()

        def grab(grad: torch.Tensor) -> None:
            state.embedding_grad = grad.detach()

        acts.register_hook(grab)
        return acts

    return hook


def _make_final_hidden_hook(state: HookState):
    def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        state.final_hidden = acts
        return acts

    return hook


def install_transcoder_hooks(
    model: HookedTransformer,
    transcoders: dict[int, SingleLayerTranscoder],
    state: HookState,
) -> list[FwdHook]:
    """Install per-forward hooks for transcoder substitution + caching.

    Returns the ``fwd_hooks`` list intended to be passed to ``model.hooks(...)``
    or ``model.run_with_hooks(...)``. Caller owns the lifetime — the hooks are
    *not* installed permanently here. (Permanent hooks live on the model after
    :func:`install_freezes`; these per-forward hooks should be wrapped by a
    ``model.hooks(...)`` context manager so they auto-clean between requests.)
    """
    fwd_hooks: list[FwdHook] = []
    for layer in state.layers:
        if layer not in transcoders:
            raise KeyError(f"No transcoder provided for tracked layer {layer}")
        # MLP input = LN-normalised resid_mid (what the MLP weights actually
        # receive). This is also what the transcoders were trained on.
        fwd_hooks.append((f"blocks.{layer}.ln2.hook_normalized", _make_mlp_in_hook(layer, state)))
        fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _make_mlp_out_hook(layer, state)))

    fwd_hooks.append(("hook_embed", _make_embed_hook(state)))
    fwd_hooks.append(("ln_final.hook_normalized", _make_final_hidden_hook(state)))
    return fwd_hooks
