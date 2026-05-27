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

from biology_server.attribution import LayerFeatureData, layer_feature_data

FwdHook = tuple[str, Callable[[torch.Tensor, HookPoint], torch.Tensor]]

__all__ = [
    "HookState",
    "LayerFeatureData",
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


def _make_mlp_in_hook(layer: int, state: HookState):
    transcoder = state.transcoders[layer]

    def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        state.mlp_inputs[layer] = acts
        with torch.no_grad():
            features = transcoder.encode(acts).detach()
            data = layer_feature_data(transcoder, features.squeeze(0))
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
        if transcoder.W_skip is not None:
            skip = transcoder.compute_skip(mlp_input).to(acts.dtype)
            replacement = skip + (reconstruction - skip).detach().requires_grad_(True)
        else:
            replacement = reconstruction.detach().requires_grad_(True)

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
        state.token_vectors = acts.detach().squeeze(0)

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
