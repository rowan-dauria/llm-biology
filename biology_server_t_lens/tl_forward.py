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
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from biology_server.attribution import ActiveFeature, LayerFeatureData, layer_feature_data

FwdHook = tuple[str, Callable[[torch.Tensor, HookPoint], torch.Tensor]]
DEFAULT_ZERO_POSITIONS = slice(0, 1)

__all__ = [
    "ActiveFeature",
    "DEFAULT_ZERO_POSITIONS",
    "FwdHook",
    "HookState",
    "LayerFeatureData",
    "ReplacementMLP",
    "ensure_replacement_mlp_hooks",
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
    zero_positions: slice | None = DEFAULT_ZERO_POSITIONS
    mlp_inputs: dict[int, torch.Tensor] = field(default_factory=dict)
    feature_values: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_features: dict[int, LayerFeatureData] = field(default_factory=dict)
    original_mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    reconstructions: dict[int, torch.Tensor] = field(default_factory=dict)
    error_vectors: dict[int, torch.Tensor] = field(default_factory=dict)
    output_grads: dict[int, torch.Tensor] = field(default_factory=dict)
    embedding: torch.Tensor | None = None
    token_vectors: torch.Tensor | None = None
    embedding_grad: torch.Tensor | None = None
    final_hidden: torch.Tensor | None = None

    def clear(self) -> None:
        self.mlp_inputs.clear()
        self.feature_values.clear()
        self.layer_features.clear()
        self.original_mlp_outputs.clear()
        self.reconstructions.clear()
        self.error_vectors.clear()
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


class ReplacementMLP(nn.Module):
    """Wrap a TL MLP with circuit-tracer-style input/output hook points."""

    def __init__(self, old_mlp: nn.Module) -> None:
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


def _has_replacement_hooks(mlp: nn.Module) -> bool:
    return isinstance(mlp, ReplacementMLP) or (
        isinstance(getattr(mlp, "hook_in", None), HookPoint)
        and isinstance(getattr(mlp, "hook_out", None), HookPoint)
    )


def ensure_replacement_mlp_hooks(model: HookedTransformer, layers: list[int]) -> None:
    """Expose post-LN MLP input/output hook points for tracked layers.

    TransformerLens' built-in ``blocks.{l}.hook_mlp_in`` sits before ``ln2``.
    Circuit-tracer's ``ReplacementMLP.hook_in`` sits on the tensor actually fed
    to the MLP, after ``ln2`` and its learned weight.
    """

    wrapped_any = False
    for layer in layers:
        block = model.blocks[layer]
        if _has_replacement_hooks(block.mlp):
            continue
        block.mlp = ReplacementMLP(block.mlp)
        wrapped_any = True
    if wrapped_any:
        model.setup()


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


def _zero_positions_(acts: torch.Tensor, zero_positions: slice | None) -> torch.Tensor:
    if zero_positions is not None:
        if acts.dim() == 3:
            acts[:, zero_positions, :] = 0
        elif acts.dim() == 2:
            acts[zero_positions, :] = 0
        else:
            raise ValueError(f"expected rank-2 or rank-3 tensor, got {acts.dim()}")
    return acts


def _make_mlp_in_hook(layer: int, state: HookState):
    transcoder = state.transcoders[layer]

    def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        state.mlp_inputs[layer] = acts
        with torch.no_grad():
            features = transcoder.encode(acts).detach()
            _zero_positions_(features, state.zero_positions)
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
            original = acts[0].detach()
            prompt_reconstruction = reconstruction[0].detach()
            error = original - prompt_reconstruction
            _zero_positions_(error, state.zero_positions)
            state.original_mlp_outputs[layer] = original
            state.reconstructions[layer] = prompt_reconstruction
            state.error_vectors[layer] = error
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
    ensure_replacement_mlp_hooks(model, state.layers)

    fwd_hooks: list[FwdHook] = []
    for layer in state.layers:
        if layer not in transcoders:
            raise KeyError(f"No transcoder provided for tracked layer {layer}")
        # MLP input = the post-ln2 tensor actually fed to the MLP. This matches
        # circuit-tracer's ReplacementMLP.hook_in and the Qwen3 transcoder space.
        fwd_hooks.append((f"blocks.{layer}.mlp.hook_in", _make_mlp_in_hook(layer, state)))
        # Keep the replacement at the block output hook so it composes with the
        # existing permanent MLP-output freeze installed in ``tl_freeze``.
        fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _make_mlp_out_hook(layer, state)))

    fwd_hooks.append(("hook_embed", _make_embed_hook(state)))
    # In a full serial forward, this is the post-final-norm residual that the
    # unembed reads. Batched setup stops before unembed and sets this manually.
    fwd_hooks.append(("unembed.hook_in", _make_final_hidden_hook(state)))
    return fwd_hooks
