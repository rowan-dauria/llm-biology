"""Causal interventions on transcoder features (Methods §"Graph Interventions").

Implements the *primary* intervention method from the Circuit Tracing methods
paper: steering / clamping a transcoder feature on the **local replacement
model** and reading off the downstream effect on other features and on the
logits.

The local replacement model for a prompt ``p`` (methods paper) is the underlying
model with three substitutions, all sourced from a single clean forward on ``p``:

1. each tracked MLP is replaced by its transcoder reconstruction
   ``decode(encode(mlp_in))``;
2. attention patterns and normalization denominators (RMS/LayerNorm scales) are
   *frozen* to their clean values;
3. an **error term** is added to each tracked MLP output equal to
   ``true_mlp_out - reconstruction`` on the clean run.

With no intervention this reproduces the underlying model's activations and
logits exactly. The freezes (2) make the residual stream a *linear* function of
the transcoder feature activations — the only remaining non-linearity is each
transcoder's encoder activation (ReLU), recomputed at every layer. This is
exactly the model whose linearisation the attribution graph describes, so
perturbing a feature here and watching the effect propagate is a direct test of
the graph's edges.

Cross-layer transcoders (CLTs) need "constrained patching" because one feature
decodes into a *range* of layers; this project uses single-layer transcoders
(each feature decodes to its own layer only), so constrained patching collapses
to direct feature steering — the method implemented here.

Untracked layers (no transcoder) have no graph representation: the attribution
backward stops the gradient through their MLP output, treating it as a constant
background. To keep the intervention forward consistent with the graph, their
MLP outputs are frozen to clean values too (``freeze_untracked_mlp=True``).

Public surface:

- :class:`FeatureIntervention` — one clamp/steer on a ``(layer, pos, feature)``.
- :class:`InterventionResult` — clean vs intervened logits and measured feature
  activations, with convenience accessors.
- :func:`run_feature_intervention` — run the two-pass intervention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from biology_server_t_lens.tl_forward import ensure_replacement_mlp_hooks

__all__ = [
    "FeatureIntervention",
    "InterventionResult",
    "run_feature_intervention",
]

# A measured feature is keyed by (layer, position, feature_id).
FeatureKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class FeatureIntervention:
    """One steer/clamp applied to a transcoder feature.

    ``layer`` must be a tracked layer (have a transcoder). The new activation at
    ``(layer, pos, feature)`` is chosen as:

    - ``value`` if given — clamp to an absolute activation;
    - else ``factor * clean_activation`` if ``factor`` is given — multiplicative
      steering against the feature's *clean* activation (``factor=-1`` negates,
      ``factor=4`` amplifies 4x, ``factor=0`` ablates);
    - else ``0.0`` — ablation (the default).

    ``value`` and ``factor`` are mutually exclusive.
    """

    layer: int
    pos: int
    feature: int
    value: float | None = None
    factor: float | None = None

    def __post_init__(self) -> None:
        if self.value is not None and self.factor is not None:
            raise ValueError("FeatureIntervention takes value XOR factor, not both")

    def resolve(self, clean_activation: float) -> float:
        if self.value is not None:
            return float(self.value)
        if self.factor is not None:
            return float(self.factor) * clean_activation
        return 0.0


@dataclass(slots=True)
class InterventionResult:
    """Outcome of :func:`run_feature_intervention`.

    Logits are ``(n_pos, d_vocab)`` for the single prompt (batch axis squeezed).
    The activation dicts cover every ``(layer, pos, feature)`` that was either
    intervened on or listed in ``measure_features``.
    """

    clean_logits: torch.Tensor
    intervened_logits: torch.Tensor
    clean_feature_acts: dict[FeatureKey, float] = field(default_factory=dict)
    intervened_feature_acts: dict[FeatureKey, float] = field(default_factory=dict)

    def logit_diff(self, pos: int = -1) -> torch.Tensor:
        """``intervened - clean`` logits at ``pos`` (default last)."""
        return self.intervened_logits[pos] - self.clean_logits[pos]

    def top_logit_changes(self, pos: int = -1, k: int = 10) -> list[tuple[int, float]]:
        """``k`` token ids with the largest absolute logit change at ``pos``."""
        diff = self.logit_diff(pos)
        order = torch.argsort(diff.abs(), descending=True)[:k]
        return [(int(t), float(diff[t])) for t in order]

    def feature_fraction(self, key: FeatureKey) -> float:
        """Intervened activation as a fraction of the clean activation.

        Returns ``nan`` when the clean activation is 0.
        """
        clean = self.clean_feature_acts.get(key, 0.0)
        intervened = self.intervened_feature_acts.get(key, 0.0)
        if clean == 0.0:
            return float("nan")
        return intervened / clean


def _block_layers(model: HookedTransformer) -> range:
    return range(model.cfg.n_layers)


def _capture_clean(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    transcoders: dict[int, SingleLayerTranscoder],
    tracked: list[int],
) -> dict:
    """Single clean forward; capture everything the local replacement freezes.

    Captures, all from the underlying model's true forward on the prompt:

    - ``patterns[L]`` — attention pattern at every block;
    - ``ln1[L]`` / ``ln2[L]`` / ``ln_final`` — normalization scales;
    - ``untracked_mlp[L]`` — true MLP output of layers without a transcoder;
    - ``features[L]`` — transcoder feature activations at tracked layers;
    - ``error[L]`` — ``true_mlp_out - decode(features)`` at tracked layers; added
      back verbatim (no position zeroing) so the replacement reproduces the
      underlying model exactly.
    - ``logits`` — clean logits ``(n_pos, d_vocab)``.
    """
    cap: dict = {
        "patterns": {},
        "ln1": {},
        "ln2": {},
        "untracked_mlp": {},
        "features": {},
        "error": {},
        "ln_final": None,
        "logits": None,
    }

    fwd_hooks: list = []

    def _pattern_hook(layer: int):
        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            cap["patterns"][layer] = acts.detach().clone()
            return acts

        return hook

    def _scale_hook(store: dict, key):
        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            store[key] = acts.detach().clone()
            return acts

        return hook

    def _untracked_mlp_hook(layer: int):
        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            cap["untracked_mlp"][layer] = acts.detach().clone()
            return acts

        return hook

    def _tracked_in_hook(layer: int):
        transcoder = transcoders[layer]

        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            with torch.no_grad():
                features = transcoder.encode(acts).detach()
            cap["features"][layer] = features
            cap[f"_mlp_in_{layer}"] = acts.detach()
            return acts

        return hook

    def _tracked_out_hook(layer: int):
        transcoder = transcoders[layer]

        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            features = cap["features"][layer]
            mlp_in = cap[f"_mlp_in_{layer}"]
            with torch.no_grad():
                reconstruction = transcoder.decode(
                    features.to(transcoder.W_dec.dtype),
                    mlp_in if transcoder.W_skip is not None else None,
                ).to(acts.dtype)
                error = acts.detach() - reconstruction
            cap["error"][layer] = error
            return acts

        return hook

    for layer in _block_layers(model):
        fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", _pattern_hook(layer)))
        fwd_hooks.append((f"blocks.{layer}.ln1.hook_scale", _scale_hook(cap["ln1"], layer)))
        fwd_hooks.append((f"blocks.{layer}.ln2.hook_scale", _scale_hook(cap["ln2"], layer)))
        if layer in tracked:
            fwd_hooks.append((f"blocks.{layer}.mlp.hook_in", _tracked_in_hook(layer)))
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _tracked_out_hook(layer)))
        else:
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _untracked_mlp_hook(layer)))

    final_scale = getattr(model.ln_final, "hook_scale", None)
    if isinstance(final_scale, HookPoint):
        fwd_hooks.append(("ln_final.hook_scale", _scale_hook(cap, "ln_final")))

    with torch.no_grad():
        logits = model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)
    cap["logits"] = logits.detach()[0]
    return cap


def _read_feature_acts(
    features_by_layer: dict[int, torch.Tensor],
    keys: Sequence[FeatureKey],
) -> dict[FeatureKey, float]:
    out: dict[FeatureKey, float] = {}
    for layer, pos, feature in keys:
        feats = features_by_layer.get(layer)
        if feats is None:
            continue
        out[(layer, pos, feature)] = float(feats[0, pos, feature].item())
    return out


def run_feature_intervention(
    model: HookedTransformer,
    transcoders: dict[int, SingleLayerTranscoder],
    input_ids: torch.Tensor | Sequence[int] | str,
    interventions: Sequence[FeatureIntervention],
    *,
    layers: Sequence[int] | None = None,
    measure_features: Sequence[FeatureKey] = (),
    freeze_attention: bool = True,
    freeze_layernorm: bool = True,
    freeze_untracked_mlp: bool = True,
) -> InterventionResult:
    """Steer/clamp transcoder features on the local replacement model.

    Runs two forward passes: a clean pass that captures the frozen attention
    patterns, normalization denominators, untracked-MLP outputs and per-layer
    transcoder error terms, then an intervened pass over the *local replacement
    model* with those values frozen and the requested feature clamps applied.

    Returns an :class:`InterventionResult` with clean and intervened logits, and
    the clean/intervened activations of every intervened or measured feature so
    downstream effects can be checked against the attribution graph.

    With ``interventions`` empty the intervened pass reproduces the clean logits
    (up to float error) — a useful sanity check that the freezes line up.
    """
    input_ids_tensor: torch.Tensor
    if isinstance(input_ids, str):
        input_ids_tensor = model.to_tokens(input_ids, prepend_bos=False)
    else:
        input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=model.W_E.device)
    if input_ids_tensor.dim() == 1:
        input_ids_tensor = input_ids_tensor.unsqueeze(0)
    if input_ids_tensor.shape[0] != 1:
        raise ValueError(
            f"Expected a single prompt (1, n_pos), got {tuple(input_ids_tensor.shape)}"
        )

    tracked = sorted(transcoders) if layers is None else sorted(layers)
    for layer in tracked:
        if layer not in transcoders:
            raise KeyError(f"No transcoder provided for tracked layer {layer}")
    tracked_set = set(tracked)
    for iv in interventions:
        if iv.layer not in tracked_set:
            raise ValueError(
                f"intervention layer {iv.layer} is not tracked; tracked layers: {tracked}"
            )

    ensure_replacement_mlp_hooks(model, tracked)
    cap = _capture_clean(model, input_ids_tensor, transcoders, tracked)

    # Group interventions by layer for the in-hook, resolving multiplicative
    # factors against the *clean* activation captured above.
    iv_by_layer: dict[int, list[tuple[int, int, float]]] = {}
    for iv in interventions:
        clean_act = float(cap["features"][iv.layer][0, iv.pos, iv.feature].item())
        iv_by_layer.setdefault(iv.layer, []).append((iv.pos, iv.feature, iv.resolve(clean_act)))

    intervened_features: dict[int, torch.Tensor] = {}
    fwd_hooks: list = []

    def _freeze_value_hook(value: torch.Tensor):
        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            return value.to(device=acts.device, dtype=acts.dtype)

        return hook

    def _iv_in_hook(layer: int):
        transcoder = transcoders[layer]
        clamps = iv_by_layer.get(layer, [])

        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            with torch.no_grad():
                features = transcoder.encode(acts).detach()
                for pos, feature, new_value in clamps:
                    features[:, pos, feature] = new_value
            intervened_features[layer] = features
            cap[f"_iv_mlp_in_{layer}"] = acts.detach()
            return acts

        return hook

    def _iv_out_hook(layer: int):
        transcoder = transcoders[layer]

        def hook(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            features = intervened_features[layer]
            mlp_in = cap[f"_iv_mlp_in_{layer}"]
            with torch.no_grad():
                reconstruction = transcoder.decode(
                    features.to(transcoder.W_dec.dtype),
                    mlp_in if transcoder.W_skip is not None else None,
                ).to(acts.dtype)
            return reconstruction + cap["error"][layer].to(device=acts.device, dtype=acts.dtype)

        return hook

    for layer in _block_layers(model):
        if freeze_attention:
            fwd_hooks.append(
                (f"blocks.{layer}.attn.hook_pattern", _freeze_value_hook(cap["patterns"][layer]))
            )
        if freeze_layernorm:
            fwd_hooks.append(
                (f"blocks.{layer}.ln1.hook_scale", _freeze_value_hook(cap["ln1"][layer]))
            )
            fwd_hooks.append(
                (f"blocks.{layer}.ln2.hook_scale", _freeze_value_hook(cap["ln2"][layer]))
            )
        if layer in tracked_set:
            fwd_hooks.append((f"blocks.{layer}.mlp.hook_in", _iv_in_hook(layer)))
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", _iv_out_hook(layer)))
        elif freeze_untracked_mlp:
            fwd_hooks.append(
                (f"blocks.{layer}.hook_mlp_out", _freeze_value_hook(cap["untracked_mlp"][layer]))
            )

    if freeze_layernorm and cap["ln_final"] is not None:
        fwd_hooks.append(("ln_final.hook_scale", _freeze_value_hook(cap["ln_final"])))

    with torch.no_grad():
        intervened_logits = model.run_with_hooks(input_ids_tensor, fwd_hooks=fwd_hooks)
    intervened_logits = intervened_logits.detach()[0]

    # Read clean/intervened activations for intervened + requested features.
    keys: list[FeatureKey] = [(iv.layer, iv.pos, iv.feature) for iv in interventions]
    keys.extend(measure_features)
    # de-dup preserving order
    seen: set[FeatureKey] = set()
    unique_keys = [k for k in keys if not (k in seen or seen.add(k))]

    return InterventionResult(
        clean_logits=cap["logits"],
        intervened_logits=intervened_logits,
        clean_feature_acts=_read_feature_acts(cap["features"], unique_keys),
        intervened_feature_acts=_read_feature_acts(intervened_features, unique_keys),
    )
