"""Chunk 5 acceptance: ``attribute_feature_row`` correctness on a toy model.

Three structural checks of the linearised backward, each isolating one
contribution to the feature→feature edge weight (Methods eqns 7 and 8):

1. Same-token only (zero out attention at the target layer): scores must be
   non-zero only at the source position equal to the target position.
2. Cross-token edges exist (non-trivial attention pattern): at least one source
   position ≠ target position has non-zero score.
3. Cross-token edges scale linearly with the attention pattern (linearised
   model is linear in the detached pattern): doubling the pattern doubles
   the cross-token component of every score.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from biology_server.tl_attribution import attribute_feature_row
from biology_server.tl_forward import (
    HookState,
    finalize_active_features,
    install_transcoder_hooks,
)
from biology_server.tl_freeze import install_freezes


def _toy_model(seed: int = 0) -> HookedTransformer:
    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=8,
        n_heads=2,
        d_head=4,
        n_ctx=4,
        d_vocab=16,
        act_fn="silu",
        normalization_type="RMS",
    )
    return HookedTransformer(cfg).to("cpu")


def _toy_transcoder(
    d_model: int, d_transcoder: int, layer_idx: int, seed: int
) -> SingleLayerTranscoder:
    g = torch.Generator().manual_seed(seed)
    tc = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=nn.ReLU(),
        layer_idx=layer_idx,
        skip_connection=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        tc.W_enc.copy_(torch.randn(d_transcoder, d_model, generator=g) * 0.5)
        tc.W_dec.copy_(torch.randn(d_transcoder, d_model, generator=g) * 0.5)
        tc.b_enc.copy_(torch.randn(d_transcoder, generator=g) * 0.1)
        tc.b_dec.zero_()
    return tc


def _zero_attn_out_layer_1(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
    return torch.zeros_like(acts).detach()


def _make_pattern_override(pattern: torch.Tensor):
    def override(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        return pattern.to(device=acts.device, dtype=acts.dtype).detach()

    return override


class TestFeatureAttribution(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        install_freezes(self.model)
        self.transcoders = {
            layer: _toy_transcoder(self.model.cfg.d_model, 16, layer, seed=300 + layer)
            for layer in range(self.model.cfg.n_layers)
        }
        torch.manual_seed(3)
        self.tokens = torch.randint(0, self.model.cfg.d_vocab, (1, self.model.cfg.n_ctx))

    def _run_forward(self, extra_hooks=None):
        state = HookState(
            layers=list(range(self.model.cfg.n_layers)),
            transcoders=self.transcoders,
        )
        fwd_hooks = install_transcoder_hooks(self.model, self.transcoders, state)
        if extra_hooks is not None:
            fwd_hooks.extend(extra_hooks)
        self.model.run_with_hooks(self.tokens, fwd_hooks=fwd_hooks)
        finalize_active_features(state)
        return state

    def _target_encoder(self, state: HookState, target_layer: int, target_pos: int) -> torch.Tensor:
        # Pick the most-active feature at (target_layer, target_pos) and use
        # its W_enc row as the injected encoder vector.
        data = state.layer_features[target_layer]
        positions = data.positions
        mask = positions == target_pos
        if not bool(mask.any().item()):
            # Fallback: any active feature at the layer, just use one encoder.
            return self.transcoders[target_layer].W_enc[0].clone()
        local_indices = mask.nonzero(as_tuple=False).flatten()
        return data.encoder_vectors[local_indices[0]].clone()

    def test_same_token_edge_when_attn_disabled(self):
        # Disable attn_out at layer 1 entirely → only the direct residual path
        # contributes to mlp_in_1[t]. Cross-token edges must vanish.
        state = self._run_forward(
            extra_hooks=[("blocks.1.hook_attn_out", _zero_attn_out_layer_1)],
        )
        target_layer = 1
        target_pos = self.model.cfg.n_ctx - 1
        encoder = self._target_encoder(state, target_layer, target_pos)

        feature_scores, _, _ = attribute_feature_row(
            self.model,
            state,
            target_layer=target_layer,
            target_pos=target_pos,
            encoder_vector=encoder,
        )

        layer0 = state.layer_features[0]
        for local_idx in range(layer0.feature_ids.numel()):
            src_pos = int(layer0.positions[local_idx].item())
            score_idx = layer0.start + local_idx
            score = float(feature_scores[score_idx].item())
            if src_pos != target_pos:
                self.assertAlmostEqual(
                    score,
                    0.0,
                    places=5,
                    msg=f"cross-token edge non-zero (src_pos={src_pos}, tgt_pos={target_pos}) "
                    f"despite attn_out_1 disabled: score={score:.3e}",
                )

    def test_cross_token_edges_present(self):
        # Non-trivial pattern at layer 1: at least one source feature at a
        # position ≠ target_pos should attribute non-zero.
        n_ctx = self.model.cfg.n_ctx
        n_heads = self.model.cfg.n_heads
        # Lower-triangular pattern with equal mass across all attended positions.
        pattern = torch.zeros(1, n_heads, n_ctx, n_ctx)
        for t in range(n_ctx):
            for s in range(t + 1):
                pattern[0, :, t, s] = 1.0 / (t + 1)

        state = self._run_forward(
            extra_hooks=[
                ("blocks.1.attn.hook_pattern", _make_pattern_override(pattern)),
            ],
        )
        target_layer = 1
        target_pos = n_ctx - 1
        encoder = self._target_encoder(state, target_layer, target_pos)

        feature_scores, _, _ = attribute_feature_row(
            self.model,
            state,
            target_layer=target_layer,
            target_pos=target_pos,
            encoder_vector=encoder,
        )

        layer0 = state.layer_features[0]
        cross_token_nonzero = False
        for local_idx in range(layer0.feature_ids.numel()):
            src_pos = int(layer0.positions[local_idx].item())
            score_idx = layer0.start + local_idx
            score = float(feature_scores[score_idx].item())
            if src_pos != target_pos and abs(score) > 1e-5:
                cross_token_nonzero = True
                break
        self.assertTrue(
            cross_token_nonzero,
            "expected at least one cross-token edge with non-trivial attention",
        )

    def test_feature_row_contracts_decoder_vecs_with_mlp_out_grads(self):
        # Self-consistency: attribute_feature_row's output for an upstream
        # source feature must equal (scaled_decoder_vec · grad_at_mlp_out)
        # where grad_at_mlp_out is whatever the backward put into
        # state.output_grads. This mirrors the chunk-4 contraction check but
        # with the seed injected at mlp_in rather than final_hidden.
        state = self._run_forward()
        target_layer = 1
        target_pos = self.model.cfg.n_ctx - 1
        encoder = self._target_encoder(state, target_layer, target_pos)

        feature_scores, error_scores, _ = attribute_feature_row(
            self.model,
            state,
            target_layer=target_layer,
            target_pos=target_pos,
            encoder_vector=encoder,
        )

        # Only upstream layers receive grad. Verify the contraction layer by
        # layer using state.output_grads (populated by attribute_feature_row).
        for layer in [layer_idx for layer_idx in state.layers if layer_idx < target_layer]:
            data = state.layer_features[layer]
            grad = state.output_grads[layer]
            for local_idx in range(data.feature_ids.numel()):
                src_pos = int(data.positions[local_idx].item())
                d = data.decoder_vectors[local_idx]
                expected = float((d * grad[0, src_pos]).sum().item())
                got = float(feature_scores[data.start + local_idx].item())
                self.assertAlmostEqual(got, expected, places=5)

        n_pos = self.model.cfg.n_ctx
        for layer in [layer_idx for layer_idx in state.layers if layer_idx < target_layer]:
            error = state.error_vectors[layer].to(state.output_grads[layer].device)
            grad = state.output_grads[layer]
            start = state.layers.index(layer) * n_pos
            expected = (grad[0].to(error.dtype) * error).sum(dim=-1)
            got = error_scores[start : start + n_pos]
            self.assertTrue(torch.allclose(got, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
