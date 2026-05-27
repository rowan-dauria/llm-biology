"""Chunk 6 acceptance: batched backward equals serial backward.

The ``AttributionContext.compute_batch`` mechanic injects per-slot encoder
vectors (or demeaned-unembed seeds for logit slots) into one backward over a
``batch_size``-expanded forward. The result must equal — within fp32 tolerance
— the rows produced by calling :func:`attribute_feature_row` /
:func:`attribute_logit_row` one target at a time on a non-expanded forward.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer, HookedTransformerConfig

from biology_server_t_lens.tl_attribution import (
    TargetSpec,
    attribute_feature_row,
    attribute_logit_row,
    setup_attribution,
)
from biology_server_t_lens.tl_forward import (
    HookState,
    finalize_active_features,
    install_transcoder_hooks,
)
from biology_server_t_lens.tl_freeze import install_freezes


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


def _serial_forward(
    model: HookedTransformer,
    transcoders: dict[int, SingleLayerTranscoder],
    tokens: torch.Tensor,
) -> HookState:
    state = HookState(
        layers=list(range(model.cfg.n_layers)),
        transcoders=transcoders,
    )
    fwd_hooks = install_transcoder_hooks(model, transcoders, state)
    model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    finalize_active_features(state)
    return state


class TestBatchedAttribution(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        install_freezes(self.model)
        self.transcoders = {
            layer: _toy_transcoder(self.model.cfg.d_model, 16, layer, seed=400 + layer)
            for layer in range(self.model.cfg.n_layers)
        }
        torch.manual_seed(9)
        self.tokens = torch.randint(0, self.model.cfg.d_vocab, (1, self.model.cfg.n_ctx))

    def _pick_targets(self, state: HookState) -> list[TargetSpec]:
        # Mix logit + feature targets, varied positions and (for features)
        # varied target layers — exercises both injection sites in one batch.
        targets: list[TargetSpec] = [
            TargetSpec(kind="logit", pos=self.model.cfg.n_ctx - 1, token_id=3),
            TargetSpec(kind="logit", pos=self.model.cfg.n_ctx - 1, token_id=11),
        ]
        # Pick one feature target at layer 1 (so it has upstream sources).
        layer1 = state.layer_features[1]
        if layer1.feature_ids.numel() > 0:
            local = 0
            targets.append(
                TargetSpec(
                    kind="feature",
                    layer=1,
                    pos=int(layer1.positions[local].item()),
                    encoder_vector=layer1.encoder_vectors[local].clone(),
                )
            )
        return targets

    def test_batched_matches_serial(self):
        # Serial reference: build a non-expanded state, then run each target
        # one at a time.
        serial_state = _serial_forward(self.model, self.transcoders, self.tokens)
        targets = self._pick_targets(serial_state)

        # Sanity: at least one feature target made it in.
        self.assertTrue(
            any(t.kind == "feature" for t in targets),
            "expected at least one feature target in the test set",
        )

        n_features = sum(
            serial_state.layer_features[layer].feature_ids.numel() for layer in serial_state.layers
        )
        n_pos = self.model.cfg.n_ctx
        serial_feature = torch.zeros(len(targets), n_features)
        serial_embed = torch.zeros(len(targets), n_pos)
        for row, target in enumerate(targets):
            if target.kind == "logit":
                assert target.token_id is not None
                fs, es = attribute_logit_row(
                    self.model, serial_state, token_id=target.token_id, pos=target.pos
                )
            else:
                assert target.layer is not None and target.encoder_vector is not None
                fs, es = attribute_feature_row(
                    self.model,
                    serial_state,
                    target_layer=target.layer,
                    target_pos=target.pos,
                    encoder_vector=target.encoder_vector,
                )
            serial_feature[row] = fs.detach()
            serial_embed[row] = es.detach()

        # Batched: setup_attribution expands tokens to (batch_size, n_pos)
        # and the same compute_batch produces a row per slot.
        ctx = setup_attribution(
            self.model,
            self.tokens,
            batch_size=max(4, len(targets)),
            transcoders=self.transcoders,
            layers=list(range(self.model.cfg.n_layers)),
        )
        # Re-derive feature target encoders from the batched state (same
        # transcoder weights, same input, so active features are identical).
        batched_targets: list[TargetSpec] = []
        for target in targets:
            if target.kind == "feature":
                assert target.layer is not None
                data = ctx.state.layer_features[target.layer]
                mask = data.positions == target.pos
                local_idx = int(mask.nonzero(as_tuple=False).flatten()[0].item())
                batched_targets.append(
                    TargetSpec(
                        kind="feature",
                        layer=target.layer,
                        pos=target.pos,
                        encoder_vector=data.encoder_vectors[local_idx].clone(),
                    )
                )
            else:
                batched_targets.append(target)

        batched_feature, batched_embed = ctx.compute_batch(batched_targets)

        # Active-feature ordering must match between serial and batched
        # (same transcoder weights + same input).
        self.assertEqual(n_features, ctx.n_features)

        self.assertTrue(
            torch.allclose(
                batched_feature[: len(targets)],
                serial_feature,
                atol=1e-4,
                rtol=1e-4,
            ),
            f"batched feature scores differ from serial: "
            f"max |Δ| = {(batched_feature[: len(targets)] - serial_feature).abs().max().item():.3e}",
        )
        self.assertTrue(
            torch.allclose(
                batched_embed[: len(targets)],
                serial_embed,
                atol=1e-4,
                rtol=1e-4,
            ),
            f"batched embedding scores differ from serial: "
            f"max |Δ| = {(batched_embed[: len(targets)] - serial_embed).abs().max().item():.3e}",
        )

    def test_batched_no_oom_at_size_64(self):
        # Memory smoke: just verify the batched forward + a backward run
        # without OOM on the toy. Uses only logit targets so the test is
        # cheap to set up.
        batch_size = 64
        ctx = setup_attribution(
            self.model,
            self.tokens,
            batch_size=batch_size,
            transcoders=self.transcoders,
            layers=list(range(self.model.cfg.n_layers)),
        )
        targets = [
            TargetSpec(
                kind="logit", pos=self.model.cfg.n_ctx - 1, token_id=k % self.model.cfg.d_vocab
            )
            for k in range(batch_size)
        ]
        feature, embed = ctx.compute_batch(targets, retain_graph=False)
        self.assertEqual(feature.shape[0], batch_size)
        self.assertEqual(embed.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
