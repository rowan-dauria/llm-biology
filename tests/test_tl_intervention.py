"""Acceptance tests for feature interventions on the local replacement model."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer, HookedTransformerConfig

from biology_server_t_lens.tl_freeze import install_freezes
from biology_server_t_lens.tl_intervention import (
    FeatureIntervention,
    run_feature_intervention,
)


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


class TestTLIntervention(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        install_freezes(self.model)
        self.d_transcoder = 16
        self.layers = list(range(self.model.cfg.n_layers))
        self.transcoders = {
            layer: _toy_transcoder(
                d_model=self.model.cfg.d_model,
                d_transcoder=self.d_transcoder,
                layer_idx=layer,
                seed=100 + layer,
            )
            for layer in self.layers
        }
        torch.manual_seed(42)
        self.tokens = torch.randint(0, self.model.cfg.d_vocab, (1, self.model.cfg.n_ctx))
        # Probe every feature so we can find active ones.
        self.all_keys = [
            (layer, pos, feat)
            for layer in self.layers
            for pos in range(self.model.cfg.n_ctx)
            for feat in range(self.d_transcoder)
        ]

    def _baseline(self):
        return run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[],
            layers=self.layers,
            measure_features=self.all_keys,
        )

    def test_identity_no_intervention(self):
        """No interventions: the local replacement reproduces clean logits."""
        res = self._baseline()
        torch.testing.assert_close(res.intervened_logits, res.clean_logits, atol=1e-4, rtol=1e-4)

    def test_ablation_zeroes_target_and_changes_logits(self):
        base = self._baseline()
        # Pick an active feature on the last position of layer 0 (so it has
        # downstream layers + the logits to affect).
        target = next(
            key for key, act in base.clean_feature_acts.items() if key[0] == 0 and act > 1e-3
        )
        layer, pos, feat = target

        res = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[FeatureIntervention(layer=layer, pos=pos, feature=feat)],
            layers=self.layers,
        )
        # Target feature was clamped to zero.
        self.assertAlmostEqual(res.intervened_feature_acts[target], 0.0, places=5)
        self.assertGreater(res.clean_feature_acts[target], 1e-3)
        # Ablation perturbs the logits.
        self.assertGreater(res.logit_diff().abs().max().item(), 1e-5)

    def test_factor_negates_against_clean(self):
        base = self._baseline()
        target = next(key for key, act in base.clean_feature_acts.items() if act > 1e-3)
        layer, pos, feat = target
        clean = base.clean_feature_acts[target]

        res = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[FeatureIntervention(layer=layer, pos=pos, feature=feat, factor=-1.0)],
            layers=self.layers,
        )
        self.assertAlmostEqual(res.intervened_feature_acts[target], -clean, places=5)
        self.assertAlmostEqual(res.feature_fraction(target), -1.0, places=5)

    def test_value_clamp(self):
        base = self._baseline()
        target = next(iter(base.clean_feature_acts))
        layer, pos, feat = target
        res = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[FeatureIntervention(layer=layer, pos=pos, feature=feat, value=3.5)],
            layers=self.layers,
        )
        self.assertAlmostEqual(res.intervened_feature_acts[target], 3.5, places=5)

    def test_upstream_intervention_changes_downstream_feature(self):
        """Steering a layer-0 feature should move some layer-1 feature."""
        base = self._baseline()
        # Strongly amplify an active layer-0 feature.
        l0_target = next(
            key for key, act in base.clean_feature_acts.items() if key[0] == 0 and act > 1e-3
        )
        layer, pos, feat = l0_target
        res = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[FeatureIntervention(layer=layer, pos=pos, feature=feat, value=50.0)],
            layers=self.layers,
            measure_features=[k for k in self.all_keys if k[0] == 1 and k[1] == pos],
        )
        moved = max(
            abs(res.intervened_feature_acts[k] - res.clean_feature_acts[k])
            for k in res.clean_feature_acts
            if k[0] == 1 and k[1] == pos
        )
        self.assertGreater(moved, 1e-4)

    def test_residual_write_zero_is_identity(self):
        """A zero residual write leaves the local replacement at clean logits."""
        d_model = self.model.cfg.d_model
        res = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[],
            layers=self.layers,
            residual_writes={(1, 0): torch.zeros(d_model)},
        )
        torch.testing.assert_close(res.intervened_logits, res.clean_logits, atol=1e-4, rtol=1e-4)

    def test_residual_write_changes_logits(self):
        """A non-zero residual write perturbs the logits."""
        d_model = self.model.cfg.d_model
        torch.manual_seed(7)
        vec = torch.randn(d_model)
        res = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[],
            layers=self.layers,
            residual_writes={(1, self.model.cfg.n_ctx - 1): vec},
        )
        self.assertGreater(res.logit_diff().abs().max().item(), 1e-5)

    def test_residual_write_matches_last_layer_feature_clamp(self):
        """At the last tracked layer (no downstream re-encode) writing
        ``delta * W_dec[f]`` equals clamping feature ``f`` to ``clean + delta``."""
        base = self._baseline()
        last = self.model.cfg.n_layers - 1
        target = next(
            key for key, act in base.clean_feature_acts.items() if key[0] == last and act > 1e-3
        )
        layer, pos, feat = target
        clean = base.clean_feature_acts[target]
        delta = 2.5

        clamp = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[
                FeatureIntervention(layer=layer, pos=pos, feature=feat, value=clean + delta)
            ],
            layers=self.layers,
        )
        write_vec = delta * self.transcoders[layer].W_dec[feat].detach().float()
        write = run_feature_intervention(
            self.model,
            self.transcoders,
            self.tokens,
            interventions=[],
            layers=self.layers,
            residual_writes={(layer, pos): write_vec},
        )
        torch.testing.assert_close(
            write.intervened_logits, clamp.intervened_logits, atol=1e-4, rtol=1e-4
        )

    def test_residual_write_untracked_layer_rejected(self):
        d_model = self.model.cfg.d_model
        with self.assertRaises(ValueError):
            run_feature_intervention(
                self.model,
                self.transcoders,
                self.tokens,
                interventions=[],
                layers=[1],  # layer 0 untracked
                residual_writes={(0, 0): torch.zeros(d_model)},
            )

    def test_value_and_factor_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            FeatureIntervention(layer=0, pos=0, feature=0, value=1.0, factor=2.0)

    def test_untracked_layer_intervention_rejected(self):
        with self.assertRaises(ValueError):
            run_feature_intervention(
                self.model,
                self.transcoders,
                self.tokens,
                interventions=[FeatureIntervention(layer=0, pos=0, feature=0)],
                layers=[1],  # only layer 1 tracked; layer-0 intervention invalid
            )


if __name__ == "__main__":
    unittest.main()
