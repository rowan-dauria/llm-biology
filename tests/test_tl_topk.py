"""Chunk 7 acceptance: iterative top-K selects the most influential features."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer, HookedTransformerConfig

from biology_server.tl_attribution import (
    TargetSpec,
    compute_partial_influences,
    run_attribution,
)
from biology_server.tl_freeze import install_freezes


def _toy_model(seed: int = 0) -> HookedTransformer:
    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_heads=2,
        d_head=4,
        n_ctx=4,
        d_vocab=16,
        act_fn="silu",
        normalization_type="RMS",
    )
    return HookedTransformer(cfg).to("cpu")


def _dense_positive_transcoder(
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
        # Every unmasked feature is active at every non-prefix position, giving
        # 3 * 16 = 48 candidate feature nodes with the default zero-position
        # policy.
        # candidate feature nodes. Decoder rows vary, so logit influence ranks
        # are non-degenerate.
        tc.W_enc.zero_()
        tc.W_dec.copy_(torch.randn(d_transcoder, d_model, generator=g))
        tc.b_enc.copy_(torch.linspace(0.5, 1.5, d_transcoder))
        tc.b_dec.zero_()
    return tc


class TestTLTopK(unittest.TestCase):
    def test_iterative_topk_matches_bruteforce_influence(self) -> None:
        model = _toy_model()
        install_freezes(model)
        transcoders = {0: _dense_positive_transcoder(model.cfg.d_model, 16, layer_idx=0, seed=700)}
        tokens = torch.tensor([[1, 2, 3, 4]])
        logit_targets = [TargetSpec(kind="logit", pos=model.cfg.n_ctx - 1, token_id=5, prob=1.0)]

        all_features, full_matrix = run_attribution(
            model,
            transcoders,
            tokens,
            batch_size=8,
            max_feature_nodes=1000,
            update_interval=1,
            logit_targets=logit_targets,
            layers=[0],
        )
        self.assertGreaterEqual(len(all_features), 40)

        n_nodes = full_matrix.shape[0]
        full_influences = compute_partial_influences(
            full_matrix,
            torch.tensor([1.0]),
            torch.arange(n_nodes),
        )
        expected_top = {
            all_features[idx].node_id
            for idx in torch.argsort(full_influences[: len(all_features)], descending=True)[:10]
        }

        selected, _ = run_attribution(
            model,
            transcoders,
            tokens,
            batch_size=4,
            max_feature_nodes=10,
            update_interval=1,
            logit_targets=logit_targets,
            layers=[0],
        )

        self.assertEqual({feature.node_id for feature in selected}, expected_top)


if __name__ == "__main__":
    unittest.main()
