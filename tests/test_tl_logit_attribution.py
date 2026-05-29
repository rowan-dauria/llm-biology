"""Chunk 4 acceptance: ``attribute_logit_row`` correctness on a toy model.

The check is analytical: for a tracked source feature at layer ``L``, position
``s``, with scaled decoder vector ``d``, the logit-row score must equal
``d · grad_at_mlp_out_L[0, s]`` where the gradient comes from one backward
through the linearised model seeded with the demeaned unembed vector at
``final_hidden[0, pos]``. We compute that gradient by hand from
``state.output_grads`` (populated by the backward), contract it, and compare
to ``attribute_logit_row``'s output.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer, HookedTransformerConfig

from biology_server_t_lens.tl_attribution import (
    attribute_logit_row,
    collect_feature_scores,
    demeaned_unembed_vector,
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


class TestLogitAttribution(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        install_freezes(self.model)
        self.transcoders = {
            layer: _toy_transcoder(self.model.cfg.d_model, 16, layer, seed=200 + layer)
            for layer in range(self.model.cfg.n_layers)
        }
        self.state = HookState(
            layers=list(range(self.model.cfg.n_layers)),
            transcoders=self.transcoders,
        )
        torch.manual_seed(7)
        self.tokens = torch.randint(0, self.model.cfg.d_vocab, (1, self.model.cfg.n_ctx))

        fwd_hooks = install_transcoder_hooks(self.model, self.transcoders, self.state)
        self.model.run_with_hooks(self.tokens, fwd_hooks=fwd_hooks)
        self.active = finalize_active_features(self.state)
        self.assertGreater(len(self.active), 0, "need active features for a meaningful test")

    def test_demeaned_unembed_recovers_mean_logit_difference(self):
        # Property: residual @ demeaned_unembed_vec == logit[t] - mean_v(logit[v]).
        # Verify on a random residual vector.
        W_U = self.model.W_U
        token_id = 3
        v = demeaned_unembed_vector(W_U, token_id=token_id, dtype=torch.float32)
        torch.manual_seed(11)
        residual = torch.randn(self.model.cfg.d_model)
        logits = residual @ W_U
        expected = float((logits[token_id] - logits.mean()).item())
        got = float((residual @ v).item())
        self.assertAlmostEqual(got, expected, places=4)

    def test_feature_score_equals_decoder_dot_grad_at_mlp_out(self):
        token_id = 5
        pos = self.model.cfg.n_ctx - 1
        feature_scores, embedding_scores = attribute_logit_row(
            self.model, self.state, token_id=token_id, pos=pos
        )
        chunked_scores = collect_feature_scores(
            self.state,
            len(self.active),
            chunk_size=3,
        )
        self.assertTrue(torch.allclose(chunked_scores, feature_scores))

        # Independently contract decoder vectors with the gradients the
        # backward produced. attribute_logit_row left state.output_grads
        # populated, so this is the same gradient — we just verify the
        # contraction matches the legacy formula.
        for f in self.active:
            data = self.state.layer_features[f.layer]
            local_idx = f.score_index - data.start
            d = data.decoder_vectors[local_idx]
            g = self.state.output_grads[f.layer][0, f.pos]
            expected = float((d * g).sum().item())
            got = float(feature_scores[f.score_index].item())
            self.assertAlmostEqual(got, expected, places=5)

        # Embedding scores: contract embedding_grad with token_vectors.
        assert self.state.embedding_grad is not None
        assert self.state.token_vectors is not None
        for p in range(self.model.cfg.n_ctx):
            expected = float(
                (self.state.embedding_grad[0, p] * self.state.token_vectors[p]).sum().item()
            )
            got = float(embedding_scores[p].item())
            self.assertAlmostEqual(got, expected, places=5)

    def test_last_layer_grad_is_per_position(self):
        # The LAST tracked layer's mlp_out goes directly into resid_post_L
        # → ln_final → unembed; ln_final with detached scale is per-position,
        # so grad at last_layer.mlp_out[p] is zero for p != target_pos.
        # (Earlier layers' mlp_outs DO see grad at other positions because
        # later attention mixes them — that's the cross-token mechanic.)
        token_id = 2
        pos = 1
        attribute_logit_row(self.model, self.state, token_id=token_id, pos=pos)
        last_layer = self.state.layers[-1]
        grad = self.state.output_grads[last_layer]
        for p in range(self.model.cfg.n_ctx):
            if p == pos:
                continue
            self.assertAlmostEqual(
                grad[0, p].abs().max().item(),
                0.0,
                places=5,
                msg=f"last layer ({last_layer}), pos {p}: expected zero grad",
            )


if __name__ == "__main__":
    unittest.main()
