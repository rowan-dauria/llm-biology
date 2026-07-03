"""Chunk 3 acceptance tests: transcoder substitution + caching on a toy model."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformer_lens import HookedTransformer, HookedTransformerConfig

from llm_biology.model.tl_forward import HookState, install_transcoder_hooks
from llm_biology.model.tl_freeze import install_freezes


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


class TestTLForward(unittest.TestCase):
    def setUp(self):
        self.model = _toy_model()
        install_freezes(self.model)
        self.d_transcoder = 16
        self.transcoders = {
            layer: _toy_transcoder(
                d_model=self.model.cfg.d_model,
                d_transcoder=self.d_transcoder,
                layer_idx=layer,
                seed=100 + layer,
            )
            for layer in range(self.model.cfg.n_layers)
        }
        self.state = HookState(
            layers=list(range(self.model.cfg.n_layers)),
            transcoders=self.transcoders,
        )
        torch.manual_seed(42)
        self.tokens = torch.randint(0, self.model.cfg.d_vocab, (1, self.model.cfg.n_ctx))

    def _forward(self, capture: dict | None = None):
        fwd_hooks = install_transcoder_hooks(self.model, self.transcoders, self.state)

        if capture is not None:
            for layer in self.state.layers:

                def probe(acts: torch.Tensor, hook, layer=layer) -> torch.Tensor:  # noqa: ARG001
                    capture[layer] = acts.detach().clone()
                    return acts

                fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", probe))

        return self.model.run_with_hooks(self.tokens, fwd_hooks=fwd_hooks)

    def test_layer_features_populated(self):
        self._forward()
        for layer in self.state.layers:
            data = self.state.layer_features[layer]
            self.assertGreater(
                data.feature_ids.numel(),
                0,
                f"layer {layer}: no active features captured (random model should have some)",
            )
            self.assertEqual(
                data.feature_ids.shape,
                data.activations.shape,
            )
            self.assertEqual(
                data.encoder_vectors.shape,
                (data.feature_ids.numel(), self.model.cfg.d_model),
            )
            self.assertEqual(
                data.decoder_vectors.shape,
                (data.feature_ids.numel(), self.model.cfg.d_model),
            )
            # Activations must be positive (post-ReLU).
            self.assertTrue(bool((data.activations > 0).all().item()))

    def test_mlp_out_preserves_original_value(self):
        captured: dict[int, torch.Tensor] = {}
        self._forward(capture=captured)
        for layer in self.state.layers:
            expected = self.state.original_mlp_outputs[layer].unsqueeze(0)
            max_delta = (captured[layer] - expected).abs().max().item()
            self.assertTrue(
                torch.allclose(
                    captured[layer],
                    expected,
                    atol=1e-5,
                    rtol=1e-5,
                ),
                f"layer {layer}: mlp_out did not preserve original value (max |Δ|={max_delta:.3e})",
            )

    def test_feature_and_error_caches_zero_prefix_position(self):
        original_mlp_out: dict[int, torch.Tensor] = {}
        fwd_hooks = install_transcoder_hooks(self.model, self.transcoders, self.state)
        for layer in self.state.layers:

            def capture_original(acts: torch.Tensor, hook, layer=layer) -> torch.Tensor:  # noqa: ARG001
                original_mlp_out[layer] = acts.detach().clone()
                return acts

            fwd_hooks.append((f"blocks.{layer}.mlp.hook_out", capture_original))

        self.model.run_with_hooks(self.tokens, fwd_hooks=fwd_hooks)

        for layer in self.state.layers:
            mlp_in = self.state.mlp_inputs[layer]
            tc = self.transcoders[layer]
            with torch.no_grad():
                features = tc.encode(mlp_in)
                features[:, 0, :] = 0
                reconstruction = tc.decode(features.to(tc.W_dec.dtype), None).to(
                    original_mlp_out[layer].dtype
                )
                expected_error = original_mlp_out[layer][0] - reconstruction[0]
                expected_error[0] = 0

            self.assertTrue(bool((self.state.layer_features[layer].positions != 0).all().item()))
            self.assertTrue(
                torch.allclose(
                    self.state.feature_values[layer][:, 0],
                    torch.zeros_like(self.state.feature_values[layer][:, 0]),
                )
            )
            self.assertTrue(torch.allclose(self.state.reconstructions[layer], reconstruction[0]))
            self.assertTrue(torch.allclose(self.state.error_vectors[layer], expected_error))

    def test_mlp_input_is_post_ln2_weighted_tensor(self):
        for layer in self.state.layers:
            with torch.no_grad():
                self.model.blocks[layer].ln2.w.copy_(
                    torch.linspace(0.5, 1.5, self.model.cfg.d_model)
                )

        normalized: dict[int, torch.Tensor] = {}
        fwd_hooks = install_transcoder_hooks(self.model, self.transcoders, self.state)
        for layer in self.state.layers:

            def probe(acts: torch.Tensor, hook, layer=layer) -> torch.Tensor:  # noqa: ARG001
                normalized[layer] = acts.detach().clone()
                return acts

            fwd_hooks.append((f"blocks.{layer}.ln2.hook_normalized", probe))

        self.model.run_with_hooks(self.tokens, fwd_hooks=fwd_hooks)

        for layer in self.state.layers:
            ln_weight = self.model.blocks[layer].ln2.w
            expected = normalized[layer] * ln_weight
            got = self.state.mlp_inputs[layer]
            self.assertTrue(
                torch.allclose(got, expected, atol=1e-5, rtol=1e-5),
                f"layer {layer}: mlp input was not the post-ln2 weighted tensor",
            )
            self.assertFalse(
                torch.allclose(got, normalized[layer], atol=1e-5, rtol=1e-5),
                f"layer {layer}: mlp input still matches pre-weight hook_normalized",
            )

    def test_final_hidden_is_post_final_norm_unembed_input(self):
        with torch.no_grad():
            self.model.ln_final.w.copy_(torch.linspace(0.5, 1.5, self.model.cfg.d_model))

        normalized: dict[str, torch.Tensor] = {}
        unembed_input: dict[str, torch.Tensor] = {}
        fwd_hooks = install_transcoder_hooks(self.model, self.transcoders, self.state)

        def capture_normalized(acts: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
            normalized["value"] = acts.detach().clone()
            return acts

        def capture_unembed_input(acts: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
            unembed_input["value"] = acts.detach().clone()
            return acts

        fwd_hooks.extend(
            [
                ("ln_final.hook_normalized", capture_normalized),
                ("unembed.hook_in", capture_unembed_input),
            ]
        )
        self.model.run_with_hooks(self.tokens, fwd_hooks=fwd_hooks)

        assert self.state.final_hidden is not None
        expected = normalized["value"] * self.model.ln_final.w
        self.assertTrue(torch.allclose(self.state.final_hidden, unembed_input["value"]))
        self.assertTrue(torch.allclose(self.state.final_hidden, expected, atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.allclose(self.state.final_hidden, normalized["value"]))

    def test_backward_populates_grads(self):
        self._forward()
        final_hidden = self.state.final_hidden
        assert final_hidden is not None
        self.assertTrue(final_hidden.requires_grad)
        final_hidden.backward(torch.ones_like(final_hidden))

        for layer in self.state.layers:
            grad = self.state.output_grads.get(layer)
            assert grad is not None, f"layer {layer}: no output grad captured"
            self.assertEqual(grad.shape, final_hidden.shape)
            self.assertGreater(
                grad.abs().max().item(),
                0.0,
                f"layer {layer}: backward produced all-zero grad at MLP-out",
            )

        embedding_grad = self.state.embedding_grad
        assert embedding_grad is not None
        self.assertGreater(embedding_grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
