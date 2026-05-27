"""Chunk 1 acceptance tests: freeze harness on a toy TransformerLens model."""

from __future__ import annotations

import unittest

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from biology_server_t_lens.tl_freeze import install_freezes, verify_linearisation


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


class TestTLFreeze(unittest.TestCase):
    def test_verify_linearisation_passes(self):
        model = _toy_model()
        install_freezes(model)
        verify_linearisation(model)

    def test_forward_unchanged(self):
        model = _toy_model()
        tokens = torch.randint(0, model.cfg.d_vocab, (1, model.cfg.n_ctx))
        with torch.no_grad():
            before = model(tokens).clone()
        install_freezes(model)
        with torch.no_grad():
            after = model(tokens).clone()
        self.assertTrue(
            torch.allclose(before, after, atol=0.0, rtol=0.0),
            f"forward changed after install_freezes: max |Δ|={(before - after).abs().max().item():.3e}",
        )

    def test_idempotent(self):
        model = _toy_model()
        install_freezes(model)
        pattern_hooks_first = len(model.blocks[0].attn.hook_pattern.fwd_hooks)
        embed_hooks_first = len(model.hook_embed.fwd_hooks)
        install_freezes(model)
        pattern_hooks_second = len(model.blocks[0].attn.hook_pattern.fwd_hooks)
        embed_hooks_second = len(model.hook_embed.fwd_hooks)
        self.assertEqual(pattern_hooks_first, pattern_hooks_second)
        self.assertEqual(embed_hooks_first, embed_hooks_second)
        # Verification still passes after the duplicate call.
        verify_linearisation(model)

    def test_params_frozen(self):
        model = _toy_model()
        install_freezes(model)
        offenders = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(offenders, [], f"params still trainable: {offenders}")


if __name__ == "__main__":
    unittest.main()
