"""Chunk 2 acceptance test: real Qwen3-4B load + Dallas forward parity.

This test downloads / loads Qwen3-4B (~8 GB on disk, ~16 GB resident in fp32) and
is therefore slow. It is gated behind ``SKIP_SLOW_TESTS`` so CI / quick runs
skip it by default.
"""

from __future__ import annotations

import os
import time
import unittest

import torch

from llm_biology.model.tl_model import load_replacement_model

DALLAS_PROMPT = "Fact: the capital of the state containing Dallas is"
EXPECTED_TOKEN = " Austin"
MIN_PROB = 0.92


@unittest.skipIf(
    os.getenv("SKIP_SLOW_TESTS"),
    "slow test gated by SKIP_SLOW_TESTS",
)
class TestTLModelDallas(unittest.TestCase):
    def test_dallas_forward_parity(self):
        t0 = time.time()
        model = load_replacement_model(
            "Qwen/Qwen3-4B",
            device="cpu",
            dtype=torch.float32,
        )
        load_seconds = time.time() - t0

        tokenizer = model.tokenizer
        self.assertIsNotNone(tokenizer)

        tokens = tokenizer(DALLAS_PROMPT, return_tensors="pt").input_ids
        t1 = time.time()
        with torch.no_grad():
            logits = model(tokens)
        forward_seconds = time.time() - t1

        last = logits[0, -1, :]
        probs = torch.softmax(last.float(), dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        top_str = tokenizer.decode([int(top_id)])

        print(
            f"\n[tl_model] load={load_seconds:.1f}s "
            f"forward={forward_seconds:.1f}s "
            f"top1={top_str!r} p={top_prob.item():.4f}"
        )

        self.assertEqual(top_str, EXPECTED_TOKEN)
        self.assertGreaterEqual(top_prob.item(), MIN_PROB)


if __name__ == "__main__":
    unittest.main()
