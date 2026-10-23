from collections.abc import Sequence

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


class ByteTokenizer:
    """UTF-8 bytes as tokens; enough of the tokenizer API for the pipeline."""

    eos_token_id = 256
    pad_token_id = 257

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(i for i in token_ids if i < 256).decode("utf-8", errors="replace")


@pytest.fixture
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.fixture
def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=260,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()
