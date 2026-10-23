"""Stream text and cut it into fixed-length token chunks."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path

from neuron_labelling.model import Tokenizer

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"


def iter_text_file(path: Path) -> Generator[str]:
    """One document per line for ``.txt``; a ``text`` field per line for ``.jsonl``."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)["text"] if path.suffix == ".jsonl" else line.rstrip("\n")


def iter_hf_dataset(name: str, config: str | None, split: str = "train") -> Generator[str]:
    from datasets import load_dataset

    for row in load_dataset(name, config, split=split, streaming=True):
        yield row["text"]


def chunk_tokens(
    texts: Iterable[str], tokenizer: Tokenizer, seq_len: int, n_chunks: int
) -> Iterator[list[int]]:
    """Concatenate documents (EOS-separated) and yield ``n_chunks`` chunks of ``seq_len``."""
    buffer: list[int] = []
    produced = 0
    for text in texts:
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]
            produced += 1
            if produced >= n_chunks:
                return
