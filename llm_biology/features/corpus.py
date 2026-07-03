"""Corpus loading and batched tokenisation for the feature top-K pipeline.

Two backends, selected by ``corpus_spec``:

  - ``hf:<repo>:<split>:<n_prompts>`` — stream from a Hugging Face dataset.
    Reads the ``text`` field. Datasets that need a config name aren't supported
    here; use a JSONL dump if you need one of those.
  - ``jsonl:<path>`` — read a local JSON-lines file. Each line must be an
    object with a ``text`` field.

Each prompt becomes one batched window, truncated to ``max_seq_len`` tokens.
Long documents are *not* re-chunked — only the first ``max_seq_len`` tokens are
used. That's fine for a first pass; revisit if coverage needs to improve.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec.compression
import torch
from transformers import AutoTokenizer


def _register_zstd() -> None:
    """Register zstd with fsspec so streamed `.jsonl.zst` Pile shards open.

    fsspec ships codecs for gzip/bz2/xz/zip/lzma but not zstd, even when the
    `zstandard` package is installed. Without this, `monology/pile-uncopyrighted`
    fails with `ValueError: Compression type zstd not supported` after dataset
    streaming starts (which happens *after* the ~80s model load on CSD3).
    """
    try:
        import zstandard
    except ImportError:
        return

    if "zstd" in fsspec.compression.compr:
        return

    def _open(infile, mode="rb", **_kwargs):
        return zstandard.ZstdDecompressor().stream_reader(infile, read_across_frames=True)

    fsspec.compression.register_compression("zstd", _open, ["zst", "zstd"])


_register_zstd()


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_ids: torch.Tensor
    texts: list[str]


def _parse_n_prompts(value: str) -> int | None:
    """Parse the doc-count field of a corpus_spec; ``all``/``-1``/empty mean unlimited."""
    value = value.strip().lower()
    if value in {"", "all", "-1"}:
        return None
    return int(value)


def _iter_hf(
    repo: str, split: str, n_prompts: int | None, part_idx: int, num_parts: int
) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split, streaming=True)
    if num_parts > 1:
        # File-level shard when the dataset has enough underlying shards, so each
        # worker decompresses a disjoint subset rather than the whole stream.
        ds = ds.shard(num_shards=num_parts, index=part_idx)
    count = 0
    for row in ds:
        text = row.get("text")
        if not text:
            continue
        yield text
        count += 1
        if n_prompts is not None and count >= n_prompts:
            return


def _iter_jsonl(path: str, part_idx: int, num_parts: int) -> Iterator[str]:
    doc_idx = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not text:
                continue
            if doc_idx % num_parts == part_idx:
                yield text
            doc_idx += 1


def iter_texts(
    corpus_spec: str, *, part_idx: int = 0, num_parts: int = 1
) -> Iterator[tuple[int, str]]:
    """Yield ``(prompt_id, text)`` over the corpus, optionally sharded across workers.

    With ``num_parts > 1`` each worker reads a disjoint slice (file-level for HF
    datasets via ``IterableDataset.shard``), so the whole corpus is decompressed
    once in total rather than once per worker.

    ``prompt_id`` is a recovery-stable id encoding the shard: the n-th doc seen by
    ``part_idx`` gets ``n * num_parts + part_idx``. To recover its text later,
    decode ``worker = pid % num_parts`` / ``local = pid // num_parts`` and re-run
    ``iter_texts(..., part_idx=worker, num_parts=num_parts)`` — the order is
    deterministic. With ``num_parts == 1`` this is the plain enumerate index,
    identical to the unsharded single-process layout.
    """
    kind, _, rest = corpus_spec.partition(":")
    if not rest:
        raise ValueError(f"Bad corpus_spec: {corpus_spec!r}")
    if kind == "hf":
        repo, split, n_prompts = rest.rsplit(":", 2)
        raw = _iter_hf(repo, split, _parse_n_prompts(n_prompts), part_idx, num_parts)
    elif kind == "jsonl":
        raw = _iter_jsonl(rest, part_idx, num_parts)
    else:
        raise ValueError(f"Unknown corpus kind: {kind!r}")
    for local_id, text in enumerate(raw):
        yield local_id * num_parts + part_idx, text


def iter_batches(
    corpus_spec: str,
    tokenizer,
    *,
    max_seq_len: int = 256,
    batch_size: int = 8,
    device: torch.device | None = None,
    part_idx: int = 0,
    num_parts: int = 1,
) -> Iterator[Batch]:
    buf_texts: list[str] = []
    buf_ids: list[int] = []
    for prompt_id, text in iter_texts(corpus_spec, part_idx=part_idx, num_parts=num_parts):
        buf_texts.append(text)
        buf_ids.append(prompt_id)
        if len(buf_texts) == batch_size:
            yield _make_batch(buf_texts, buf_ids, tokenizer, max_seq_len, device)
            buf_texts, buf_ids = [], []
    if buf_texts:
        yield _make_batch(buf_texts, buf_ids, tokenizer, max_seq_len, device)


def _make_batch(
    texts: list[str],
    ids: list[int],
    tokenizer,
    max_seq_len: int,
    device: torch.device | None,
) -> Batch:
    enc = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_ids = torch.tensor(ids, dtype=torch.int32)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        prompt_ids = prompt_ids.to(device)
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_ids=prompt_ids,
        texts=texts,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_spec", default="hf:monology/pile-uncopyrighted:train:32")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--model_id", default="Qwen/Qwen3-4B")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for i, batch in enumerate(
        iter_batches(
            args.corpus_spec,
            tok,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
    ):
        print(
            f"Batch {i}: "
            f"input_ids {tuple(batch.input_ids.shape)}  "
            f"mask_sum={int(batch.attention_mask.sum().item())}  "
            f"prompt_ids={batch.prompt_ids.tolist()}"
        )


if __name__ == "__main__":
    main()
