"""Collect gate/value activations for a few neurons over a text corpus.

Writes ``<output-dir>/activations.npz`` (``tokens [chunks, seq]``,
``gate``/``value`` ``[neurons, chunks, seq]``) and ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from neuron_labelling.corpus import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    chunk_tokens,
    iter_hf_dataset,
    iter_text_file,
)
from neuron_labelling.model import (
    DEFAULT_MODEL_ID,
    NeuronRecorder,
    NeuronRef,
    Tokenizer,
    load_model,
    logit_effects,
)


def pick_random_neurons(model: Any, count: int, seed: int) -> list[NeuronRef]:
    """Sample neurons from the middle half of the network."""
    rng = random.Random(seed)
    n_layers = model.config.num_hidden_layers
    layers = range(n_layers // 4, max(n_layers // 4 + 1, 3 * n_layers // 4))
    width = model.config.intermediate_size
    return sorted({NeuronRef(rng.choice(layers), rng.randrange(width)) for _ in range(count)})


@torch.no_grad()
def collect_activations(
    model: Any, chunks: Iterable[list[int]], neurons: Sequence[NeuronRef], batch_size: int
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    tokens: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    values: list[np.ndarray] = []
    batch: list[list[int]] = []

    def flush() -> None:
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        with NeuronRecorder(model, neurons) as recorder:
            model.model(input_ids=input_ids)  # decoder only; the LM head is not needed
            gate, value = recorder.read()
        tokens.append(np.asarray(batch, dtype=np.int32))
        gates.append(gate)
        values.append(value)
        batch.clear()

    for chunk in tqdm(chunks, desc="chunks", unit="chunk"):
        batch.append(chunk)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    if not tokens:
        raise ValueError("corpus produced no chunks")
    return {
        "tokens": np.concatenate(tokens, axis=0),
        "gate": np.concatenate(gates, axis=1).astype(np.float32),
        "value": np.concatenate(values, axis=1).astype(np.float32),
    }


def write_collection(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    neurons: Sequence[NeuronRef],
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "activations.npz",
        tokens=arrays["tokens"],
        gate=arrays["gate"],
        value=arrays["value"],
    )
    meta = {**meta, "neurons": [str(n) for n in neurons], "n_tokens": int(arrays["tokens"].size)}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def run_collect(
    model: Any,
    tokenizer: Tokenizer,
    texts: Iterable[str],
    neurons: Sequence[NeuronRef],
    output_dir: Path,
    seq_len: int,
    n_chunks: int,
    batch_size: int,
    meta: dict[str, Any],
) -> None:
    arrays = collect_activations(
        model, chunk_tokens(texts, tokenizer, seq_len, n_chunks), neurons, batch_size
    )
    meta = {
        **meta,
        "seq_len": seq_len,
        "logit_effects": {str(n): logit_effects(model, tokenizer, n) for n in neurons},
    }
    write_collection(output_dir, arrays, neurons, meta)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--neurons", help="comma-separated, e.g. L12/N345,L20/N6789")
    parser.add_argument("--random", type=int, default=0, help="sample this many neurons instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-file", type=Path, help=".txt (one doc per line) or .jsonl")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--n-chunks", type=int, default=4000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collect"))
    args = parser.parse_args(argv)

    model, tokenizer = load_model(args.model_id, args.device, args.dtype)
    if args.neurons:
        neurons = [NeuronRef.parse(part) for part in args.neurons.split(",")]
    elif args.random:
        neurons = pick_random_neurons(model, args.random, args.seed)
    else:
        parser.error("pass --neurons or --random")
    print("neurons:", ", ".join(str(n) for n in neurons))

    if args.text_file:
        texts = iter_text_file(args.text_file)
        source = str(args.text_file)
    else:
        texts = iter_hf_dataset(args.dataset, args.dataset_config)
        source = f"{args.dataset}/{args.dataset_config}"
    try:
        run_collect(
            model,
            tokenizer,
            texts,
            neurons,
            args.output_dir,
            args.seq_len,
            args.n_chunks,
            args.batch_size,
            {"model_id": args.model_id, "corpus": source},
        )
    finally:
        texts.close()
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
