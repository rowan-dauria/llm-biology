"""Sample feature labels and their activating Pile windows for manual checking.

Usage:
    python feature_lookup/compare_labels_to_windows.py
    python -m feature_lookup.compare_labels_to_windows --samples_per_layer 10

The report juxtaposes each saved label/rationale from ``data/feature_labels``
with the feature's saved top-K activating windows, reconstructed by re-streaming
the corpus referenced by ``data/feature_topk/topk_layer_<L>.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

try:
    from .windows import active_prompt_ids, collect_prompt_texts, get_windows, load_tokenizer
except ImportError:
    from windows import active_prompt_ids, collect_prompt_texts, get_windows, load_tokenizer

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_LABELS_DIR = DATA_DIR / "feature_labels"
DEFAULT_TOPK_DIR = DATA_DIR / "feature_topk"
DEFAULT_OUTPUT = DEFAULT_LABELS_DIR / "label_window_sample.txt"
DEFAULT_SAMPLES_PER_LAYER = 10
DEFAULT_WINDOW = 10


def _layer_from_label_path(path: Path) -> int:
    stem = path.stem
    prefix = "layer_"
    if not stem.startswith(prefix):
        raise ValueError(f"Expected label file named layer_<N>.jsonl, got {path}")
    return int(stem.removeprefix(prefix))


def _discover_label_paths(labels_dir: Path, layers: list[int] | None) -> list[Path]:
    if layers is not None:
        return [labels_dir / f"layer_{layer}.jsonl" for layer in layers]

    paths = sorted(labels_dir.glob("layer_*.jsonl"), key=_layer_from_label_path)
    if not paths:
        raise FileNotFoundError(f"No layer_*.jsonl files found in {labels_dir}")
    return paths


def _read_labels(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "feature" not in record:
                raise ValueError(f"{path}:{line_number} is missing a 'feature' field")
            records.append(record)
    return records


def _sample_records(
    records: list[dict[str, Any]],
    samples_per_layer: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if samples_per_layer <= 0:
        raise ValueError("--samples_per_layer must be positive")
    if len(records) <= samples_per_layer:
        return records
    return rng.sample(records, samples_per_layer)


def _format_header(args: argparse.Namespace, label_paths: list[Path]) -> list[str]:
    seed_text = "system random" if args.seed is None else str(args.seed)
    layers = ", ".join(str(_layer_from_label_path(path)) for path in label_paths)
    return [
        "Feature label / Pile-window sense check",
        "=" * 80,
        f"Layers: {layers}",
        f"Labels dir: {args.labels_dir}",
        f"Top-K dir: {args.topk_dir}",
        f"Samples per layer: {args.samples_per_layer}",
        f"Token window radius: {args.window}",
        f"Seed: {seed_text}",
        "",
    ]


def _format_windows(
    layer_data: dict[str, Any],
    feature: int,
    tokenizer: Any,
    text_by_prompt_id: dict[int, str],
    *,
    window_radius: int,
    max_windows: int | None,
) -> list[str]:
    windows = get_windows(
        layer_data,
        feature,
        tokenizer,
        window=window_radius,
        text_by_prompt_id=text_by_prompt_id,
    )
    if max_windows is not None:
        windows = windows[:max_windows]

    lines: list[str] = []
    for window in windows:
        if not window.active:
            lines.append(f"  #{window.rank}: {window.rendered}")
            continue
        lines.append(
            "  "
            f"#{window.rank}: val={window.value:.3f} "
            f"prompt_id={window.prompt_id} token_pos={window.token_pos}"
        )
        lines.append(f"      {window.rendered}")
    return lines or ["  (no windows)"]


def _append_layer_report(
    lines: list[str],
    *,
    layer: int,
    label_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    topk_dir: Path,
    corpus_override: str | None,
    window_radius: int,
    max_windows: int | None,
    tokenizer_cache: dict[str, Any],
) -> None:
    topk_path = topk_dir / f"topk_layer_{layer}.pt"
    if not topk_path.exists():
        raise FileNotFoundError(f"Missing top-K file for layer {layer}: {topk_path}")

    layer_data = torch.load(topk_path, weights_only=False)
    corpus_spec = corpus_override or str(layer_data["corpus_spec"])
    model_id = str(layer_data["model_id"])
    tokenizer = tokenizer_cache.get(model_id)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_id)
        tokenizer_cache[model_id] = tokenizer

    needed_prompt_ids: set[int] = set()
    for record in selected_records:
        needed_prompt_ids.update(active_prompt_ids(layer_data, int(record["feature"])))
    text_by_prompt_id = collect_prompt_texts(corpus_spec, needed_prompt_ids)

    k = int(layer_data["K"])
    max_windows_text = "all" if max_windows is None else str(max_windows)
    lines.extend(
        [
            "",
            "=" * 80,
            f"Layer {layer}",
            "=" * 80,
            f"Sampled {len(selected_records)} of {len(label_records)} labels.",
            f"Corpus: {corpus_spec}",
            f"Saved top-K per feature: {k}; windows shown per feature: {max_windows_text}",
            "",
        ]
    )

    for index, record in enumerate(selected_records, start=1):
        feature = int(record["feature"])
        label = str(record.get("label", "(missing label)"))
        rationale = str(record.get("rationale", "(missing rationale)"))
        max_activation = record.get("max_activation")
        distinct_prompts = record.get("n_distinct_prompts")

        activation_text = ""
        if isinstance(max_activation, (int, float)) and math.isfinite(max_activation):
            activation_text = f" max_activation={float(max_activation):.3f}"
        prompt_text = (
            f" n_distinct_prompts={distinct_prompts}" if isinstance(distinct_prompts, int) else ""
        )

        lines.extend(
            [
                "-" * 80,
                f"[{index}] layer={layer} feature={feature}{activation_text}{prompt_text}",
                f"label: {label}",
                f"rationale: {rationale}",
                "windows:",
            ]
        )
        lines.extend(
            _format_windows(
                layer_data,
                feature,
                tokenizer,
                text_by_prompt_id,
                window_radius=window_radius,
                max_windows=max_windows,
            )
        )
        lines.append("")


def build_report(args: argparse.Namespace) -> str:
    label_paths = _discover_label_paths(args.labels_dir, args.layers)
    rng = random.Random(args.seed)
    tokenizer_cache: dict[str, Any] = {}
    lines = _format_header(args, label_paths)

    for label_path in label_paths:
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        layer = _layer_from_label_path(label_path)
        label_records = _read_labels(label_path)
        selected_records = _sample_records(label_records, args.samples_per_layer, rng)
        _append_layer_report(
            lines,
            layer=layer,
            label_records=label_records,
            selected_records=selected_records,
            topk_dir=args.topk_dir,
            corpus_override=args.corpus_spec,
            window_radius=args.window,
            max_windows=args.max_windows,
            tokenizer_cache=tokenizer_cache,
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="Directory containing layer_<L>.jsonl label files.",
    )
    parser.add_argument(
        "--topk_dir",
        type=Path,
        default=DEFAULT_TOPK_DIR,
        help="Directory containing topk_layer_<L>.pt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Text report path.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of layers to sample.",
    )
    parser.add_argument(
        "--samples_per_layer",
        type=int,
        default=DEFAULT_SAMPLES_PER_LAYER,
        help="Number of labeled features to sample per layer.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Number of tokens to show on each side of the activating token.",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Optional cap on displayed top-K windows per sampled feature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible samples; omit with --seed_from_system.",
    )
    parser.add_argument(
        "--seed_from_system",
        action="store_true",
        help="Use system randomness instead of the default reproducible seed.",
    )
    parser.add_argument(
        "--corpus_spec",
        default=None,
        help="Override the corpus spec saved in the top-K file.",
    )
    args = parser.parse_args()
    if args.seed_from_system:
        args.seed = None
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max_windows must be positive when provided")

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
