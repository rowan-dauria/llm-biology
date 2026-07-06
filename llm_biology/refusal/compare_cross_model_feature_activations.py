"""Measure curated transcoder features in a comparison model and classify outcomes.

This implements stages 4-5 of the base-vs-jailbroken comparison:

4. take a human-curated panel of important source-graph features and measure
   those exact ``(layer, pos, feature)`` activations in another model;
5. classify each feature as absent, reduced, shared-active, or ambiguous.

Panel input may be JSON or CSV. JSON can be either a list of rows or an object
with ``metadata`` and ``features`` fields, for example:

{
  "metadata": {
    "prompt": "How do I ...?",
    "prompt_format": "chat",
    "direction": "base_to_jailbroken"
  },
  "features": [
    {
      "node_id": "12_34567_18",
      "label": "refusal / safety policy",
      "category": "refusal",
      "source_activation": 2.31,
      "source_influence": 0.04,
      "prompt_token": " assistant"
    }
  ]
}

Rows must contain either ``node_id`` in frontend form ``layer_feature_pos`` or
the separate integer fields ``layer``, ``pos`` and ``feature``. If
``source_activation`` is omitted, pass ``--source-model-id`` and the script will
measure the source model first, then release it before loading the comparison
model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGGER = logging.getLogger("compare_cross_model_feature_activations")


FeatureKey = tuple[int, int, int]  # (layer, pos, feature)


@dataclass(slots=True)
class PanelRow:
    """One parsed row of the human-curated feature panel, with its resolved feature key."""

    index: int
    key: FeatureKey
    node_id: str
    prompt: str
    prompt_id: str
    direction: str
    label: str
    category: str
    prompt_token: str | None
    source_activation: float | None
    source_influence: float | None
    raw: dict[str, Any]


@dataclass(slots=True)
class ModelMeasurement:
    """Per-prompt measured feature activations and token metadata from one model."""

    activations: dict[tuple[str, FeatureKey], float]
    tokens_by_prompt_id: dict[str, list[str]]
    prompt_lengths: dict[str, int]
    elapsed_seconds: float


def setup_logging() -> None:
    """Configure root logging to stream INFO-and-above to stdout with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_feature_node_id(node_id: str) -> FeatureKey:
    """Parse a frontend feature node id (``"<layer>_<feature>_<pos>"``) into ``(layer, pos, feature)``."""
    parts = node_id.split("_")
    if len(parts) != 3 or parts[0] == "E":
        raise ValueError(f"not a frontend feature node id: {node_id!r}")
    layer, feature, pos = (int(part) for part in parts)
    if layer < 0 or pos < 0 or feature < 0:
        raise ValueError(f"negative feature node id part: {node_id!r}")
    return layer, pos, feature


def feature_node_id(key: FeatureKey) -> str:
    """Build the frontend feature node id (``"<layer>_<feature>_<pos>"``) from a ``FeatureKey``."""
    layer, pos, feature = key
    return f"{layer}_{feature}_{pos}"


def as_float(value: Any) -> float | None:
    """Parse ``value`` as a float, returning ``None`` for blanks, non-numeric, or non-finite values."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any, *, name: str) -> int:
    """Parse ``value`` as a required int field, raising ``ValueError`` naming ``name`` if missing."""
    if value is None or value == "":
        raise ValueError(f"missing required integer field {name!r}")
    return int(value)


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    """Return the first non-empty value among ``row[name]`` for ``name`` in ``names``, else ``None``."""
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def row_key(row: dict[str, Any]) -> tuple[FeatureKey, str]:
    """Resolve a panel row's ``(FeatureKey, node_id)`` from its ``node_id`` or separate layer/pos/feature fields."""
    node_id = first_present(row, ("node_id", "node", "id"))
    if isinstance(node_id, str) and node_id:
        key = parse_feature_node_id(node_id)
        return key, feature_node_id(key)

    layer = as_int(first_present(row, ("layer", "layer_idx")), name="layer")
    pos = as_int(first_present(row, ("pos", "position", "token_pos")), name="pos")
    feature = as_int(
        first_present(row, ("feature", "feature_id", "feature_idx")),
        name="feature",
    )
    key = (layer, pos, feature)
    return key, feature_node_id(key)


def read_panel(
    path: Path, *, cli_prompt: str | None, cli_direction: str | None
) -> tuple[dict[str, Any], list[PanelRow]]:
    """Load a feature panel (JSON or CSV) into ``(metadata, rows)``.

    Accepts a bare list of rows or an object with ``metadata``/``features``
    (or ``rows``/``items``/``panel``) keys. CLI-supplied prompt/direction
    fill in for rows or metadata that omit them.
    """
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            payload: Any = list(csv.DictReader(handle))
        metadata: dict[str, Any] = {}
        raw_rows = payload
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            metadata = {}
            raw_rows = payload
        elif isinstance(payload, dict):
            raw_metadata = payload.get("metadata", {})
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            raw_rows = (
                payload.get("features")
                or payload.get("rows")
                or payload.get("items")
                or payload.get("panel")
            )
        else:
            raise ValueError(f"panel must be a JSON object/list or CSV table: {path}")

    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"panel contains no feature rows: {path}")

    default_prompt = cli_prompt or metadata.get("prompt")
    default_prompt_id = str(metadata.get("prompt_id") or metadata.get("slug") or "prompt_0")
    default_direction = cli_direction or metadata.get("direction") or ""

    rows: list[PanelRow] = []
    for index, item in enumerate(raw_rows):
        if not isinstance(item, dict):
            raise ValueError(f"panel row {index} must be an object")
        key, node_id = row_key(item)
        prompt = first_present(item, ("prompt", "source_prompt")) or default_prompt
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                f"panel row {index} has no prompt; provide metadata.prompt or --prompt"
            )
        prompt_id = str(first_present(item, ("prompt_id", "slug")) or default_prompt_id)
        direction = str(first_present(item, ("direction",)) or default_direction)
        label = str(first_present(item, ("label", "clerp", "description")) or "")
        category = str(first_present(item, ("category", "feature_category", "type")) or "")
        prompt_token = first_present(item, ("prompt_token", "token", "source_token"))
        source_activation = as_float(
            first_present(item, ("source_activation", "graph_activation", "activation"))
        )
        source_influence = as_float(
            first_present(item, ("source_influence", "graph_influence", "influence"))
        )
        rows.append(
            PanelRow(
                index=index,
                key=key,
                node_id=node_id,
                prompt=prompt,
                prompt_id=prompt_id,
                direction=direction,
                label=label,
                category=category,
                prompt_token=str(prompt_token) if prompt_token not in (None, "") else None,
                source_activation=source_activation,
                source_influence=source_influence,
                raw=item,
            )
        )
    return metadata, rows


def prompt_format_from_args(raw: str, metadata: dict[str, Any]) -> str:
    """Resolve ``--prompt-format``: explicit value wins, else infer from panel metadata, else ``"chat"``."""
    if raw != "auto":
        return raw
    panel_format = metadata.get("prompt_format")
    if isinstance(panel_format, str) and panel_format.lower() in {"chat", "direct"}:
        return panel_format.lower()
    chat_template = metadata.get("chat_template")
    if isinstance(chat_template, bool):
        return "chat" if chat_template else "direct"
    return "chat"


def tokenize_prompt(
    tokenizer: Any, prompt: str, *, prompt_format: str, device: Any
) -> tuple[Any, list[str]]:
    """Tokenize ``prompt`` (optionally through the chat template) and return ``(input_ids, tokens)``."""
    from llm_biology.attribution.attribution import prepend_special_prefix

    text = prompt
    if prompt_format == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    input_ids = prepend_special_prefix(tokenizer, input_ids)
    tokens = tokenizer.batch_decode(
        [[int(token_id)] for token_id in input_ids[0].detach().cpu().tolist()]
    )
    return input_ids, tokens


def rows_by_prompt(rows: list[PanelRow]) -> dict[str, list[PanelRow]]:
    """Group panel rows by ``prompt_id`` so each prompt is tokenised/measured only once."""
    grouped: dict[str, list[PanelRow]] = {}
    for row in rows:
        grouped.setdefault(row.prompt_id, []).append(row)
    return grouped


def measure_model(
    *,
    model_id: str,
    rows: list[PanelRow],
    layers: list[int],
    prompt_format: str,
    tl_model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> ModelMeasurement:
    """Load one model + transcoders, measure every panel row's feature activation, then release it.

    Runs one clean :func:`~llm_biology.interventions.tl_intervention.run_feature_intervention`
    per distinct prompt (grouping rows by ``prompt_id``), then frees the model,
    transcoders, and tokenizer and clears the accelerator cache before returning.
    """
    import torch
    from transformers import AutoTokenizer

    from llm_biology.attribution.attribution import CACHE_DIR, load_transcoders, pick_device_dtype
    from llm_biology.interventions.tl_intervention import run_feature_intervention
    from llm_biology.model.tl_model import load_replacement_model

    start = time.time()
    device, dtype = pick_device_dtype()
    LOGGER.info("loading model=%s device=%s dtype=%s", model_id, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id or model_id, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # A local merged checkpoint (e.g. an obliterated/jailbroken export) isn't a
    # HookedTransformer-recognised architecture name. In that case tl_model_id
    # carries the base architecture for TransformerLens while model_id is
    # loaded as the actual weights via hf_model, mirroring
    # BiologyAttributionRunner._ensure_loaded.
    resolved_tl_model_id = tl_model_id or model_id
    hf_model_id = model_id if tl_model_id and tl_model_id != model_id else None
    model = load_replacement_model(
        resolved_tl_model_id,
        device=device,
        dtype=dtype,
        cache_dir=CACHE_DIR,
        hf_model_id=hf_model_id,
    )
    transcoders = load_transcoders(layers, device=device, dtype=dtype)

    activations: dict[tuple[str, FeatureKey], float] = {}
    tokens_by_prompt_id: dict[str, list[str]] = {}
    prompt_lengths: dict[str, int] = {}

    for prompt_id, group in rows_by_prompt(rows).items():
        prompt = group[0].prompt
        if any(row.prompt != prompt for row in group):
            raise ValueError(f"prompt_id={prompt_id!r} contains multiple prompt strings")
        keys = list(dict.fromkeys(row.key for row in group))
        LOGGER.info(
            "measuring prompt_id=%s features=%d tokens_format=%s",
            prompt_id,
            len(keys),
            prompt_format,
        )
        input_ids, tokens = tokenize_prompt(
            tokenizer,
            prompt,
            prompt_format=prompt_format,
            device=device,
        )
        tokens_by_prompt_id[prompt_id] = tokens
        prompt_lengths[prompt_id] = len(tokens)
        LOGGER.info("prompt_id=%s tokenized_length=%d", prompt_id, len(tokens))

        result = run_feature_intervention(
            model,
            transcoders,
            input_ids,
            interventions=[],
            layers=layers,
            measure_features=keys,
        )
        for key, activation in result.clean_feature_acts.items():
            activations[(prompt_id, key)] = activation

    elapsed = time.time() - start
    LOGGER.info("finished model=%s elapsed_seconds=%.1f", model_id, elapsed)

    del model
    del transcoders
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    return ModelMeasurement(
        activations=activations,
        tokens_by_prompt_id=tokens_by_prompt_id,
        prompt_lengths=prompt_lengths,
        elapsed_seconds=elapsed,
    )


def token_status(row: PanelRow, tokens: list[str] | None) -> tuple[str, str | None]:
    """Check a row's recorded prompt token against the actually-tokenized text at that position.

    Returns ``(status, measured_token)`` where ``status`` is one of
    ``not_measured``, ``position_out_of_range``, ``token_mismatch``, ``ok``,
    or ``unchecked`` (no ``prompt_token`` was recorded to compare against).
    """
    if tokens is None:
        return "not_measured", None
    _layer, pos, _feature = row.key
    if pos < 0 or pos >= len(tokens):
        return "position_out_of_range", None
    measured = tokens[pos]
    if row.prompt_token is not None and measured != row.prompt_token:
        return "token_mismatch", measured
    return "ok" if row.prompt_token is not None else "unchecked", measured


def classify(
    *,
    source_activation: float | None,
    comparison_activation: float | None,
    source_token_status: str,
    comparison_token_status: str,
    min_source_activation: float,
    absent_abs_threshold: float,
    absent_ratio_threshold: float,
    reduced_ratio_threshold: float,
) -> str:
    """Classify one feature's cross-model fate from its source/comparison activations and token checks.

    Returns one of ``missing_measurement``, ``ambiguous_token_mismatch``,
    ``source_activation_missing``, ``weak_source``, ``absent``, ``reduced``,
    or ``shared_active``, in that precedence order.
    """
    if comparison_activation is None:
        return "missing_measurement"
    mismatch_statuses = {"position_out_of_range", "token_mismatch"}
    if source_token_status in mismatch_statuses or comparison_token_status in mismatch_statuses:
        return "ambiguous_token_mismatch"
    if source_activation is None:
        return "source_activation_missing"
    if abs(source_activation) < min_source_activation:
        return "weak_source"

    ratio = abs(comparison_activation) / abs(source_activation)
    if abs(comparison_activation) <= absent_abs_threshold or ratio <= absent_ratio_threshold:
        return "absent"
    if ratio <= reduced_ratio_threshold:
        return "reduced"
    return "shared_active"


def finite_ratio(num: float | None, den: float | None) -> float | None:
    """Return ``|num| / |den|``, or ``None`` if either input is missing, zero, or non-finite."""
    if num is None or den is None or den == 0:
        return None
    out = abs(num) / abs(den)
    return out if math.isfinite(out) else None


def default_output_base(*, output_dir: Path, direction: str, comparison_model_id: str) -> Path:
    """Build a timestamped output path stem when ``--output-base`` is omitted."""
    from llm_biology.attribution.attribution import slugify

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    direction_slug = slugify(direction or "feature-panel")
    model_slug = slugify(comparison_model_id.rstrip("/").split("/")[-1])
    return output_dir / f"{stamp}__{direction_slug}__vs-{model_slug}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the classified comparison rows to CSV with a fixed column order."""
    fieldnames = [
        "direction",
        "prompt_id",
        "node_id",
        "layer",
        "pos",
        "feature",
        "label",
        "category",
        "prompt_token",
        "source_token",
        "comparison_token",
        "token_status",
        "source_token_status",
        "comparison_token_status",
        "source_activation",
        "source_activation_panel",
        "source_activation_measured",
        "comparison_activation",
        "activation_ratio_abs",
        "activation_delta",
        "source_influence",
        "outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the cross-model feature-activation comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_panel", type=Path, help="Human-curated JSON/CSV feature panel.")
    parser.add_argument(
        "--comparison-model-id",
        required=True,
        help="Model/path in which to measure the curated source features.",
    )
    parser.add_argument(
        "--comparison-tl-model-id",
        default=None,
        help=(
            "TransformerLens architecture id for --comparison-model-id, e.g. "
            "Qwen/Qwen3-4B. Set this when --comparison-model-id points at a local "
            "merged checkpoint (e.g. an obliterated/jailbroken export) rather than "
            "a HookedTransformer-recognised name."
        ),
    )
    parser.add_argument(
        "--comparison-tokenizer-id",
        default=None,
        help="Tokenizer source for --comparison-model-id. Defaults to --comparison-model-id.",
    )
    parser.add_argument(
        "--source-model-id",
        default=None,
        help=(
            "Optional model/path for measuring source activations. If omitted, every "
            "panel row must contain source_activation/graph_activation/activation."
        ),
    )
    parser.add_argument(
        "--source-tl-model-id",
        default=None,
        help="TransformerLens architecture id for --source-model-id (see --comparison-tl-model-id).",
    )
    parser.add_argument(
        "--source-tokenizer-id",
        default=None,
        help="Tokenizer source for --source-model-id. Defaults to --source-model-id.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to use when not supplied in the panel metadata or rows.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("auto", "chat", "direct"),
        default="auto",
        help=(
            "Prompt formatting. auto reads metadata.prompt_format/chat_template and "
            "falls back to chat."
        ),
    )
    parser.add_argument(
        "--direction",
        default=None,
        help="Label for this comparison, e.g. base_to_jailbroken or jailbroken_to_base.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated tracked layers. Defaults to the layers present in the panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "base_jailbreak_comparison",
    )
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument("--min-source-activation", type=float, default=1e-6)
    parser.add_argument("--absent-abs-threshold", type=float, default=1e-6)
    parser.add_argument("--absent-ratio-threshold", type=float, default=0.05)
    parser.add_argument("--reduced-ratio-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """CLI entry point: measure a curated feature panel in a comparison model and classify outcomes."""
    setup_logging()
    args = parse_args()

    from llm_biology.attribution.attribution import parse_layers

    panel_path = args.feature_panel.expanduser().resolve()
    metadata, panel_rows = read_panel(
        panel_path, cli_prompt=args.prompt, cli_direction=args.direction
    )
    prompt_format = prompt_format_from_args(args.prompt_format, metadata)
    layers = (
        parse_layers(args.layers)
        if args.layers is not None
        else sorted({row.key[0] for row in panel_rows})
    )
    missing_layers = sorted({row.key[0] for row in panel_rows} - set(layers))
    if missing_layers:
        raise ValueError(f"panel contains layers absent from --layers: {missing_layers}")

    if args.source_model_id is None:
        missing_source = [row.node_id for row in panel_rows if row.source_activation is None]
        if missing_source:
            preview = ", ".join(missing_source[:5])
            raise ValueError(
                "source activations are missing for some rows and --source-model-id was not "
                f"provided. First missing node_ids: {preview}"
            )

    LOGGER.info("panel=%s rows=%d", panel_path, len(panel_rows))
    LOGGER.info("layers=%s prompt_format=%s", layers, prompt_format)
    LOGGER.info("comparison_model_id=%s", args.comparison_model_id)
    if args.source_model_id:
        LOGGER.info("source_model_id=%s", args.source_model_id)

    source_measurement = None
    if args.source_model_id is not None:
        source_measurement = measure_model(
            model_id=args.source_model_id,
            rows=panel_rows,
            layers=layers,
            prompt_format=prompt_format,
            tl_model_id=args.source_tl_model_id,
            tokenizer_id=args.source_tokenizer_id,
        )

    comparison_measurement = measure_model(
        model_id=args.comparison_model_id,
        rows=panel_rows,
        layers=layers,
        prompt_format=prompt_format,
        tl_model_id=args.comparison_tl_model_id,
        tokenizer_id=args.comparison_tokenizer_id,
    )

    output_rows: list[dict[str, Any]] = []
    for row in panel_rows:
        source_measured = (
            source_measurement.activations.get((row.prompt_id, row.key))
            if source_measurement is not None
            else None
        )
        source_activation = (
            source_measured if source_measured is not None else row.source_activation
        )
        comparison_activation = comparison_measurement.activations.get((row.prompt_id, row.key))

        source_tokens = (
            source_measurement.tokens_by_prompt_id.get(row.prompt_id)
            if source_measurement is not None
            else None
        )
        comparison_tokens = comparison_measurement.tokens_by_prompt_id.get(row.prompt_id)
        source_status, source_token = token_status(row, source_tokens)
        comparison_status, comparison_token = token_status(row, comparison_tokens)

        ratio = finite_ratio(comparison_activation, source_activation)
        delta = (
            comparison_activation - source_activation
            if comparison_activation is not None and source_activation is not None
            else None
        )
        outcome = classify(
            source_activation=source_activation,
            comparison_activation=comparison_activation,
            source_token_status=source_status,
            comparison_token_status=comparison_status,
            min_source_activation=args.min_source_activation,
            absent_abs_threshold=args.absent_abs_threshold,
            absent_ratio_threshold=args.absent_ratio_threshold,
            reduced_ratio_threshold=args.reduced_ratio_threshold,
        )
        layer, pos, feature = row.key
        output_rows.append(
            {
                "direction": row.direction,
                "prompt_id": row.prompt_id,
                "prompt": row.prompt,
                "node_id": row.node_id,
                "layer": layer,
                "pos": pos,
                "feature": feature,
                "label": row.label,
                "category": row.category,
                "prompt_token": row.prompt_token,
                "source_token": source_token,
                "comparison_token": comparison_token,
                "token_status": comparison_status,
                "source_token_status": source_status,
                "comparison_token_status": comparison_status,
                "source_activation": source_activation,
                "source_activation_panel": row.source_activation,
                "source_activation_measured": source_measured,
                "comparison_activation": comparison_activation,
                "activation_ratio_abs": ratio,
                "activation_delta": delta,
                "source_influence": row.source_influence,
                "outcome": outcome,
                "raw": row.raw,
            }
        )

    outcome_counts = dict(Counter(row["outcome"] for row in output_rows))
    category_counts = dict(Counter(row["category"] or "(uncategorised)" for row in output_rows))
    direction = args.direction or metadata.get("direction") or "feature-panel"
    output_base = (
        args.output_base.expanduser().resolve()
        if args.output_base is not None
        else default_output_base(
            output_dir=args.output_dir.expanduser().resolve(),
            direction=str(direction),
            comparison_model_id=args.comparison_model_id,
        )
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "panel": str(panel_path),
            "panel_metadata": metadata,
            "source_model_id": args.source_model_id,
            "comparison_model_id": args.comparison_model_id,
            "prompt_format": prompt_format,
            "layers": layers,
            "thresholds": {
                "min_source_activation": args.min_source_activation,
                "absent_abs_threshold": args.absent_abs_threshold,
                "absent_ratio_threshold": args.absent_ratio_threshold,
                "reduced_ratio_threshold": args.reduced_ratio_threshold,
            },
            "source_elapsed_seconds": (
                source_measurement.elapsed_seconds if source_measurement is not None else None
            ),
            "comparison_elapsed_seconds": comparison_measurement.elapsed_seconds,
        },
        "summary": {
            "n_features": len(output_rows),
            "outcome_counts": outcome_counts,
            "category_counts": category_counts,
        },
        "prompt_tokens": {
            "source": (
                source_measurement.tokens_by_prompt_id if source_measurement is not None else {}
            ),
            "comparison": comparison_measurement.tokens_by_prompt_id,
        },
        "rows": output_rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    write_csv(csv_path, output_rows)

    LOGGER.info("wrote %s", json_path)
    LOGGER.info("wrote %s", csv_path)
    LOGGER.info("outcome_counts=%s", outcome_counts)


if __name__ == "__main__":
    main()
