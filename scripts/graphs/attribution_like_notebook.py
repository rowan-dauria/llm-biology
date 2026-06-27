"""Run the TransformerLens attribution path locally on CSD3 (no Colab runtime).

This is the non-Colab counterpart of
``notebooks/dallas_tl_attribution_colab.ipynb``. It runs the
TransformerLens-backed attribution path through ``BiologyAttributionRunner``
and saves the attribution graph JSON (plus an
optional compact ``.pt`` summary and feature sidecars) under
``/home/rd761/rds/hpc-work/<dir-name>``.

Unlike the notebook there is no ``git clone`` / Drive mount / dependency install:
the script lives inside the repo and uses the surrounding modules directly.

Run on a GPU node (A100 ampere). Comprehensive debug logging is written to stdout
and to ``<output-dir>/<slug>.run.log``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default output location requested for CSD3 runs (large-file RDS storage).
DEFAULT_OUTPUT_ROOT = Path("/home/rd761/rds/hpc-work")
DEFAULT_DIR_NAME = "csd3_attribution_graphs"
DEFAULT_PROMPT = "Fact: the capital of the state containing Dallas is"

LOGGER = logging.getLogger("attribution_like_notebook")


def _apply_circuit_tracer_shim() -> None:
    """Keep compatibility with both circuit-tracer factory names.

    The installed v0.4.1 exposes ``load_transcoder``; older builds only expose
    ``load_relu_transcoder``. ``biology_server.attribution`` imports the former,
    so alias it defensively before that import happens.
    """

    slt = importlib.import_module("circuit_tracer.transcoder.single_layer_transcoder")
    if not hasattr(slt, "load_transcoder"):
        if not hasattr(slt, "load_relu_transcoder"):
            raise ImportError(
                "circuit_tracer.transcoder.single_layer_transcoder has neither "
                "load_transcoder nor load_relu_transcoder"
            )
        slt.load_transcoder = slt.load_relu_transcoder  # type: ignore[attr-defined]
        LOGGER.debug("Added load_transcoder alias for circuit-tracer runtime.")
    else:
        LOGGER.debug("circuit-tracer already exposes load_transcoder.")


def setup_logging(log_file: Path, *, level: int = logging.DEBUG) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy_logger in ("fsspec", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    LOGGER.debug("Logging to %s", log_file)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # warn_only until biology_server TL ops are verified deterministic.
    torch.use_deterministic_algorithms(True, warn_only=True)
    LOGGER.info("Global seed set to %d", seed)


def log_environment() -> None:
    import torch

    LOGGER.info("Python executable: %s", sys.executable)
    LOGGER.info("Python version: %s", platform.python_version())
    LOGGER.info("Platform: %s", platform.platform())
    LOGGER.info("torch=%s cuda_available=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        LOGGER.info("gpu=%s", torch.cuda.get_device_name(0))
        LOGGER.info("gpu_memory_gb=%.1f", props.total_memory / 1024**3)
    else:
        LOGGER.warning("CUDA not available — running on CPU/MPS will be very slow.")
    for var in ("HF_HOME", "HF_HUB_CACHE", "CONDA_PREFIX", "SLURM_JOB_ID"):
        value = os.getenv(var)
        if value:
            LOGGER.debug("env %s=%s", var, value)


def parse_args() -> argparse.Namespace:
    # Imported lazily so --help works even if heavy deps are missing.
    from biology_server.attribution import (
        DEFAULT_EDGE_THRESHOLD,
        DEFAULT_LAYERS,
        DEFAULT_LOGIT_PROB_THRESHOLD,
        DEFAULT_MAX_LOGIT_NODES,
        DEFAULT_NODE_THRESHOLD,
        DEFAULT_TOPK_DIR,
        MODEL_ID,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--slug",
        default=None,
        help="Output slug; defaults to '<timestamp>-<slugified-prompt>'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for the run output folder.",
    )
    parser.add_argument(
        "--dir-name",
        default=DEFAULT_DIR_NAME,
        help="Name of the output folder under --output-root.",
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="Comma-separated transformer layer indices to hook.",
    )
    parser.add_argument("--target-token", default=None)
    parser.add_argument("--target-token-id", type=int, default=None)
    # Matches the notebook's A100 settings (BATCH_SIZE=32, MAX_FEATURE_NODES=3000).
    parser.add_argument("--max-feature-nodes", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--node-threshold", type=float, default=DEFAULT_NODE_THRESHOLD)
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    parser.add_argument("--logit-prob-threshold", type=float, default=DEFAULT_LOGIT_PROB_THRESHOLD)
    parser.add_argument("--max-logit-nodes", type=int, default=DEFAULT_MAX_LOGIT_NODES)
    parser.add_argument(
        "--skip-preview-tl-parity-check",
        action="store_true",
        help="Deprecated compatibility flag; the HF preview path has been removed.",
    )
    parser.add_argument(
        "--topk-dir",
        type=Path,
        default=DEFAULT_TOPK_DIR,
        help="Directory with topk_layer_<L>.pt files for feature sidecars.",
    )
    parser.add_argument(
        "--skip-feature-examples",
        action="store_true",
        help="Skip building per-feature sidecars (no topk windows needed).",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply Qwen's chat template. Off by default (factual completion prompt).",
    )
    parser.add_argument(
        "--save-pt",
        nargs="?",
        const="auto",
        default=None,
        help="Save a compact .pt copy; pass a path or omit the value for auto.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--tl-model-id",
        default=None,
        help=(
            "TransformerLens architecture/config model name. Use this when "
            "--model-id points at a local merged checkpoint of a supported base model."
        ),
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help=(
            "Tokenizer source. Use this when --model-id points at a local checkpoint "
            "whose tokenizer files are missing or incompatible."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview-top-k",
        type=int,
        default=10,
        help="Deprecated compatibility option; ignored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_chat_template = args.chat_template
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    from biology_server.attribution import parse_layers, slugify

    slug = args.slug or f"{timestamp}-{slugify(args.prompt[:50])}"
    output_dir = (args.output_root / args.dir_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / f"{slug}.run.log")
    LOGGER.info("=" * 70)
    LOGGER.info("CSD3 attribution run starting")
    LOGGER.info("prompt=%r", args.prompt)
    LOGGER.info("slug=%s", slug)
    LOGGER.info("output_dir=%s", output_dir)
    LOGGER.info("model_id=%s", args.model_id)
    LOGGER.info("tl_model_id=%s", args.tl_model_id or args.model_id)
    LOGGER.info("tokenizer_id=%s", args.tokenizer_id or args.model_id)
    LOGGER.info("use_chat_template=%s", use_chat_template)
    LOGGER.info("=" * 70)

    _apply_circuit_tracer_shim()
    set_seed(args.seed)
    log_environment()

    import biology_server
    import biology_server.attribution as attribution_module

    LOGGER.info("TL backend package: %s", biology_server.__file__)

    if args.skip_feature_examples:

        def _skip_feature_examples(**_kwargs):
            LOGGER.info("Skipping feature-example sidecars (per --skip-feature-examples).")
            return {}

        attribution_module.build_feature_examples = _skip_feature_examples

    layers = parse_layers(args.layers)
    LOGGER.info("layers=%s", layers)
    LOGGER.info("max_feature_nodes=%d batch_size=%d", args.max_feature_nodes, args.batch_size)
    LOGGER.info("topk_dir=%s", args.topk_dir)

    runner = attribution_module.BiologyAttributionRunner(
        layers=layers,
        model_id=args.model_id,
        tl_model_id=args.tl_model_id,
        tokenizer_id=args.tokenizer_id,
        graph_file_dir=output_dir,
        batch_size=args.batch_size,
        max_feature_nodes=args.max_feature_nodes,
        topk_dir=args.topk_dir,
    )

    # --- 1. Run TL attribution and save the graph ----------------------------
    LOGGER.info("-" * 70)
    LOGGER.info("Stage 1: TransformerLens attribution + graph export")
    graph_start = time.time()
    result = runner.generate_graph(
        args.prompt,
        slug=slug,
        target_token_id=args.target_token_id,
        target_token=args.target_token,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        logit_prob_threshold=args.logit_prob_threshold,
        max_logit_nodes=args.max_logit_nodes,
        use_chat_template=use_chat_template,
        save_pt=args.save_pt,
    )
    LOGGER.info("generate_graph completed in %.1fs", time.time() - graph_start)

    # --- 2. Inspect saved artifacts ------------------------------------------
    LOGGER.info("-" * 70)
    LOGGER.info("Stage 2: saved artifacts")
    LOGGER.info("graph_json=%s", result.graph_path)
    LOGGER.info("compact_pt=%s", result.pt_path)
    LOGGER.info("target=%r p=%.6f", result.target_token_str, result.target_token_prob)
    LOGGER.info("feature_nodes=%d", len(result.selected_features))
    LOGGER.info("links=%d", len(result.links))
    LOGGER.info("error_nodes=%d", len(result.error_nodes))
    LOGGER.info("logit_targets=%d", len(result.logit_targets))

    graph_path = Path(result.graph_path)
    if graph_path.exists():
        size_mb = graph_path.stat().st_size / 1024**2
        LOGGER.info("graph json size=%.2f MB", size_mb)
    LOGGER.info("Done. All outputs under %s", output_dir)


if __name__ == "__main__":
    main()
