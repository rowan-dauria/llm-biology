"""Generate circuit-tracer-frontend graphs with custom Qwen3-4B transcoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biology_server.attribution import (  # noqa: E402
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_GRAPH_DIR,
    DEFAULT_LAYERS,
    DEFAULT_LOGIT_PROB_THRESHOLD,
    DEFAULT_MAX_LOGIT_NODES,
    DEFAULT_NODE_THRESHOLD,
    DEFAULT_PROMPT,
    MODEL_ID,
    BiologyAttributionRunner,
    parse_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--slug", default=None)
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="Comma-separated transformer layer indices to hook.",
    )
    parser.add_argument("--target-token", default=None)
    parser.add_argument("--target-token-id", type=int, default=None)
    parser.add_argument("--node-threshold", type=float, default=DEFAULT_NODE_THRESHOLD)
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    parser.add_argument(
        "--logit-prob-threshold",
        type=float,
        default=DEFAULT_LOGIT_PROB_THRESHOLD,
    )
    parser.add_argument("--max-logit-nodes", type=int, default=DEFAULT_MAX_LOGIT_NODES)
    parser.add_argument("--graph-file-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Tokenize the prompt directly instead of applying Qwen's chat template.",
    )
    parser.add_argument(
        "--save-pt",
        nargs="?",
        const="auto",
        default=None,
        help="Optionally save a compact .pt copy; pass a path or omit the value for auto.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = BiologyAttributionRunner(
        layers=parse_layers(args.layers),
        model_id=args.model_id,
        graph_file_dir=args.graph_file_dir,
    )
    runner.generate_graph(
        args.prompt,
        slug=args.slug,
        target_token_id=args.target_token_id,
        target_token=args.target_token,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        logit_prob_threshold=args.logit_prob_threshold,
        max_logit_nodes=args.max_logit_nodes,
        save_pt=args.save_pt,
        use_chat_template=not args.no_chat_template,
    )


if __name__ == "__main__":
    main()
