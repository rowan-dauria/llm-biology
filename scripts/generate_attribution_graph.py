"""Generate circuit-tracer-frontend graphs with custom Qwen3-4B transcoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biology_server.attribution import (  # noqa: E402
    DEFAULT_EDGE_TOP_K,
    DEFAULT_GRAPH_DIR,
    DEFAULT_LAYERS,
    DEFAULT_MAX_FEATURE_NODES,
    DEFAULT_PROMPT,
    MODEL_ID,
    BiologyAttributionRunner,
    parse_layers,
)
from circuit_graph_export import LOCAL_SCAN  # noqa: E402


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
    parser.add_argument("--max-feature-nodes", type=int, default=DEFAULT_MAX_FEATURE_NODES)
    parser.add_argument("--edge-top-k", type=int, default=DEFAULT_EDGE_TOP_K)
    parser.add_argument("--graph-file-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--scan", default=LOCAL_SCAN)
    parser.add_argument("--feature-dir-name", default=None)
    parser.add_argument("--feature-json-base-url", default=None)
    parser.add_argument("--neuronpedia-source-set", default=None)
    parser.add_argument("--neuronpedia-lorsa-source-set", default=None)
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
        scan=args.scan,
        feature_dir_name=args.feature_dir_name,
        feature_json_base_url=args.feature_json_base_url,
        neuronpedia_source_set=args.neuronpedia_source_set,
        neuronpedia_lorsa_source_set=args.neuronpedia_lorsa_source_set,
    )
    runner.generate_graph(
        args.prompt,
        slug=args.slug,
        target_token_id=args.target_token_id,
        target_token=args.target_token,
        max_feature_nodes=args.max_feature_nodes,
        edge_top_k=args.edge_top_k,
        save_pt=args.save_pt,
    )


if __name__ == "__main__":
    main()
