"""Attribution-first feature labelling CLI.

Reads an attribution graph JSON, selects unlabelled ``(layer, feature)``
targets ranked by ``abs(target-logit effect)`` and graph centrality, labels
them via the existing per-layer top-K windows, then patches ``clerp`` on the
graph (and ``label`` on per-feature sidecars).

The torch-heavy labelling import is deferred until after argparse so that
``--help`` and ``--dry-run`` work without a torch install.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from llm_biology.features.graph_targets import GraphTarget, select_unlabeled_targets
from llm_biology.features.labels import load_feature_labels
from llm_biology.features.patch_graph_labels import DEFAULT_SCAN_DIR, patch_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "ui_graphs"
DEFAULT_TOPK_DIR = PROJECT_ROOT / "data" / "feature_topk" / "150k-pile"
DEFAULT_ALPHA = 0.5
DEFAULT_PROVIDER = "transformers"
DEFAULT_CONCURRENCY = 8


def _resolve_graph_path(args: argparse.Namespace) -> Path:
    """Resolve the graph JSON path from ``--graph``, or ``--slug``/``--graph-dir``."""
    if args.graph is not None:
        return args.graph
    if args.slug is None:
        raise SystemExit("provide --graph or --slug")
    return args.graph_dir / f"{args.slug}.json"


def _print_target_summary(targets: list[GraphTarget]) -> None:
    """Print a per-layer count and score-range summary of the selected targets."""
    if not targets:
        print("[INFO] no unlabelled targets")
        return
    by_layer: dict[int, list[GraphTarget]] = defaultdict(list)
    for target in targets:
        by_layer[target.layer].append(target)
    for layer in sorted(by_layer):
        group = by_layer[layer]
        scores = [t.score for t in group]
        print(
            f"[INFO] layer {layer}: {len(group)} targets, "
            f"score range [{min(scores):.3f}, {max(scores):.3f}]"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the attribution-first labelling pipeline."""
    parser = argparse.ArgumentParser(
        description=("Label graph-surfaced unlabelled features then patch the graph JSON."),
    )
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Cap on features labelled this run. Default: no cap.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--provider",
        choices=("transformers", "ollama", "openai", "anthropic"),
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--ollama_host", default=None)
    parser.add_argument("--scan-dir", default=DEFAULT_SCAN_DIR)
    parser.add_argument("--topk-dir", type=Path, default=DEFAULT_TOPK_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected targets; do not call any model.",
    )
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="Write labels but leave the graph JSON untouched.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: select unlabelled graph targets, label them, and patch the graph."""
    args = parse_args()
    graph_path = _resolve_graph_path(args)
    if not graph_path.exists():
        raise SystemExit(f"graph not found: {graph_path}")

    labels = load_feature_labels()
    targets = select_unlabeled_targets(
        graph_path,
        labels,
        top_n=args.top_n,
        alpha=args.alpha,
    )
    print(f"[INFO] {len(targets)} unlabelled targets in {graph_path.name}")
    _print_target_summary(targets)

    if args.dry_run:
        for target in targets:
            print(
                f"  layer={target.layer} feature={target.feature} "
                f"target_effect={target.target_effect:.4f} "
                f"centrality={target.centrality:.4f} "
                f"score={target.score:.4f} "
                f"n_positions={target.n_positions}"
            )
        return

    if not targets:
        return

    # Deferred so --help / --dry-run don't require torch.
    from llm_biology.features.label_features import label_features_subset

    targets_by_layer: dict[int, list[GraphTarget]] = defaultdict(list)
    for target in targets:
        targets_by_layer[target.layer].append(target)

    totals: dict[str, int] = {
        "requested": 0,
        "skipped_existing": 0,
        "no_active_windows": 0,
        "written": 0,
        "failed": 0,
    }
    for layer in sorted(targets_by_layer):
        topk_path = args.topk_dir / f"topk_layer_{layer}.pt"
        if not topk_path.exists():
            print(f"[WARN] layer {layer}: missing {topk_path}; skipped")
            continue
        feature_indices = [target.feature for target in targets_by_layer[layer]]
        result = label_features_subset(
            layer,
            feature_indices,
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            ollama_host=args.ollama_host,
            source_tag="on_demand",
            topk_dir=args.topk_dir,
        )
        print(f"[INFO] layer {layer}: {result}")
        for key, value in result.items():
            totals[key] = totals.get(key, 0) + value

    print(f"[INFO] label totals: {totals}")

    if args.skip_patch:
        print("[INFO] --skip-patch set; graph JSON unchanged")
        return

    fresh_labels = load_feature_labels()
    patch_counts = patch_graph(graph_path, fresh_labels, scan_dir=args.scan_dir)
    print(f"[INFO] patched graph: {patch_counts}")


if __name__ == "__main__":
    main()
