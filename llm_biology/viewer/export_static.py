"""Export the graph viewer as static files for read-only hosting."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from llm_biology.viewer.server import (
    DEFAULT_GRAPH_DIR,
    DEFAULT_STATIC_DIR,
    PROJECT_ROOT,
    available_graph_metadata,
    resolve_frontend_dir,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path("dist") / "graph-viewer"
DEFAULT_EXTERNAL_FEATURE_DIR = (
    PROJECT_ROOT.parent / "data" / "llm-biology" / "ui_graphs" / "features"
)


def export_static_viewer(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    graph_file_dir: Path | str = DEFAULT_GRAPH_DIR,
    frontend_dir: Path | str | None = None,
    static_dir: Path | str = DEFAULT_STATIC_DIR,
    feature_dir: Path | str | None = None,
    clean: bool = True,
) -> Path:
    output_path = Path(output_dir).resolve()
    graph_path = Path(graph_file_dir).resolve()
    frontend_path = Path(frontend_dir).resolve() if frontend_dir else resolve_frontend_dir()
    static_path = Path(static_dir).resolve()
    feature_path = _resolve_feature_dir(graph_path, feature_dir)

    if not graph_path.exists():
        raise FileNotFoundError(f"graph file directory does not exist: {graph_path}")
    if not (graph_path / "graph-metadata.json").exists():
        raise FileNotFoundError(f"missing graph metadata: {graph_path / 'graph-metadata.json'}")
    if not frontend_path.exists():
        raise FileNotFoundError(f"frontend directory does not exist: {frontend_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"static directory does not exist: {static_path}")

    input_paths = [graph_path, frontend_path, static_path]
    if feature_path is not None:
        input_paths.append(feature_path)
    _check_output_path(output_path, *input_paths)
    if clean and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    _copy_tree(static_path, output_path)
    _copy_tree(frontend_path, output_path / "ct")
    (output_path / "data").mkdir(parents=True, exist_ok=True)
    if feature_path is not None:
        _copy_tree(feature_path, output_path / "data" / "features")
    write_json(output_path / "data" / "graph-metadata.json", available_graph_metadata(graph_path))

    graph_data_path = output_path / "graph_data"
    graph_data_path.mkdir(parents=True, exist_ok=True)
    for graph_file in sorted(graph_path.glob("*.json")):
        if graph_file.name == "graph-metadata.json":
            continue
        shutil.copy2(graph_file, graph_data_path / graph_file.name)

    return output_path


def _resolve_feature_dir(graph_path: Path, feature_dir: Path | str | None) -> Path | None:
    if feature_dir is not None:
        explicit_path = Path(feature_dir).resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(f"feature directory does not exist: {explicit_path}")
        return explicit_path

    graph_feature_path = graph_path / "features"
    if graph_feature_path.exists():
        return graph_feature_path.resolve()

    if DEFAULT_EXTERNAL_FEATURE_DIR.exists():
        return DEFAULT_EXTERNAL_FEATURE_DIR.resolve()

    return None


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _check_output_path(output_path: Path, *input_paths: Path) -> None:
    for input_path in input_paths:
        if (
            output_path == input_path
            or input_path in output_path.parents
            or output_path in input_path.parents
        ):
            raise ValueError(
                "output directory must not overlap input directories: "
                f"{output_path} and {input_path}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a static read-only graph viewer.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graph-file-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--frontend-dir", type=Path, default=None)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help=(
            "Directory of Neuronpedia feature sidecars to copy to data/features. "
            "Defaults to graph-file-dir/features, then ../data/llm-biology/ui_graphs/features."
        ),
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear the output directory before copying files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_path = export_static_viewer(
        output_dir=args.output_dir,
        graph_file_dir=args.graph_file_dir,
        frontend_dir=args.frontend_dir,
        static_dir=args.static_dir,
        feature_dir=args.feature_dir,
        clean=not args.no_clean,
    )
    print(output_path)


if __name__ == "__main__":
    main()
