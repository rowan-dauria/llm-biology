"""Validate exported Neuronpedia graph and feature-detail JSON files.

This is intentionally read-only: it reports schema failures and never rewrites
existing graph or feature-detail artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit(
        "jsonschema is required: install it or update the project environment"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "ui_graphs"
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "data" / "neuronpedia-schemas"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: Path
    schema_name: str
    instance_path: str
    schema_path: str
    message: str

    def render(self, base_dir: Path) -> str:
        try:
            display_path = self.path.relative_to(base_dir)
        except ValueError:
            display_path = self.path
        return (
            f"{display_path} [{self.schema_name}] "
            f"{self.instance_path}: {self.message} "
            f"(schema: {self.schema_path})"
        )


def _json_path(parts: Any) -> str:
    items = list(parts)
    return "/" + "/".join(str(part) for part in items) if items else "<root>"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_schema(schema_dir: Path, filename: str) -> dict[str, Any]:
    path = schema_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {path}")
    schema = _load_json(path)
    if not isinstance(schema, dict):
        raise TypeError(f"schema must be a JSON object: {path}")
    return schema


def iter_graph_files(graph_dir: Path) -> list[Path]:
    if not graph_dir.exists():
        return []
    return sorted(path for path in graph_dir.glob("*.json") if path.name != "graph-metadata.json")


def iter_feature_files(graph_dir: Path, feature_dir_name: str | None = None) -> list[Path]:
    feature_root = graph_dir / "features"
    if not feature_root.exists():
        return []
    if feature_dir_name:
        return sorted((feature_root / feature_dir_name).glob("*.json"))
    return sorted(feature_root.glob("*/*.json"))


def validate_json_file(
    path: Path,
    *,
    validator: jsonschema.Draft7Validator,
    schema_name: str,
) -> list[ValidationIssue]:
    payload = _load_json(path)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    return [
        ValidationIssue(
            path=path,
            schema_name=schema_name,
            instance_path=_json_path(error.path),
            schema_path=_json_path(error.schema_path),
            message=error.message,
        )
        for error in errors
    ]


def check_exports(
    *,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    schema_dir: Path = DEFAULT_SCHEMA_DIR,
    feature_dir_name: str | None = None,
) -> list[ValidationIssue]:
    graph_schema = _load_schema(schema_dir, "attribution-graph.json")
    feature_schema = _load_schema(schema_dir, "feature-details.json")
    graph_validator = jsonschema.Draft7Validator(graph_schema)
    feature_validator = jsonschema.Draft7Validator(feature_schema)

    issues: list[ValidationIssue] = []
    for path in iter_graph_files(graph_dir):
        issues.extend(
            validate_json_file(path, validator=graph_validator, schema_name="attribution-graph")
        )
    for path in iter_feature_files(graph_dir, feature_dir_name):
        issues.extend(
            validate_json_file(path, validator=feature_validator, schema_name="feature-details")
        )
    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation for Neuronpedia attribution graph exports.",
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument(
        "--feature-dir-name",
        default=None,
        help="Optional features/<name>/ subdirectory to validate. Default: all feature dirs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    issues = check_exports(
        graph_dir=args.graph_dir,
        schema_dir=args.schema_dir,
        feature_dir_name=args.feature_dir_name,
    )
    if not issues:
        print("Neuronpedia export validation passed.")
        return 0

    print(f"Neuronpedia export validation failed: {len(issues)} issue(s).", file=sys.stderr)
    base_dir = args.graph_dir.parent
    for issue in issues:
        print(issue.render(base_dir), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
