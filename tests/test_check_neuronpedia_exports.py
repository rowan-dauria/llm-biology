from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_neuronpedia_exports import check_exports


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class CheckNeuronpediaExportsTests(unittest.TestCase):
    def test_check_exports_validates_graphs_and_features_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_dir = root / "schemas"
            graph_dir = root / "graphs"
            _write_json(
                schema_dir / "attribution-graph.json",
                {
                    "type": "object",
                    "properties": {
                        "metadata": {"type": "object"},
                        "qParams": {"type": "object"},
                        "nodes": {"type": "array"},
                        "links": {"type": "array"},
                    },
                    "required": ["metadata", "qParams", "nodes", "links"],
                },
            )
            _write_json(
                schema_dir / "feature-details.json",
                {
                    "type": "object",
                    "properties": {
                        "index": {"type": "number"},
                        "examples_quantiles": {"type": "array"},
                    },
                    "required": ["index", "examples_quantiles"],
                },
            )
            _write_json(graph_dir / "graph-metadata.json", {"graphs": "not validated"})
            _write_json(
                graph_dir / "ok.json",
                {"metadata": {}, "qParams": {}, "nodes": [], "links": []},
            )
            feature_path = graph_dir / "features" / "qwen3-4b" / "1.json"
            _write_json(feature_path, {"index": 1, "examples_quantiles": []})

            before = feature_path.read_text(encoding="utf-8")
            issues = check_exports(graph_dir=graph_dir, schema_dir=schema_dir)

            self.assertEqual(issues, [])
            self.assertEqual(feature_path.read_text(encoding="utf-8"), before)

    def test_check_exports_reports_schema_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_dir = root / "schemas"
            graph_dir = root / "graphs"
            _write_json(
                schema_dir / "attribution-graph.json",
                {
                    "type": "object",
                    "properties": {"nodes": {"type": "array"}},
                    "required": ["nodes"],
                },
            )
            _write_json(
                schema_dir / "feature-details.json",
                {
                    "type": "object",
                    "properties": {"index": {"type": "number"}},
                    "required": ["index"],
                },
            )
            _write_json(graph_dir / "bad.json", {"nodes": "nope"})
            _write_json(graph_dir / "features" / "qwen3-4b" / "1.json", {"feature": 1})

            issues = check_exports(graph_dir=graph_dir, schema_dir=schema_dir)

            self.assertEqual(len(issues), 2)
            self.assertEqual(
                {issue.schema_name for issue in issues}, {"attribution-graph", "feature-details"}
            )
            self.assertTrue(any(issue.instance_path == "/nodes" for issue in issues))
            self.assertTrue(any(issue.instance_path == "<root>" for issue in issues))


if __name__ == "__main__":
    unittest.main()
