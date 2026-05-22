from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feature_lookup.labels import FeatureLabel, FeatureLabelMap
from feature_lookup.patch_graph_labels import (
    DEFAULT_SCAN_DIR,
    _cantor_pair,
    patch_graph,
)


def _feature_label(layer: int, feature: int, label: str) -> FeatureLabel:
    return FeatureLabel(layer=layer, feature=feature, label=label, rationale="-")


def _write_graph(graph_path: Path, payload: dict[str, object]) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_feature_example(graph_path: Path, layer: int, feature: int, label: str) -> Path:
    feature_dir = graph_path.parent / "features" / DEFAULT_SCAN_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)
    paired = _cantor_pair(layer, feature)
    sidecar = feature_dir / f"{paired}.json"
    sidecar.write_text(
        json.dumps({"feature": paired, "featureIndex": paired, "label": label}),
        encoding="utf-8",
    )
    return sidecar


class PatchGraphLabelsTests(unittest.TestCase):
    def _graph_payload(self) -> dict[str, object]:
        return {
            "metadata": {"slug": "test"},
            "qParams": {},
            "nodes": [
                {
                    "node_id": "2_100_0",
                    "feature_type": "cross layer transcoder",
                    "clerp": "L2 F100",
                    "layer": "2",
                    "ctx_idx": 0,
                },
                {
                    "node_id": "12_500_3",
                    "feature_type": "cross layer transcoder",
                    "clerp": "L12 F500",
                    "layer": "12",
                    "ctx_idx": 3,
                },
                {
                    "node_id": "E_42_0",
                    "feature_type": "embedding",
                    "clerp": "",
                    "layer": "E",
                    "ctx_idx": 0,
                },
                {
                    "node_id": "37_999_5",
                    "feature_type": "logit",
                    "clerp": 'Output " transport" (p=0.4)',
                    "layer": "37",
                    "ctx_idx": 5,
                },
            ],
            "links": [],
        }

    def test_patches_clerp_and_feature_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = Path(tmp) / "graph.json"
            _write_graph(graph_path, self._graph_payload())
            sidecar = _write_feature_example(graph_path, 2, 100, "stale-label")

            labels: FeatureLabelMap = {
                (2, 100): _feature_label(2, 100, "hemoglobin binding"),
                (12, 500): _feature_label(12, 500, "verb tense"),
            }
            counts = patch_graph(graph_path, labels)

            self.assertEqual(counts["clerp_updated"], 2)
            self.assertEqual(counts["examples_updated"], 1)
            self.assertEqual(counts["examples_missing"], 1)

            updated_graph = json.loads(graph_path.read_text())
            clerps_by_id = {node["node_id"]: node["clerp"] for node in updated_graph["nodes"]}
            self.assertEqual(clerps_by_id["2_100_0"], "hemoglobin binding")
            self.assertEqual(clerps_by_id["12_500_3"], "verb tense")
            self.assertEqual(clerps_by_id["E_42_0"], "")
            self.assertEqual(clerps_by_id["37_999_5"], 'Output " transport" (p=0.4)')

            sidecar_payload = json.loads(sidecar.read_text())
            self.assertEqual(sidecar_payload["label"], "hemoglobin binding")

    def test_no_changes_when_labels_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = Path(tmp) / "graph.json"
            payload = self._graph_payload()
            nodes = payload["nodes"]
            assert isinstance(nodes, list)
            for node in nodes:
                if node.get("node_id") == "2_100_0":
                    node["clerp"] = "hemoglobin binding"
                if node.get("node_id") == "12_500_3":
                    node["clerp"] = "verb tense"
            _write_graph(graph_path, payload)
            _write_feature_example(graph_path, 2, 100, "hemoglobin binding")

            labels: FeatureLabelMap = {
                (2, 100): _feature_label(2, 100, "hemoglobin binding"),
                (12, 500): _feature_label(12, 500, "verb tense"),
            }
            counts = patch_graph(graph_path, labels)

        self.assertEqual(counts["clerp_updated"], 0)
        self.assertEqual(counts["examples_updated"], 0)
        self.assertEqual(counts["examples_missing"], 1)

    def test_falls_back_to_unlabelled_prefix_when_no_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = Path(tmp) / "graph.json"
            payload = self._graph_payload()
            # Seed one node with the new fallback (so no rewrite) and another with
            # a stale clerp (so the fallback overwrite is observable).
            nodes = payload["nodes"]
            assert isinstance(nodes, list)
            for node in nodes:
                if node.get("node_id") == "2_100_0":
                    node["clerp"] = "stale"
                if node.get("node_id") == "12_500_3":
                    node["clerp"] = "[?] L12 F500"
            _write_graph(graph_path, payload)

            counts = patch_graph(graph_path, {})

            # Only the node whose existing clerp differs from the fallback is rewritten.
            self.assertEqual(counts["clerp_updated"], 1)
            updated = json.loads(graph_path.read_text())
            clerps_by_id = {node["node_id"]: node["clerp"] for node in updated["nodes"]}
            self.assertEqual(clerps_by_id["2_100_0"], "[?] L2 F100")
            self.assertEqual(clerps_by_id["12_500_3"], "[?] L12 F500")


if __name__ == "__main__":
    unittest.main()
