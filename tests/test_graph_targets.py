from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feature_lookup.graph_targets import (
    compute_centrality,
    parse_feature_node_id,
    select_unlabeled_targets,
)
from feature_lookup.labels import FeatureLabel, FeatureLabelMap


def _make_graph(
    nodes: list[dict[str, object]],
    links: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "metadata": {"slug": "test"},
        "qParams": {},
        "nodes": nodes,
        "links": links,
    }


def _write_graph(tmp_path: Path, graph: dict[str, object]) -> Path:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    return graph_path


def _feature_label(layer: int, feature: int, label: str = "stub") -> FeatureLabel:
    return FeatureLabel(
        layer=layer,
        feature=feature,
        label=label,
        rationale="-",
    )


class ParseNodeIdTests(unittest.TestCase):
    def test_accepts_well_formed_feature_id(self) -> None:
        self.assertEqual(parse_feature_node_id("12_500_3"), (12, 500, 3))

    def test_rejects_embedding(self) -> None:
        with self.assertRaises(ValueError):
            parse_feature_node_id("E_42_0")

    def test_rejects_wrong_part_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_feature_node_id("12_500")
        with self.assertRaises(ValueError):
            parse_feature_node_id("12_500_3_4")

    def test_rejects_non_integer_parts(self) -> None:
        with self.assertRaises(ValueError):
            parse_feature_node_id("12_foo_3")

    def test_rejects_negative_parts(self) -> None:
        with self.assertRaises(ValueError):
            parse_feature_node_id("12_-1_3")


class ComputeCentralityTests(unittest.TestCase):
    def test_sums_abs_weights_aggregating_across_positions(self) -> None:
        keys = {(2, 100), (12, 500)}
        links = [
            # Both endpoints land on (2, 100) — counted on both sides.
            {"source": "2_100_0", "target": "2_100_5", "weight": 0.3},
            # Cross-feature link: weight added to both (2, 100) and (12, 500).
            {"source": "2_100_5", "target": "12_500_3", "weight": -0.6},
            # Embedding endpoint: only the feature end is counted.
            {"source": "E_42_0", "target": "2_100_0", "weight": 0.1},
            # Logit endpoint: not in keys, so only (12, 500) end counts.
            {"source": "12_500_3", "target": "37_999_5", "weight": -0.5},
            # Zero-weight link is ignored.
            {"source": "12_500_3", "target": "2_100_0", "weight": 0.0},
        ]
        centrality = compute_centrality(keys, links)
        self.assertAlmostEqual(centrality[(2, 100)], 0.3 + 0.3 + 0.6 + 0.1)
        self.assertAlmostEqual(centrality[(12, 500)], 0.6 + 0.5)


class SelectTargetsTests(unittest.TestCase):
    def _build_synthetic_graph(self) -> dict[str, object]:
        nodes = [
            {
                "node_id": "2_100_0",
                "feature_type": "cross layer transcoder",
                "influence": 0.8,
                "clerp": "L2 F100",
                "layer": "2",
                "ctx_idx": 0,
            },
            {
                "node_id": "2_100_5",
                "feature_type": "cross layer transcoder",
                "influence": 1.2,
                "clerp": "L2 F100",
                "layer": "2",
                "ctx_idx": 5,
            },
            {
                "node_id": "12_500_3",
                "feature_type": "cross layer transcoder",
                "influence": 0.4,
                "clerp": "L12 F500",
                "layer": "12",
                "ctx_idx": 3,
            },
            {
                "node_id": "24_700_2",
                "feature_type": "cross layer transcoder",
                "influence": 2.0,
                "clerp": "L24 F700",
                "layer": "24",
                "ctx_idx": 2,
            },
            {
                "node_id": "E_42_0",
                "feature_type": "embedding",
                "layer": "E",
                "ctx_idx": 0,
                "clerp": "",
            },
            {
                "node_id": "37_999_5",
                "feature_type": "logit",
                "layer": "37",
                "ctx_idx": 5,
                "clerp": 'Output " transport" (p=0.4)',
            },
        ]
        links = [
            {"source": "2_100_0", "target": "2_100_5", "weight": 0.3},
            {"source": "2_100_5", "target": "12_500_3", "weight": -0.6},
            {"source": "E_42_0", "target": "2_100_0", "weight": 0.1},
            {"source": "12_500_3", "target": "24_700_2", "weight": 0.5},
            {"source": "24_700_2", "target": "37_999_5", "weight": -0.9},
        ]
        return _make_graph(nodes, links)

    def test_filters_labelled_aggregates_positions_and_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = _write_graph(Path(tmp), self._build_synthetic_graph())
            labels: FeatureLabelMap = {(12, 500): _feature_label(12, 500)}
            targets = select_unlabeled_targets(graph_path, labels, alpha=0.5)

        self.assertEqual([(t.layer, t.feature) for t in targets], [(24, 700), (2, 100)])

        first, second = targets
        self.assertAlmostEqual(first.target_effect, 2.0)
        self.assertEqual(first.n_positions, 1)
        self.assertAlmostEqual(second.target_effect, 1.2)
        self.assertEqual(second.n_positions, 2)
        # (2, 100) gets weight from the self-feature link counted on both ends.
        self.assertAlmostEqual(second.centrality, 0.3 + 0.3 + 0.6 + 0.1)

    def test_top_n_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = _write_graph(Path(tmp), self._build_synthetic_graph())
            targets = select_unlabeled_targets(graph_path, {}, top_n=1, alpha=0.5)
        self.assertEqual(len(targets), 1)

    def test_alpha_weighting_diverging_metrics(self) -> None:
        nodes = [
            {
                "node_id": "2_100_0",
                "feature_type": "cross layer transcoder",
                "influence": 0.5,
                "clerp": "L2 F100",
                "layer": "2",
                "ctx_idx": 0,
            },
            {
                "node_id": "24_700_0",
                "feature_type": "cross layer transcoder",
                "influence": 2.0,
                "clerp": "L24 F700",
                "layer": "24",
                "ctx_idx": 0,
            },
        ]
        # (2, 100) gets high centrality, (24, 700) gets low centrality.
        links = [
            {"source": "2_100_0", "target": "24_700_0", "weight": 0.5},
            {"source": "E_5_0", "target": "2_100_0", "weight": 1.5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = _write_graph(Path(tmp), _make_graph(nodes, links))
            te_only = select_unlabeled_targets(graph_path, {}, alpha=1.0)
            ce_only = select_unlabeled_targets(graph_path, {}, alpha=0.0)

        self.assertEqual([(t.layer, t.feature) for t in te_only], [(24, 700), (2, 100)])
        self.assertEqual([(t.layer, t.feature) for t in ce_only], [(2, 100), (24, 700)])


if __name__ == "__main__":
    unittest.main()
