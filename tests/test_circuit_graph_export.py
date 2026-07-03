from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_biology.attribution.circuit_graph_export import (
    ErrorNode,
    FeatureNode,
    GraphLink,
    LogitNode,
    embedding_node_id,
    error_node_id,
    export_circuit_graph,
    feature_node_id,
    make_feature_example_payload,
    merge_qparams,
    paired_feature_index,
)


class CircuitGraphExportTests(unittest.TestCase):
    def test_export_circuit_graph_writes_valid_schema_and_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paired = paired_feature_index(2, 123)
            feature_nodes = [
                FeatureNode(
                    layer=2,
                    pos=1,
                    feature=123,
                    activation=4.5,
                    clerp="date token",
                    influence=0.9,
                )
            ]
            links = [
                GraphLink(
                    source=embedding_node_id(42, 0),
                    target=feature_node_id(2, 123, 1),
                    weight=0.2,
                ),
                GraphLink(
                    source=feature_node_id(2, 123, 1),
                    target="37_999_2",
                    weight=0.7,
                ),
            ]
            examples = {
                paired: make_feature_example_payload(
                    feature_index=paired,
                    label="date token",
                    windows=[
                        {
                            "tokens": ["before ", "target", " after"],
                            "tokens_acts_list": [0.0, 1.2, 0.0],
                            "train_token_ind": 1,
                            "is_repeated_datapoint": False,
                        }
                    ],
                )
            }

            graph_path = export_circuit_graph(
                output_dir=tmp_path,
                slug="demo",
                prompt="hello world",
                prompt_tokens=["hello", " world", "!"],
                input_token_ids=[42, 84, 126],
                num_layers=36,
                feature_nodes=feature_nodes,
                links=links,
                target_token_id=999,
                target_token_str=" target",
                target_token_prob=0.42,
                feature_examples=examples,
            )

            graph = json.loads(graph_path.read_text())
            self.assertEqual(graph["metadata"]["slug"], "demo")
            self.assertEqual(graph["metadata"]["scan"], "./data/features/qwen3-4b-transcoders")
            self.assertEqual(graph["metadata"]["node_threshold"], 1.0)
            self.assertEqual(graph["qParams"]["clickedId"], "37_999_2")
            self.assertTrue(any(node["clerp"] == "date token" for node in graph["nodes"]))
            feature_node = next(node for node in graph["nodes"] if node["clerp"] == "date token")
            self.assertEqual(feature_node["influence"], 0.9)

            node_ids = {node["node_id"] for node in graph["nodes"]}
            for link in graph["links"]:
                self.assertIn(link["source"], node_ids)
                self.assertIn(link["target"], node_ids)

            feature_path = tmp_path / "features" / "qwen3-4b-transcoders" / f"{paired}.json"
            feature_payload = json.loads(feature_path.read_text())
            self.assertEqual(feature_payload["label"], "date token")
            example = feature_payload["examples_quantiles"][0]["examples"][0]
            self.assertEqual(example["train_token_ind"], 1)

    def test_feature_example_payload_emits_logit_chips(self) -> None:
        payload = make_feature_example_payload(
            feature_index=1,
            label="x",
            windows=[],
            top_logits=[" alpha", " beta"],
            bottom_logits=[" omega"],
        )
        self.assertEqual(payload["top_logits"], [" alpha", " beta"])
        self.assertEqual(payload["bottom_logits"], [" omega"])

    def test_feature_example_payload_defaults_to_empty_logit_lists(self) -> None:
        payload = make_feature_example_payload(feature_index=1, label="x", windows=[])
        self.assertEqual(payload["top_logits"], [])
        self.assertEqual(payload["bottom_logits"], [])

    def test_export_preserves_precomputed_influence_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = export_circuit_graph(
                output_dir=Path(tmp),
                slug="scores",
                prompt="hi",
                prompt_tokens=["hi"],
                input_token_ids=[1],
                num_layers=36,
                feature_nodes=[
                    FeatureNode(
                        layer=2,
                        pos=0,
                        feature=100,
                        activation=1.0,
                        clerp="a",
                        influence=0.75,
                    ),
                    FeatureNode(
                        layer=3,
                        pos=0,
                        feature=200,
                        activation=1.0,
                        clerp="b",
                        influence=0.25,
                    ),
                ],
                links=[],
                target_token_id=2,
                target_token_str=" out",
                target_token_prob=0.5,
            )

            graph = json.loads(graph_path.read_text())
            influences = {
                node["node_id"]: node["influence"]
                for node in graph["nodes"]
                if node.get("feature_type") == "cross layer transcoder"
            }

            self.assertEqual(influences["2_100_0"], 0.75)
            self.assertEqual(influences["3_200_0"], 0.25)

    def test_export_emits_multiple_logit_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = export_circuit_graph(
                output_dir=Path(tmp),
                slug="multi-logit",
                prompt="hi",
                prompt_tokens=["hi"],
                input_token_ids=[1],
                num_layers=36,
                feature_nodes=[],
                links=[],
                target_token_id=2,
                target_token_str=" out",
                target_token_prob=0.5,
                logit_nodes=[
                    LogitNode(vocab_idx=2, token=" out", token_prob=0.5),
                    LogitNode(vocab_idx=3, token=" alt", token_prob=0.3),
                ],
            )

            graph = json.loads(graph_path.read_text())
            logit_nodes = [node for node in graph["nodes"] if node["feature_type"] == "logit"]
            self.assertEqual([node["node_id"] for node in logit_nodes], ["37_2_0", "37_3_0"])
            self.assertEqual(graph["qParams"]["clickedId"], "37_2_0")

    def test_export_emits_error_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error_id = error_node_id(2, 1)
            graph_path = export_circuit_graph(
                output_dir=Path(tmp),
                slug="errors",
                prompt="hi",
                prompt_tokens=["hi", "!"],
                input_token_ids=[1, 2],
                num_layers=36,
                feature_nodes=[],
                error_nodes=[ErrorNode(layer=2, pos=1)],
                links=[
                    GraphLink(
                        source=error_id,
                        target="37_3_1",
                        weight=0.25,
                    )
                ],
                target_token_id=3,
                target_token_str=" out",
                target_token_prob=0.5,
            )

            graph = json.loads(graph_path.read_text())
            error_node = next(node for node in graph["nodes"] if node["node_id"] == error_id)
            self.assertEqual(error_node["feature_type"], "mlp reconstruction error")
            self.assertEqual(error_node["feature"], -1)
            self.assertEqual(graph["links"][0]["source"], error_id)

    def test_merge_qparams_filters_stale_ids_and_keeps_state(self) -> None:
        existing = {
            "pinnedIds": ["2_100_0", "stale_node", 42],
            "supernodes": [
                ["meaningful group", "2_100_0", "stale_node"],
                ["empty after filter", "stale_node"],
                ["malformed"],
                "not a list",
            ],
            "linkType": "input",
            "clickedId": "stale_node",
            "sg_pos": "12,34",
        }
        node_ids = {"2_100_0", "37_999_5"}
        merged = merge_qparams(existing, node_ids, default_logit_id="37_999_5")

        self.assertEqual(merged["pinnedIds"], ["2_100_0"])
        self.assertEqual(merged["supernodes"], [["meaningful group", "2_100_0"]])
        self.assertEqual(merged["linkType"], "input")
        self.assertEqual(merged["clickedId"], "37_999_5")
        self.assertEqual(merged["sg_pos"], "12,34")

    def test_merge_qparams_preserves_valid_clicked_id(self) -> None:
        existing = {"clickedId": "2_100_0"}
        merged = merge_qparams(existing, {"2_100_0"}, default_logit_id="37_999_5")
        self.assertEqual(merged["clickedId"], "2_100_0")

    def test_merge_qparams_accepts_ui_serialized_pin_state(self) -> None:
        existing = {
            "pinnedIds": "2_100_0,12_42_1,stale_node",
            "supernodes": '[["saved group", "2_100_0", "stale_node"]]',
        }
        merged = merge_qparams(
            existing,
            {"2_100_0", "12_42_1", "37_999_5"},
            default_logit_id="37_999_5",
        )

        self.assertEqual(merged["pinnedIds"], ["2_100_0", "12_42_1"])
        self.assertEqual(merged["supernodes"], [["saved group", "2_100_0"]])

    def test_merge_qparams_defaults_for_missing_fields(self) -> None:
        merged = merge_qparams({}, set(), default_logit_id="37_999_5")
        self.assertEqual(merged["pinnedIds"], [])
        self.assertEqual(merged["supernodes"], [])
        self.assertEqual(merged["linkType"], "both")
        self.assertEqual(merged["clickedId"], "37_999_5")
        self.assertEqual(merged["sg_pos"], "")

    def test_export_preserves_qparams_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_nodes = [
                FeatureNode(layer=2, pos=1, feature=123, activation=4.5, clerp="x"),
                FeatureNode(layer=12, pos=0, feature=42, activation=1.0, clerp="y"),
            ]

            def write(nodes: list[FeatureNode]) -> Path:
                return export_circuit_graph(
                    output_dir=tmp_path,
                    slug="reuse",
                    prompt="hi",
                    prompt_tokens=["hi"],
                    input_token_ids=[1],
                    num_layers=36,
                    feature_nodes=nodes,
                    links=[],
                    target_token_id=2,
                    target_token_str=" out",
                    target_token_prob=0.5,
                )

            graph_path = write(feature_nodes)
            payload = json.loads(graph_path.read_text())
            payload["qParams"]["pinnedIds"] = ["2_123_1", "12_42_0", "ghost_node"]
            payload["qParams"]["supernodes"] = [
                ["my group", "2_123_1", "12_42_0"],
                ["ghost only", "ghost_node"],
            ]
            payload["qParams"]["clickedId"] = "2_123_1"
            payload["qParams"]["linkType"] = "output"
            graph_path.write_text(json.dumps(payload))

            # Second run drops the second feature node; UI state for surviving
            # IDs should carry over and stale references should be filtered.
            write(feature_nodes[:1])
            updated = json.loads(graph_path.read_text())
            q = updated["qParams"]
            self.assertEqual(q["pinnedIds"], ["2_123_1"])
            self.assertEqual(q["supernodes"], [["my group", "2_123_1"]])
            self.assertEqual(q["clickedId"], "2_123_1")
            self.assertEqual(q["linkType"], "output")

    def test_graph_metadata_is_idempotent_by_slug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def write_graph() -> None:
                export_circuit_graph(
                    output_dir=tmp_path,
                    slug="same-slug",
                    prompt="prompt",
                    prompt_tokens=["prompt"],
                    input_token_ids=[1],
                    num_layers=36,
                    feature_nodes=[],
                    links=[],
                    target_token_id=2,
                    target_token_str=" output",
                    target_token_prob=0.5,
                )

            write_graph()
            write_graph()

            metadata = json.loads((tmp_path / "graph-metadata.json").read_text())
            self.assertEqual([entry["slug"] for entry in metadata["graphs"]], ["same-slug"])


if __name__ == "__main__":
    unittest.main()
