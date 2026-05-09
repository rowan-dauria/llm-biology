from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from circuit_graph_export import (
    FeatureNode,
    GraphLink,
    embedding_node_id,
    export_circuit_graph,
    feature_node_id,
    make_feature_example_payload,
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
            self.assertEqual(graph["qParams"]["clickedId"], "37_999_2")
            self.assertTrue(any(node["clerp"] == "date token" for node in graph["nodes"]))

            node_ids = {node["node_id"] for node in graph["nodes"]}
            for link in graph["links"]:
                self.assertIn(link["source"], node_ids)
                self.assertIn(link["target"], node_ids)

            feature_path = tmp_path / "features" / "qwen3-4b-transcoders" / f"{paired}.json"
            feature_payload = json.loads(feature_path.read_text())
            self.assertEqual(feature_payload["label"], "date token")
            example = feature_payload["examples_quantiles"][0]["examples"][0]
            self.assertEqual(example["train_token_ind"], 1)

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
