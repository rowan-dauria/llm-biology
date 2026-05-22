from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_lookup.labels import load_feature_labels


class LoadFeatureLabelsTests(unittest.TestCase):
    def test_loads_valid_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_dir = Path(tmp)
            (labels_dir / "layer_2.jsonl").write_text(
                "\n"
                '{"layer": 2, "feature": 89085, "label": "Proper nouns", '
                '"rationale": "Activates on names."}\n',
                encoding="utf-8",
            )

            labels = load_feature_labels({2}, labels_dir=labels_dir)

        self.assertEqual(labels[(2, 89085)].label, "Proper nouns")

    def test_invalid_jsonl_record_names_source_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_dir = Path(tmp)
            (labels_dir / "layer_2.jsonl").write_text(
                'thanks {"layer": 2, "feature": 89085}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                r"layer_2\.jsonl:1: invalid JSON label record: Expecting value",
            ):
                load_feature_labels({2}, labels_dir=labels_dir)


if __name__ == "__main__":
    unittest.main()
