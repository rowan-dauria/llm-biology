from __future__ import annotations

import json
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


class CorpusPartitioningTests(unittest.TestCase):
    def test_partitioning(self) -> None:
        import torch

        from feature_lookup.corpus import iter_batches

        class MockTokenizer:
            def __init__(self):
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"

            def __call__(self, texts, max_length, truncation, padding, return_tensors):
                n = len(texts)
                return {
                    "input_ids": torch.zeros((n, 10), dtype=torch.long),
                    "attention_mask": torch.ones((n, 10), dtype=torch.long),
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test.jsonl"
            with open(jsonl_path, "w") as f:
                for i in range(10):
                    f.write(json.dumps({"text": f"text {i}"}) + "\n")

            corpus_spec = f"jsonl:{jsonl_path}"
            tokenizer = MockTokenizer()

            # Test part 0 of 2 (expect prompt IDs: 0, 2, 4, 6, 8)
            batches = list(
                iter_batches(corpus_spec, tokenizer, batch_size=2, num_parts=2, part_idx=0)
            )
            self.assertEqual(len(batches), 3)
            pids = torch.cat([b.prompt_ids for b in batches]).tolist()
            self.assertEqual(pids, [0, 2, 4, 6, 8])

            # Test part 1 of 2 (expect prompt IDs: 1, 3, 5, 7, 9)
            batches2 = list(
                iter_batches(corpus_spec, tokenizer, batch_size=2, num_parts=2, part_idx=1)
            )
            self.assertEqual(len(batches2), 3)
            pids2 = torch.cat([b.prompt_ids for b in batches2]).tolist()
            self.assertEqual(pids2, [1, 3, 5, 7, 9])


class TopKMergingTests(unittest.TestCase):
    def test_merge_worker_files(self) -> None:
        import torch

        from feature_lookup import build_topk

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_output_dir = build_topk.OUTPUT_DIR
            build_topk.OUTPUT_DIR = Path(tmpdir)

            try:
                w0_data = {
                    "layer": 2,
                    "topk_vals": torch.tensor([[10.0, 5.0], [8.0, 3.0]]),
                    "topk_prompt_id": torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
                    "topk_token_pos": torch.tensor([[10, 11], [12, 13]], dtype=torch.int16),
                    "K": 2,
                    "corpus_spec": "test",
                    "max_seq_len": 256,
                    "model_id": "test",
                }
                w1_data = {
                    "layer": 2,
                    "topk_vals": torch.tensor([[12.0, 4.0], [7.0, 6.0]]),
                    "topk_prompt_id": torch.tensor([[1, 1], [1, 1]], dtype=torch.int32),
                    "topk_token_pos": torch.tensor([[20, 21], [22, 23]], dtype=torch.int16),
                    "K": 2,
                    "corpus_spec": "test",
                    "max_seq_len": 256,
                    "model_id": "test",
                }

                for layer in build_topk.LAYERS_TO_HOOK:
                    torch.save(w0_data, Path(tmpdir) / f"topk_layer_{layer}_worker_0_of_2.pt")
                    torch.save(w1_data, Path(tmpdir) / f"topk_layer_{layer}_worker_1_of_2.pt")

                build_topk.merge_worker_files(workers=2)

                merged_path = Path(tmpdir) / "topk_layer_2.pt"
                self.assertTrue(merged_path.exists())
                merged = torch.load(merged_path)

                expected_vals = torch.tensor([[12.0, 6.0], [10.0, 5.0]])
                self.assertTrue(torch.allclose(merged["topk_vals"], expected_vals))

                expected_pid = torch.tensor([[1, 1], [0, 0]], dtype=torch.int32)
                self.assertTrue(torch.all(merged["topk_prompt_id"] == expected_pid))

                expected_tpos = torch.tensor([[20, 23], [10, 11]], dtype=torch.int16)
                self.assertTrue(torch.all(merged["topk_token_pos"] == expected_tpos))

                self.assertEqual(int(merged["num_parts"]), 2)

                self.assertFalse((Path(tmpdir) / "topk_layer_2_worker_0_of_2.pt").exists())
                self.assertFalse((Path(tmpdir) / "topk_layer_2_worker_1_of_2.pt").exists())

            finally:
                build_topk.OUTPUT_DIR = orig_output_dir


class PromptIdRecoveryTests(unittest.TestCase):
    def _write_corpus(self, tmpdir: str, n: int) -> str:
        jsonl_path = Path(tmpdir) / "corpus.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(n):
                f.write(json.dumps({"text": f"doc-{i}"}) + "\n")
        return f"jsonl:{jsonl_path}"

    def test_sharded_ids_round_trip_to_text(self) -> None:
        from feature_lookup.corpus import iter_texts
        from feature_lookup.windows import collect_prompt_texts

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._write_corpus(tmpdir, 10)

            num_parts = 2
            seen = {
                pid: text
                for w in range(num_parts)
                for pid, text in iter_texts(spec, part_idx=w, num_parts=num_parts)
            }
            # Encoded ids cover the corpus exactly once and each maps to its own doc.
            self.assertEqual(seen, {i: f"doc-{i}" for i in range(10)})

            needed = {0, 3, 6, 9}
            recovered = collect_prompt_texts(spec, needed, num_parts)
            self.assertEqual(recovered, {pid: seen[pid] for pid in needed})

    def test_single_worker_ids_are_enumerate_index(self) -> None:
        from feature_lookup.corpus import iter_texts
        from feature_lookup.windows import collect_prompt_texts

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._write_corpus(tmpdir, 5)

            self.assertEqual([pid for pid, _ in iter_texts(spec)], [0, 1, 2, 3, 4])
            self.assertEqual(collect_prompt_texts(spec, {1, 4}), {1: "doc-1", 4: "doc-4"})


if __name__ == "__main__":
    unittest.main()
