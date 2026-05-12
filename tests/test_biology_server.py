from __future__ import annotations

import http.client
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, cast

import torch

from biology_server.attribution import (
    BiologyAttributionRunner,
    GraphResult,
    PreviewResult,
    TokenCandidate,
)
from biology_server.server import serve


class FakeRunner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.preview_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def preview(
        self,
        prompt: str,
        *,
        slug: str | None = None,
        top_k: int | None = None,
    ) -> PreviewResult:
        self.preview_calls.append({"prompt": prompt, "slug": slug, "top_k": top_k})
        return PreviewResult(
            prompt=prompt,
            slug=slug or "fake-slug",
            prompt_tokens=["hello"],
            input_token_ids=[1],
            target_token_id=2,
            target_token_str=" world",
            target_token_prob=0.75,
            top_tokens=[TokenCandidate(token_id=2, token=" world", prob=0.75)],
        )

    def generate_graph(
        self,
        prompt: str,
        *,
        slug: str | None = None,
        target_token_id: int | None = None,
        target_token: str | None = None,
        max_feature_nodes: int,
        edge_top_k: int,
        graph_file_dir: Path | str | None = None,
        save_pt: str | None = None,
    ) -> GraphResult:
        if self.fail:
            raise RuntimeError("boom")
        self.generate_calls.append(
            {
                "prompt": prompt,
                "slug": slug,
                "target_token_id": target_token_id,
                "target_token": target_token,
                "max_feature_nodes": max_feature_nodes,
                "edge_top_k": edge_top_k,
                "save_pt": save_pt,
            }
        )
        if graph_file_dir is None:
            raise AssertionError("graph_file_dir is required")
        out_dir = Path(graph_file_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        graph_slug = slug or "fake-slug"
        graph_path = out_dir / f"{graph_slug}.json"
        graph = {
            "metadata": {
                "slug": graph_slug,
                "scan": "./data/features/qwen3-4b-transcoders",
                "transcoder_list": [],
                "prompt_tokens": ["hello"],
                "prompt": prompt,
                "schema_version": 1,
            },
            "qParams": {"clickedId": "37_2_0"},
            "nodes": [
                {
                    "node_id": "37_2_0",
                    "feature": 2,
                    "layer": "37",
                    "ctx_idx": 0,
                    "feature_type": "logit",
                    "token_prob": 0.75,
                    "is_target_logit": True,
                    "run_idx": 0,
                    "reverse_ctx_idx": 0,
                    "jsNodeId": "L_2-0",
                    "clerp": 'Output " world" (p=0.750)',
                }
            ],
            "links": [],
        }
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        (out_dir / "graph-metadata.json").write_text(
            json.dumps({"graphs": [graph["metadata"]]}),
            encoding="utf-8",
        )
        return GraphResult(
            prompt=prompt,
            slug=graph_slug,
            graph_path=graph_path,
            target_token_id=target_token_id or 2,
            target_token_str=" world",
            target_token_prob=0.75,
            prompt_tokens=["hello"],
            input_token_ids=[1],
            selected_features=[],
            links=[],
        )


class BiologyServerTests(unittest.TestCase):
    def test_runner_tokenizes_chat_template_for_plain_prompt(self) -> None:
        tokenizer = FakeChatTokenizer()
        runner = BiologyAttributionRunner()
        runner._device = torch.device("cpu")

        _inputs, input_token_ids, prompt_tokens = runner._inputs_for_prompt(
            cast(Any, tokenizer),
            "hello",
        )

        self.assertEqual(
            tokenizer.template_calls,
            [
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": True,
                }
            ],
        )
        self.assertEqual(
            tokenizer.tokenized_texts,
            [["<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"]],
        )
        self.assertEqual(input_token_ids, [11, 22, 33])
        self.assertEqual(prompt_tokens, ["tok11", "tok22", "tok33"])

    def test_preview_and_background_graph_job(self) -> None:
        with run_test_server(FakeRunner()) as client:
            preview = client.post("/api/preview", {"prompt": "hello", "slug": "demo"})
            self.assertEqual(preview["slug"], "demo")
            self.assertEqual(preview["target_token"]["id"], 2)

            job = client.post(
                "/api/graphs",
                {
                    "preview_id": preview["preview_id"],
                    "max_feature_nodes": 7,
                    "edge_top_k": 3,
                },
                expected_status=202,
            )
            finished = client.wait_for_job(job["job_id"])
            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["graph_url"], "/graph_data/demo.json")
            graph = client.get("/graph_data/demo.json")
            self.assertEqual(graph["metadata"]["slug"], "demo")

    def test_unknown_preview_id_is_404(self) -> None:
        with run_test_server(FakeRunner()) as client:
            response = client.post(
                "/api/graphs",
                {"preview_id": "missing"},
                expected_status=404,
            )
            self.assertIn("unknown preview_id", response["error"])

    def test_failed_job_reports_error_and_logs(self) -> None:
        with run_test_server(FakeRunner(fail=True)) as client:
            preview = client.post("/api/preview", {"prompt": "hello"})
            job = client.post(
                "/api/graphs",
                {"preview_id": preview["preview_id"]},
                expected_status=202,
            )
            finished = client.wait_for_job(job["job_id"])
            self.assertEqual(finished["status"], "failed")
            self.assertEqual(finished["error"], "boom")
            self.assertTrue(any("RuntimeError: boom" in line for line in finished["logs"]))

    def test_missing_metadata_returns_empty_list(self) -> None:
        with run_test_server(FakeRunner()) as client:
            metadata = client.get("/data/graph-metadata.json")
            self.assertEqual(metadata, {"graphs": []})

    def test_save_graph_updates_qparams(self) -> None:
        with run_test_server(FakeRunner()) as client:
            preview = client.post("/api/preview", {"prompt": "hello", "slug": "demo"})
            job = client.post(
                "/api/graphs", {"preview_id": preview["preview_id"]}, expected_status=202
            )
            client.wait_for_job(job["job_id"])

            client.post("/save_graph/demo", {"qParams": {"clickedId": "changed"}})
            graph = client.get("/graph_data/demo.json")
            self.assertEqual(graph["qParams"], {"clickedId": "changed"})


class HttpTestClient:
    def __init__(self, port: int) -> None:
        self.port = port

    def get(self, path: str, *, expected_status: int = 200) -> dict[str, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        return self._read_json(conn, expected_status)

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return self._read_json(conn, expected_status)

    def wait_for_job(self, job_id: str) -> dict[str, Any]:
        for _ in range(50):
            job = self.get(f"/api/jobs/{job_id}")
            if job["status"] in {"succeeded", "failed"}:
                return job
            time.sleep(0.05)
        raise AssertionError(f"job did not finish: {job_id}")

    def _read_json(self, conn: http.client.HTTPConnection, expected_status: int) -> dict[str, Any]:
        response = conn.getresponse()
        body = response.read()
        conn.close()
        self_status = response.status
        if self_status != expected_status:
            raise AssertionError(f"expected {expected_status}, got {self_status}: {body!r}")
        return json.loads(body.decode("utf-8"))


class FakeBatch:
    def __init__(self) -> None:
        self.input_ids = torch.tensor([[11, 22, 33]])
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> FakeBatch:
        self.device = device
        return self


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []
        self.tokenized_texts: list[list[str]] = []
        self.return_tensors: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, texts: list[str], *, return_tensors: str) -> FakeBatch:
        self.tokenized_texts.append(texts)
        self.return_tensors = return_tensors
        return FakeBatch()

    def batch_decode(self, token_groups: list[list[int]]) -> list[str]:
        return [f"tok{group[0]}" for group in token_groups]


class run_test_server:
    def __init__(self, runner: FakeRunner) -> None:
        self.runner = runner
        self.tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.server = None

    def __enter__(self) -> HttpTestClient:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        graph_dir = root / "graphs"
        frontend_dir = root / "frontend"
        static_dir = root / "static"
        frontend_dir.mkdir()
        static_dir.mkdir()
        (static_dir / "index.html").write_text("ok", encoding="utf-8")
        self.server = serve(
            graph_file_dir=graph_dir,
            frontend_dir=frontend_dir,
            static_dir=static_dir,
            runner=self.runner,  # type: ignore[arg-type]
            port=0,
            host="127.0.0.1",
        )
        return HttpTestClient(self.server.httpd.server_address[1])

    def __exit__(self, *_: object) -> None:
        if self.server is not None:
            self.server.stop()
        if self.tempdir is not None:
            self.tempdir.cleanup()


if __name__ == "__main__":
    unittest.main()
