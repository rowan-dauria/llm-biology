from __future__ import annotations

import http.client
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

import biology_server.server as server_module
from biology_server.attribution import (
    ActiveFeature,
    GraphLink,
    LogitTarget,
    SelectedFeature,
    dense_edge_matrix_links,
    indirect_logit_influence,
    logit_weights_from_dense_matrix,
    prepend_special_prefix,
    prune_edges_by_thresholded_influence,
    prune_graph_by_indirect_influence,
    row_links,
    select_logit_targets,
)
from biology_server.circuit_graph_export import (
    ErrorNode,
    embedding_node_id,
    error_node_id,
    logit_node_id,
)
from biology_server.server import serve


class TinyTokenizer:
    def batch_decode(self, token_batches: list[list[int]]) -> list[str]:
        return [f"T{token_ids[0]}" for token_ids in token_batches]

    def decode(self, token_ids: list[int]) -> str:
        return f"T{token_ids[0]}"

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(token.removeprefix("T"))]


class BiologyServerTests(unittest.TestCase):
    def test_row_links_keeps_top_feature_and_embedding_sources(self) -> None:
        selected = [
            SelectedFeature(
                layer=2,
                pos=0,
                feature=10,
                activation=1.0,
                logit_weight=0.0,
                clerp="x",
                score_index=0,
            ),
            SelectedFeature(
                layer=3,
                pos=1,
                feature=11,
                activation=1.0,
                logit_weight=0.0,
                clerp="y",
                score_index=1,
            ),
        ]

        links = row_links(
            target_id="37_2_1",
            feature_scores=torch.tensor([0.2, -0.5]),
            embedding_scores=torch.tensor([0.4, 0.1]),
            selected=selected,
            input_token_ids=[101, 102],
            edge_top_k=2,
        )

        self.assertEqual(
            [(link.source, link.target, round(link.weight, 3)) for link in links],
            [("3_11_1", "37_2_1", -0.5), ("E_101_0", "37_2_1", 0.4)],
        )

    def test_indirect_logit_influence_uses_target_source_adjacency_orientation(self) -> None:
        links = [GraphLink(source="feature", target="logit", weight=-2.0)]

        scores = indirect_logit_influence(
            node_ids={"feature", "logit"},
            links=links,
            logit_weights={"logit": 0.7},
        )

        self.assertAlmostEqual(scores["feature"], 0.7)
        self.assertAlmostEqual(scores["logit"], 0.0)

    def test_indirect_logit_influence_counts_embedding_feature_logit_paths(self) -> None:
        links = [
            GraphLink(source="embedding", target="feature", weight=3.0),
            GraphLink(source="feature", target="logit", weight=5.0),
        ]

        scores = indirect_logit_influence(
            node_ids={"embedding", "feature", "logit"},
            links=links,
            logit_weights={"logit": 0.4},
        )

        self.assertAlmostEqual(scores["feature"], 0.4)
        self.assertAlmostEqual(scores["embedding"], 0.4)

    def test_prune_graph_keeps_embeddings_logits_and_feature_influence_prefix(self) -> None:
        logit_id = logit_node_id(36, 99, 0)
        feature_a = SelectedFeature(
            layer=2,
            pos=0,
            feature=10,
            activation=1.0,
            logit_weight=0.0,
            clerp="a",
        )
        feature_b = SelectedFeature(
            layer=3,
            pos=0,
            feature=11,
            activation=1.0,
            logit_weight=0.0,
            clerp="b",
        )
        links = [
            GraphLink(source=embedding_node_id(1, 0), target=feature_a.node_id, weight=2.0),
            GraphLink(source=feature_a.node_id, target=logit_id, weight=3.0),
            GraphLink(source=embedding_node_id(1, 0), target=feature_b.node_id, weight=4.0),
        ]

        pruned_features, pruned_links = prune_graph_by_indirect_influence(
            selected=[feature_a, feature_b],
            input_token_ids=[1],
            links=links,
            logit_targets=[LogitTarget(token_id=99, token="T99", prob=0.9, node_id=logit_id)],
            node_threshold=0.8,
            edge_threshold=1.0,
        )

        self.assertEqual([feature.node_id for feature in pruned_features], [feature_a.node_id])
        self.assertEqual(
            {(link.source, link.target) for link in pruned_links},
            {
                (embedding_node_id(1, 0), feature_a.node_id),
                (feature_a.node_id, logit_id),
            },
        )

    def test_node_pruning_keeps_feature_that_crosses_threshold(self) -> None:
        logit_id = logit_node_id(36, 99, 0)
        feature_a = SelectedFeature(
            layer=2,
            pos=0,
            feature=10,
            activation=1.0,
            logit_weight=0.0,
            clerp="a",
        )
        feature_b = SelectedFeature(
            layer=3,
            pos=0,
            feature=11,
            activation=1.0,
            logit_weight=0.0,
            clerp="b",
        )
        links = [
            GraphLink(source=feature_a.node_id, target=logit_id, weight=7.0),
            GraphLink(source=feature_b.node_id, target=logit_id, weight=3.0),
        ]

        pruned_features, _ = prune_graph_by_indirect_influence(
            selected=[feature_a, feature_b],
            input_token_ids=[],
            links=links,
            logit_targets=[LogitTarget(token_id=99, token="T99", prob=1.0, node_id=logit_id)],
            node_threshold=0.8,
            edge_threshold=1.0,
        )

        self.assertEqual(
            [feature.node_id for feature in pruned_features],
            [feature_a.node_id, feature_b.node_id],
        )

    def test_node_pruning_counts_unpruned_embeddings_in_cutoff(self) -> None:
        logit_id = logit_node_id(36, 99, 0)
        feature = SelectedFeature(
            layer=2,
            pos=0,
            feature=10,
            activation=1.0,
            logit_weight=0.0,
            clerp="a",
        )
        embedding_id = embedding_node_id(1, 0)
        links = [
            GraphLink(source=embedding_id, target=logit_id, weight=9.0),
            GraphLink(source=feature.node_id, target=logit_id, weight=1.0),
        ]

        pruned_features, pruned_links = prune_graph_by_indirect_influence(
            selected=[feature],
            input_token_ids=[1],
            links=links,
            logit_targets=[LogitTarget(token_id=99, token="T99", prob=1.0, node_id=logit_id)],
            node_threshold=0.8,
            edge_threshold=1.0,
        )

        self.assertEqual(pruned_features, [])
        self.assertEqual(
            [(link.source, link.target, link.weight) for link in pruned_links],
            [(embedding_id, logit_id, 9.0)],
        )

    def test_dense_matrix_translation_includes_error_nodes_between_features_and_tokens(
        self,
    ) -> None:
        feature = SelectedFeature(
            layer=2,
            pos=0,
            feature=10,
            activation=1.0,
            logit_weight=0.0,
            clerp="a",
            score_index=0,
        )
        error_nodes = [ErrorNode(layer=2, pos=0), ErrorNode(layer=2, pos=1)]
        logit_id = logit_node_id(36, 99, 0)
        matrix = torch.zeros(5, 5)
        matrix[0, 2] = -0.3
        matrix[4, 1] = 0.8

        links = dense_edge_matrix_links(
            selected=[feature],
            error_nodes=error_nodes,
            input_token_ids=[1],
            logit_targets=[LogitTarget(token_id=99, token="T99", prob=0.9, node_id=logit_id)],
            dense_edge_matrix=matrix,
        )
        weights = logit_weights_from_dense_matrix(
            dense_edge_matrix=matrix,
            selected=[
                ActiveFeature(
                    layer=2,
                    pos=0,
                    feature=10,
                    activation=1.0,
                    encoder_vector=torch.zeros(2),
                    score_index=0,
                )
            ],
            logit_targets=[LogitTarget(token_id=99, token="T99", prob=0.9, node_id=logit_id)],
            n_pos=1,
            n_error_nodes=2,
        )

        self.assertEqual(
            {(link.source, link.target, round(link.weight, 3)) for link in links},
            {
                (error_node_id(2, 1), feature.node_id, -0.3),
                (error_node_id(2, 0), logit_id, 0.8),
            },
        )
        self.assertEqual(weights, {0: 0.0})

    def test_prepend_special_prefix_adds_bos_when_first_token_is_content(self) -> None:
        class _Tok:
            all_special_ids = [100, 101]
            bos_token_id = None
            pad_token_id = 100
            eos_token_id = 101

        out = prepend_special_prefix(_Tok(), torch.tensor([[5, 6, 7]]))
        self.assertEqual(out.tolist(), [[100, 5, 6, 7]])

    def test_prepend_special_prefix_noop_when_first_token_already_special(self) -> None:
        class _Tok:
            all_special_ids = [100, 101]
            bos_token_id = 100
            pad_token_id = 100
            eos_token_id = 101

        out = prepend_special_prefix(_Tok(), torch.tensor([[100, 5, 6]]))
        self.assertEqual(out.tolist(), [[100, 5, 6]])

    def test_prepend_special_prefix_warns_when_no_special_token(self) -> None:
        class _Tok:
            all_special_ids: list[int] = []
            bos_token_id = None
            pad_token_id = None
            eos_token_id = None

        with self.assertWarns(UserWarning):
            out = prepend_special_prefix(_Tok(), torch.tensor([[5, 6]]))
        self.assertEqual(out.tolist(), [[5, 6]])

    def test_edge_pruning_uses_target_score_and_preserves_signed_weights(self) -> None:
        links = [
            GraphLink(source="a", target="c", weight=-2.0),
            GraphLink(source="b", target="c", weight=1.0),
            GraphLink(source="c", target="logit", weight=1.0),
        ]

        pruned = prune_edges_by_thresholded_influence(
            links=links,
            node_ids={"a", "b", "c", "logit"},
            logit_weights={"logit": 1.0},
            threshold=0.6,
        )

        self.assertEqual(
            [(link.source, link.target, link.weight) for link in pruned],
            [("a", "c", -2.0), ("c", "logit", 1.0)],
        )

    def test_select_logit_targets_reaches_probability_mass_and_caps_count(self) -> None:
        tokenizer = TinyTokenizer()
        logits = torch.log(torch.tensor([0.50, 0.30, 0.10, 0.05, 0.05]))

        targets = select_logit_targets(
            tokenizer,
            logits,
            pos=4,
            prob_threshold=0.75,
            max_logit_nodes=10,
        )
        capped = select_logit_targets(
            tokenizer,
            logits,
            pos=4,
            prob_threshold=0.99,
            max_logit_nodes=3,
        )

        self.assertEqual([target.token_id for target in targets], [0, 1])
        self.assertEqual(len(capped), 3)

    def test_select_logit_targets_default_cap_matches_paper(self) -> None:
        tokenizer = TinyTokenizer()
        logits = torch.zeros(20)

        targets = select_logit_targets(tokenizer, logits, pos=4)

        self.assertEqual(len(targets), 10)

    def test_missing_metadata_returns_empty_list(self) -> None:
        with run_test_server() as client:
            metadata = client.get("/data/graph-metadata.json")
            self.assertEqual(metadata, {"graphs": []})

    def test_graph_and_feature_asset_routes_use_graph_dir(self) -> None:
        with run_test_server() as client:
            write_demo_graph(client.graph_dir, slug="demo")

            metadata = client.get("/data/graph-metadata.json")
            graph = client.get("/graph_data/demo.json")
            feature_via_data = client.get("/data/features/qwen3-4b-transcoders/1.json")
            feature = client.get("/features/qwen3-4b-transcoders/1.json")

            self.assertEqual(metadata["graphs"][0]["slug"], "demo")
            self.assertEqual(graph["metadata"]["slug"], "demo")
            self.assertEqual(feature_via_data, {"featureIndex": 1})
            self.assertEqual(feature, {"featureIndex": 1})

    def test_page_and_ct_assets_served(self) -> None:
        with run_test_server(ct_assets={"util.js": "window.util = {}"}) as client:
            status, body = client.get_raw("/")
            self.assertEqual(status, 200)
            self.assertEqual(body, "ok")

            status, asset = client.get_raw("/ct/util.js")
            self.assertEqual(status, 200)
            self.assertEqual(asset, "window.util = {}")

            self.assertEqual(client.get_header("/", "Cache-Control"), "no-store")
            self.assertIsNone(client.get_header("/ct/util.js", "Cache-Control"))

    def test_removed_preview_and_job_routes_return_404(self) -> None:
        with run_test_server() as client:
            response = client.post("/api/preview", {"prompt": "hello"}, expected_status=404)
            self.assertEqual(response["error"], "not found")
            response = client.post("/api/graphs", {"prompt": "hello"}, expected_status=404)
            self.assertEqual(response["error"], "not found")
            response = client.get("/api/jobs/demo", expected_status=404)
            self.assertEqual(response["error"], "not found")

    def test_removed_iframe_shims_return_404(self) -> None:
        with run_test_server() as client:
            self.assertEqual(client.get_raw("/ct/data/graph-metadata.json")[0], 404)
            self.assertEqual(client.get_raw("/ct/graph_data/demo.json")[0], 404)

    def test_resolve_frontend_dir_requires_sibling_checkout(self) -> None:
        original_project_root = server_module.PROJECT_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "mphil-project"
            project_root.mkdir()
            server_module.PROJECT_ROOT = project_root
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Could not find circuit-tracer frontend assets",
                ):
                    server_module.resolve_frontend_dir()
            finally:
                server_module.PROJECT_ROOT = original_project_root

    def test_save_graph_updates_qparams(self) -> None:
        with run_test_server() as client:
            write_demo_graph(client.graph_dir, slug="demo")

            client.post("/save_graph/demo", {"qParams": {"clickedId": "changed"}})
            graph = client.get("/graph_data/demo.json")
            self.assertEqual(graph["qParams"], {"clickedId": "changed"})

    def test_graph_data_is_not_cached(self) -> None:
        with run_test_server() as client:
            write_demo_graph(client.graph_dir, slug="demo")

            self.assertEqual(
                client.get_header("/graph_data/demo.json", "Cache-Control"), "no-store"
            )
            self.assertIsNone(
                client.get_header("/data/features/qwen3-4b-transcoders/1.json", "Cache-Control")
            )

    def test_upload_graph_writes_graph_and_metadata(self) -> None:
        with run_test_server() as client:
            graph = upload_graph_payload(slug="saved graph")
            response = client.post(
                "/api/upload_graph",
                {
                    "graph": graph,
                    "slug": "Edited Graph",
                    "filename": "ignored.json",
                },
                expected_status=201,
            )

            self.assertEqual(response["slug"], "edited-graph")
            served = client.get("/graph_data/edited-graph.json")
            self.assertEqual(served["metadata"]["slug"], "edited-graph")
            self.assertEqual(served["qParams"], {"clickedId": "37_2_0"})
            metadata = client.get("/data/graph-metadata.json")
            self.assertEqual([entry["slug"] for entry in metadata["graphs"]], ["edited-graph"])

    def test_upload_graph_uses_metadata_slug_without_override(self) -> None:
        with run_test_server() as client:
            response = client.post(
                "/api/upload_graph",
                {"graph": upload_graph_payload(slug="Saved Graph")},
                expected_status=201,
            )

            self.assertEqual(response["slug"], "saved-graph")
            self.assertEqual(
                client.get("/graph_data/saved-graph.json")["metadata"]["slug"], "saved-graph"
            )

    def test_upload_graph_rejects_invalid_payload(self) -> None:
        with run_test_server() as client:
            response = client.post(
                "/api/upload_graph",
                {"graph": {"metadata": {"slug": "bad"}, "nodes": [], "links": []}},
                expected_status=400,
            )

            self.assertIn("graph.metadata.prompt_tokens", response["error"])


def upload_graph_payload(*, slug: str) -> dict[str, Any]:
    return {
        "metadata": {
            "slug": slug,
            "scan": "./data/features/qwen3-4b-transcoders",
            "transcoder_list": [],
            "prompt_tokens": ["hello"],
            "prompt": "hello",
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


def write_demo_graph(graph_dir: Path, *, slug: str) -> None:
    graph = upload_graph_payload(slug=slug)
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / f"{slug}.json").write_text(json.dumps(graph), encoding="utf-8")
    (graph_dir / "graph-metadata.json").write_text(
        json.dumps({"graphs": [graph["metadata"]]}),
        encoding="utf-8",
    )
    feature_dir = graph_dir / "features" / "qwen3-4b-transcoders"
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "1.json").write_text(json.dumps({"featureIndex": 1}), encoding="utf-8")


class HttpTestClient:
    def __init__(self, port: int, graph_dir: Path) -> None:
        self.port = port
        self.graph_dir = graph_dir

    def get(self, path: str, *, expected_status: int = 200) -> dict[str, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        return self._read_json(conn, expected_status)

    def get_raw(self, path: str) -> tuple[int, str]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        conn.close()
        return response.status, body.decode("utf-8")

    def get_header(self, path: str, name: str) -> str | None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        response.read()
        conn.close()
        return response.getheader(name)

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

    def _read_json(self, conn: http.client.HTTPConnection, expected_status: int) -> dict[str, Any]:
        response = conn.getresponse()
        body = response.read()
        conn.close()
        self_status = response.status
        if self_status != expected_status:
            raise AssertionError(f"expected {expected_status}, got {self_status}: {body!r}")
        return json.loads(body.decode("utf-8"))


class run_test_server:
    def __init__(self, *, ct_assets: dict[str, str] | None = None) -> None:
        self.ct_assets = ct_assets or {}
        self.tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.server = None
        self.graph_dir: Path | None = None

    def __enter__(self) -> HttpTestClient:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.graph_dir = root / "graphs"
        frontend_dir = root / "frontend"
        static_dir = root / "static"
        frontend_dir.mkdir()
        static_dir.mkdir()
        (static_dir / "index.html").write_text("ok", encoding="utf-8")
        for rel_path, content in self.ct_assets.items():
            asset_path = frontend_dir / rel_path
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text(content, encoding="utf-8")
        self.server = serve(
            graph_file_dir=self.graph_dir,
            frontend_dir=frontend_dir,
            static_dir=static_dir,
            port=0,
            host="127.0.0.1",
        )
        return HttpTestClient(self.server.httpd.server_address[1], self.graph_dir)

    def __exit__(self, *_: object) -> None:
        if self.server is not None:
            self.server.stop()
        if self.tempdir is not None:
            self.tempdir.cleanup()
