from __future__ import annotations

import http.client
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import torch
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder

import biology_server.server as server_module
from biology_server.attribution import (
    GraphLink,
    GraphResult,
    HookState,
    LayerFeatureData,
    LogitTarget,
    PreviewResult,
    SelectedFeature,
    TokenCandidate,
    backward_from_feature,
    build_source_groups,
    build_target_groups,
    collect_feature_scores,
    direct_feature_group_links,
    direct_feature_row_links,
    indirect_logit_influence,
    make_mlp_hook,
    prune_edges_by_thresholded_influence,
    prune_graph_by_indirect_influence,
    row_links,
    select_logit_targets,
)
from biology_server.server import serve
from circuit_graph_export import embedding_node_id, logit_node_id


class TinyTokenizer:
    def batch_decode(self, token_batches: list[list[int]]) -> list[str]:
        return [f"T{token_ids[0]}" for token_ids in token_batches]

    def decode(self, token_ids: list[int]) -> str:
        return f"T{token_ids[0]}"

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(token.removeprefix("T"))]


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
        use_chat_template: bool = True,
    ) -> PreviewResult:
        self.preview_calls.append(
            {"prompt": prompt, "slug": slug, "top_k": top_k, "use_chat_template": use_chat_template}
        )
        return PreviewResult(
            prompt=prompt,
            slug=slug or "fake-slug",
            use_chat_template=use_chat_template,
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
        node_threshold: float,
        edge_threshold: float,
        logit_prob_threshold: float,
        max_logit_nodes: int,
        graph_file_dir: Path | str | None = None,
        save_pt: str | None = None,
        use_chat_template: bool = True,
    ) -> GraphResult:
        if self.fail:
            raise RuntimeError("boom")
        self.generate_calls.append(
            {
                "prompt": prompt,
                "slug": slug,
                "target_token_id": target_token_id,
                "target_token": target_token,
                "node_threshold": node_threshold,
                "edge_threshold": edge_threshold,
                "logit_prob_threshold": logit_prob_threshold,
                "max_logit_nodes": max_logit_nodes,
                "save_pt": save_pt,
                "use_chat_template": use_chat_template,
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
        feature_dir = out_dir / "features" / "qwen3-4b-transcoders"
        feature_dir.mkdir(parents=True, exist_ok=True)
        (feature_dir / "1.json").write_text(json.dumps({"featureIndex": 1}), encoding="utf-8")
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
    def test_feature_score_collection_contracts_gradients_with_scaled_decoders(self) -> None:
        state = HookState(
            layers=[2],
            transcoders={},
            mlp_inputs={},
            feature_values={},
            layer_features={
                2: LayerFeatureData(
                    positions=torch.tensor([0, 1]),
                    feature_ids=torch.tensor([10, 11]),
                    activations=torch.tensor([2.0, 3.0]),
                    encoder_vectors=torch.zeros(2, 2),
                    decoder_vectors=torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
                    start=0,
                )
            },
            output_grads={2: torch.tensor([[[5.0, 7.0], [11.0, 13.0]]])},
        )

        scores = collect_feature_scores(state, 2)

        self.assertEqual(scores.tolist(), [10.0, 39.0])

    def test_feature_score_collection_can_limit_to_causal_source_layers(self) -> None:
        state = HookState(
            layers=[2, 12],
            transcoders={},
            mlp_inputs={},
            feature_values={},
            layer_features={
                2: LayerFeatureData(
                    positions=torch.tensor([0]),
                    feature_ids=torch.tensor([10]),
                    activations=torch.tensor([2.0]),
                    encoder_vectors=torch.zeros(1, 2),
                    decoder_vectors=torch.tensor([[2.0, 0.0]]),
                    start=0,
                ),
                12: LayerFeatureData(
                    positions=torch.tensor([0]),
                    feature_ids=torch.tensor([11]),
                    activations=torch.tensor([3.0]),
                    encoder_vectors=torch.zeros(1, 2),
                    decoder_vectors=torch.tensor([[0.0, 3.0]]),
                    start=1,
                ),
            },
            output_grads={2: torch.tensor([[[5.0, 7.0]]])},
        )

        scores = collect_feature_scores(state, 2, layers=[2])

        self.assertEqual(scores.tolist(), [10.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "layer 12"):
            collect_feature_scores(state, 2)

    def test_backward_from_feature_uses_direct_same_token_transcoder_formula(self) -> None:
        state = HookState(
            layers=[0, 2, 4],
            transcoders={},
            mlp_inputs={4: torch.zeros(1, 3, 2)},
            feature_values={},
            layer_features={
                0: LayerFeatureData(
                    positions=torch.tensor([1, 0]),
                    feature_ids=torch.tensor([10, 11]),
                    activations=torch.tensor([1.0, 1.0]),
                    encoder_vectors=torch.zeros(2, 2),
                    decoder_vectors=torch.tensor([[2.0, 4.0], [10.0, 20.0]]),
                    start=0,
                ),
                2: LayerFeatureData(
                    positions=torch.tensor([1]),
                    feature_ids=torch.tensor([12]),
                    activations=torch.tensor([1.0]),
                    encoder_vectors=torch.zeros(1, 2),
                    decoder_vectors=torch.tensor([[-1.0, 3.0]]),
                    start=2,
                ),
            },
            output_grads={},
        )
        downstream = SelectedFeature(
            layer=4,
            pos=1,
            feature=99,
            activation=1.0,
            logit_weight=0.0,
            clerp="target",
            encoder_vector=torch.tensor([5.0, 7.0]),
        )

        backward_from_feature(state, downstream)
        scores = collect_feature_scores(state, 3, layers=[0, 2])

        self.assertEqual(scores.tolist(), [38.0, 0.0, 16.0])

    def test_direct_feature_row_links_only_scan_same_token_upstream_sources(self) -> None:
        state = HookState(
            layers=[0, 2, 4],
            transcoders={},
            mlp_inputs={},
            feature_values={},
            layer_features={
                0: LayerFeatureData(
                    positions=torch.tensor([1, 0]),
                    feature_ids=torch.tensor([10, 11]),
                    activations=torch.tensor([1.0, 1.0]),
                    encoder_vectors=torch.zeros(2, 2),
                    decoder_vectors=torch.tensor([[2.0, 4.0], [10.0, 20.0]]),
                    start=0,
                ),
                2: LayerFeatureData(
                    positions=torch.tensor([1]),
                    feature_ids=torch.tensor([12]),
                    activations=torch.tensor([1.0]),
                    encoder_vectors=torch.zeros(1, 2),
                    decoder_vectors=torch.tensor([[-1.0, 3.0]]),
                    start=2,
                ),
            },
            output_grads={},
            token_vectors=torch.tensor([[0.0, 0.0], [7.0, 11.0], [0.0, 0.0]]),
        )
        selected = [
            SelectedFeature(0, 1, 10, 1.0, 0.0, "a", score_index=0),
            SelectedFeature(0, 0, 11, 1.0, 0.0, "b", score_index=1),
            SelectedFeature(2, 1, 12, 1.0, 0.0, "c", score_index=2),
        ]
        downstream = SelectedFeature(
            layer=4,
            pos=1,
            feature=99,
            activation=1.0,
            logit_weight=0.0,
            clerp="target",
            encoder_vector=torch.tensor([5.0, 7.0]),
        )

        links = direct_feature_row_links(
            downstream=downstream,
            source_groups=build_source_groups(selected=selected, state=state),
            state=state,
            input_token_ids=[101, 102, 103],
        )

        self.assertEqual(
            [(link.source, link.target, link.weight) for link in links],
            [
                ("0_10_1", "4_99_1", 38.0),
                ("2_12_1", "4_99_1", 16.0),
                ("E_102_1", "4_99_1", 112.0),
            ],
        )

    def test_direct_feature_group_links_matches_row_formula_for_target_blocks(self) -> None:
        state = HookState(
            layers=[0, 2, 4],
            transcoders={},
            mlp_inputs={},
            feature_values={},
            layer_features={
                0: LayerFeatureData(
                    positions=torch.tensor([1]),
                    feature_ids=torch.tensor([10]),
                    activations=torch.tensor([1.0]),
                    encoder_vectors=torch.zeros(1, 2),
                    decoder_vectors=torch.tensor([[2.0, 4.0]]),
                    start=0,
                ),
                2: LayerFeatureData(
                    positions=torch.tensor([1]),
                    feature_ids=torch.tensor([12]),
                    activations=torch.tensor([1.0]),
                    encoder_vectors=torch.zeros(1, 2),
                    decoder_vectors=torch.tensor([[-1.0, 3.0]]),
                    start=1,
                ),
            },
            output_grads={},
            token_vectors=torch.tensor([[0.0, 0.0], [7.0, 11.0]]),
        )
        sources = [
            SelectedFeature(0, 1, 10, 1.0, 0.0, "a", score_index=0),
            SelectedFeature(2, 1, 12, 1.0, 0.0, "b", score_index=1),
        ]
        targets = [
            SelectedFeature(4, 1, 99, 1.0, 0.0, "x", encoder_vector=torch.tensor([5.0, 7.0])),
            SelectedFeature(4, 1, 100, 1.0, 0.0, "y", encoder_vector=torch.tensor([1.0, 2.0])),
        ]

        links = direct_feature_group_links(
            target_layer=4,
            target_pos=1,
            target_group=next(iter(build_target_groups(selected=targets).values())),
            source_groups=build_source_groups(selected=sources, state=state),
            state=state,
            input_token_ids=[101, 102],
        )

        self.assertEqual(
            [(link.source, link.target, link.weight) for link in links],
            [
                ("0_10_1", "4_99_1", 38.0),
                ("0_10_1", "4_100_1", 10.0),
                ("2_12_1", "4_99_1", 16.0),
                ("2_12_1", "4_100_1", 5.0),
                ("E_102_1", "4_99_1", 112.0),
                ("E_102_1", "4_100_1", 29.0),
            ],
        )

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

    def test_mlp_hook_preserves_skip_reconstruction_value_and_skip_gradient(self) -> None:
        transcoder = SingleLayerTranscoder(
            d_model=2,
            d_transcoder=2,
            activation_function=torch.nn.ReLU(),
            layer_idx=0,
            skip_connection=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        with torch.no_grad():
            transcoder.W_enc.copy_(torch.eye(2))
            transcoder.W_dec.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
            transcoder.b_enc.zero_()
            transcoder.b_dec.copy_(torch.tensor([0.5, -0.5]))
            transcoder.W_skip.copy_(torch.tensor([[10.0, 0.0], [0.0, 20.0]]))

        state = HookState(
            layers=[0],
            transcoders={0: transcoder},
            mlp_inputs={},
            feature_values={},
            layer_features={},
            output_grads={},
        )
        mlp_input = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
        original_output = torch.zeros_like(mlp_input)
        replacement = make_mlp_hook(0, transcoder, state)(None, (mlp_input,), original_output)

        expected = transcoder.decode(transcoder.encode(mlp_input), mlp_input)
        self.assertTrue(torch.allclose(replacement, expected))

        replacement.sum().backward()
        self.assertTrue(torch.allclose(mlp_input.grad, torch.tensor([[[10.0, 20.0]]])))
        self.assertTrue(torch.allclose(state.output_grads[0], torch.ones_like(replacement)))

    def test_preview_and_background_graph_job(self) -> None:
        runner = FakeRunner()
        with run_test_server(runner) as client:
            preview = client.post(
                "/api/preview",
                {"prompt": "hello", "slug": "demo", "use_chat_template": False},
            )
            self.assertEqual(preview["slug"], "demo")
            self.assertFalse(preview["use_chat_template"])
            self.assertEqual(preview["target_token"]["id"], 2)

            job = client.post(
                "/api/graphs",
                {
                    "preview_id": preview["preview_id"],
                    "node_threshold": 0.7,
                    "edge_threshold": 0.9,
                    "logit_prob_threshold": 0.95,
                    "max_logit_nodes": 5,
                },
                expected_status=202,
            )
            finished = client.wait_for_job(job["job_id"])
            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["graph_url"], "/graph_data/demo.json")
            graph = client.get("/graph_data/demo.json")
            self.assertEqual(graph["metadata"]["slug"], "demo")
            self.assertFalse(runner.preview_calls[0]["use_chat_template"])
            self.assertFalse(runner.generate_calls[0]["use_chat_template"])
            self.assertEqual(runner.generate_calls[0]["node_threshold"], 0.7)
            self.assertEqual(runner.generate_calls[0]["edge_threshold"], 0.9)
            self.assertEqual(runner.generate_calls[0]["max_logit_nodes"], 5)

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

    def test_in_page_asset_routes_use_graph_dir(self) -> None:
        with run_test_server(FakeRunner()) as client:
            preview = client.post("/api/preview", {"prompt": "hello", "slug": "demo"})
            job = client.post(
                "/api/graphs", {"preview_id": preview["preview_id"]}, expected_status=202
            )
            client.wait_for_job(job["job_id"])

            metadata = client.get("/data/graph-metadata.json")
            graph = client.get("/graph_data/demo.json")
            feature_via_data = client.get("/data/features/qwen3-4b-transcoders/1.json")
            feature = client.get("/features/qwen3-4b-transcoders/1.json")

            self.assertEqual(metadata["graphs"][0]["slug"], "demo")
            self.assertEqual(graph["metadata"]["slug"], "demo")
            self.assertEqual(feature_via_data, {"featureIndex": 1})
            self.assertEqual(feature, {"featureIndex": 1})

    def test_page_and_ct_assets_served(self) -> None:
        with run_test_server(FakeRunner(), ct_assets={"util.js": "window.util = {}"}) as client:
            status, body = client.get_raw("/")
            self.assertEqual(status, 200)
            self.assertEqual(body, "ok")

            status, asset = client.get_raw("/ct/util.js")
            self.assertEqual(status, 200)
            self.assertEqual(asset, "window.util = {}")

            # App shell is uncached; vendored /ct/ assets stay cacheable.
            self.assertEqual(client.get_header("/", "Cache-Control"), "no-store")
            self.assertIsNone(client.get_header("/ct/util.js", "Cache-Control"))

    def test_removed_iframe_shims_return_404(self) -> None:
        with run_test_server(FakeRunner()) as client:
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
        with run_test_server(FakeRunner()) as client:
            preview = client.post("/api/preview", {"prompt": "hello", "slug": "demo"})
            job = client.post(
                "/api/graphs", {"preview_id": preview["preview_id"]}, expected_status=202
            )
            client.wait_for_job(job["job_id"])

            client.post("/save_graph/demo", {"qParams": {"clickedId": "changed"}})
            graph = client.get("/graph_data/demo.json")
            self.assertEqual(graph["qParams"], {"clickedId": "changed"})

    def test_graph_data_is_not_cached(self) -> None:
        with run_test_server(FakeRunner()) as client:
            preview = client.post("/api/preview", {"prompt": "hello", "slug": "demo"})
            job = client.post(
                "/api/graphs", {"preview_id": preview["preview_id"]}, expected_status=202
            )
            client.wait_for_job(job["job_id"])

            self.assertEqual(
                client.get_header("/graph_data/demo.json", "Cache-Control"), "no-store"
            )
            # Immutable feature files stay cacheable.
            self.assertIsNone(
                client.get_header("/data/features/qwen3-4b-transcoders/1.json", "Cache-Control")
            )

    def test_upload_graph_writes_graph_and_metadata(self) -> None:
        with run_test_server(FakeRunner()) as client:
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
        with run_test_server(FakeRunner()) as client:
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
        with run_test_server(FakeRunner()) as client:
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


class HttpTestClient:
    def __init__(self, port: int) -> None:
        self.port = port

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


class run_test_server:
    def __init__(self, runner: FakeRunner, *, ct_assets: dict[str, str] | None = None) -> None:
        self.runner = runner
        self.ct_assets = ct_assets or {}
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
        for rel_path, content in self.ct_assets.items():
            asset_path = frontend_dir / rel_path
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text(content, encoding="utf-8")
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
