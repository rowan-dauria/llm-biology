"""Reusable Qwen3-4B attribution runner for scripts and the local server.

Only uses circuit-tracer's per-layer ``SingleLayerTranscoder`` class, loaded via
``load_relu_transcoder``. The attribution, graph construction, feature labels,
and frontend export are custom project code.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder

try:
    from circuit_tracer.transcoder.single_layer_transcoder import load_relu_transcoder
except ImportError:
    from circuit_tracer.transcoder.single_layer_transcoder import (
        load_transcoder as load_relu_transcoder,
    )
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from circuit_graph_export import (
    FeatureNode,
    GraphLink,
    embedding_node_id,
    export_circuit_graph,
    feature_node_id,
    logit_node_id,
    make_feature_example_payload,
    paired_feature_index,
)
from feature_lookup.labels import FeatureLabelMap, get_feature_label, load_feature_labels
from feature_lookup.windows import active_prompt_ids, collect_prompt_texts, get_windows

MODEL_ID = "Qwen/Qwen3-4B"
TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"
NUM_LAYERS = 36
DEFAULT_LAYERS = [2, NUM_LAYERS // 3, (2 * NUM_LAYERS) // 3, NUM_LAYERS - 3]
DEFAULT_PROMPT = "The biological function of hemoglobin is to"
DEFAULT_MAX_FEATURE_NODES = 300
DEFAULT_EDGE_TOP_K = 20
DEFAULT_LOGITS_TOP_K = 10

CACHE_DIR = os.getenv("HF_HOME")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "ui_graphs"
DEFAULT_PT_DIR = PROJECT_ROOT / "data" / "attribution_graphs"
DEFAULT_WINDOW = 10


@dataclass(slots=True)
class HookState:
    features: dict[int, torch.Tensor]
    mlp_inputs: dict[int, torch.Tensor]
    embedding: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class SelectedFeature:
    layer: int
    pos: int
    feature: int
    activation: float
    logit_weight: float
    clerp: str

    @property
    def node_id(self) -> str:
        return feature_node_id(self.layer, self.feature, self.pos)


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    token_id: int
    token: str
    prob: float


@dataclass(frozen=True, slots=True)
class PreviewResult:
    prompt: str
    slug: str
    prompt_tokens: list[str]
    input_token_ids: list[int]
    target_token_id: int
    target_token_str: str
    target_token_prob: float
    top_tokens: list[TokenCandidate]


@dataclass(frozen=True, slots=True)
class GraphResult:
    prompt: str
    slug: str
    graph_path: Path
    target_token_id: int
    target_token_str: str
    target_token_prob: float
    prompt_tokens: list[str]
    input_token_ids: list[int]
    selected_features: list[SelectedFeature]
    links: list[GraphLink]
    pt_path: Path | None = None


def pick_device_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def timed(label: str):
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"[START] {label}")
            return self

        def __exit__(self, *_):
            print(f"[DONE]  {label} ({time.time() - self.start:.1f}s)")

    return Timer()


def parse_layers(raw: str) -> list[int]:
    layers = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not layers:
        raise ValueError("--layers must contain at least one layer")
    return layers


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip().lower()).strip("-")
    return slug or "qwen3-circuit"


def make_embedding_hook(state: HookState):
    def hook(_module, _inputs, output):
        embedding = output.detach().clone().requires_grad_(True)
        state.embedding = embedding
        return embedding

    return hook


def make_mlp_hook(
    layer_idx: int,
    transcoder: SingleLayerTranscoder,
    state: HookState,
):
    """Replace an MLP with transcoder features and capture differentiable leaves."""

    def hook(_module, inputs, output):
        mlp_input = inputs[0]
        state.mlp_inputs[layer_idx] = mlp_input
        with torch.no_grad():
            features_value = transcoder.encode(mlp_input)
        features = features_value.detach().clone().requires_grad_(True)
        state.features[layer_idx] = features
        reconstruction = transcoder.decode(features, mlp_input)
        return reconstruction.to(output.dtype)

    return hook


def zero_captured_grads(state: HookState) -> None:
    if state.embedding is not None:
        state.embedding.grad = None
    for features in state.features.values():
        features.grad = None


def load_transcoders(
    layers: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, SingleLayerTranscoder]:
    wanted_files = [f"layer_{layer}.safetensors" for layer in layers]
    with timed("Resolving transcoder paths"):
        transcoder_dir = Path(snapshot_download(TRANSCODER_REPO, allow_patterns=wanted_files))

    transcoders: dict[int, SingleLayerTranscoder] = {}
    for layer in layers:
        path = transcoder_dir / f"layer_{layer}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"No transcoder for layer {layer}: {path}")
        with timed(f"Loading transcoder layer {layer}"):
            transcoder = load_relu_transcoder(
                str(path),
                layer,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )
            for param in transcoder.parameters():
                param.requires_grad_(False)
            transcoders[layer] = transcoder
            print(
                f"  d_model={transcoder.d_model}  "
                f"d_transcoder={transcoder.d_transcoder}  "
                f"skip={transcoder.W_skip is not None}"
            )
    return transcoders


def select_target_token(
    tokenizer: PreTrainedTokenizerBase,
    last_logits: torch.Tensor,
    *,
    target_token_id: int | None = None,
    target_token: str | None = None,
) -> tuple[int, str, float]:
    if target_token_id is not None:
        selected_token_id = int(target_token_id)
    elif target_token is not None:
        token_ids = tokenizer.encode(target_token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                "--target-token must encode to exactly one token; "
                f"got {len(token_ids)} ids: {token_ids}"
            )
        selected_token_id = int(token_ids[0])
    else:
        selected_token_id = int(last_logits.argmax())

    probs = torch.softmax(last_logits.float(), dim=-1)
    return (
        selected_token_id,
        tokenizer.decode([selected_token_id]),
        float(probs[selected_token_id].item()),
    )


def top_token_candidates(
    tokenizer: PreTrainedTokenizerBase,
    last_logits: torch.Tensor,
    *,
    top_k: int,
) -> list[TokenCandidate]:
    if top_k <= 0:
        return []
    probs = torch.softmax(last_logits.float(), dim=-1)
    cap = min(top_k, probs.shape[-1])
    top_probs, top_idx = probs.topk(cap)
    probs_list = top_probs.detach().cpu().tolist()
    ids_list = [int(token_id) for token_id in top_idx.detach().cpu().tolist()]
    tokens = tokenizer.batch_decode([[tid] for tid in ids_list])
    return [
        TokenCandidate(token_id=tid, token=token, prob=float(prob))
        for prob, tid, token in zip(probs_list, ids_list, tokens, strict=True)
    ]


def select_seed_features(
    *,
    state: HookState,
    layers: list[int],
    labels: FeatureLabelMap,
    max_feature_nodes: int,
) -> list[SelectedFeature]:
    candidates: list[SelectedFeature] = []
    for layer in layers:
        features = state.features[layer]
        grads = features.grad
        if grads is None:
            print(f"[WARN] no logit gradient captured for layer {layer}")
            continue

        attribution = (features.detach() * grads).squeeze(0)
        flat_abs = attribution.abs().flatten()
        k = min(max_feature_nodes, flat_abs.numel())
        top_abs, top_flat_idx = flat_abs.topk(k)
        feature_dim = attribution.shape[-1]
        feature_values = features.detach().squeeze(0)

        for abs_value, flat_idx_tensor in zip(top_abs.tolist(), top_flat_idx.tolist(), strict=True):
            if abs_value <= 0:
                continue
            flat_idx = int(flat_idx_tensor)
            pos = flat_idx // feature_dim
            feature = flat_idx % feature_dim
            signed = float(attribution[pos, feature].item())
            activation = float(feature_values[pos, feature].item())
            candidates.append(
                SelectedFeature(
                    layer=layer,
                    pos=pos,
                    feature=feature,
                    activation=activation,
                    logit_weight=signed,
                    clerp=get_feature_label(labels, layer, feature),
                )
            )

    candidates.sort(key=lambda item: abs(item.logit_weight), reverse=True)
    selected = candidates[:max_feature_nodes]
    print(f"[INFO] selected {len(selected)} feature nodes")
    return selected


def build_seed_links(selected: list[SelectedFeature], logit_id: str) -> list[GraphLink]:
    return [
        GraphLink(source=feature.node_id, target=logit_id, weight=feature.logit_weight)
        for feature in selected
        if feature.logit_weight != 0.0
    ]


def build_circuit_links(
    *,
    selected: list[SelectedFeature],
    state: HookState,
    transcoders: dict[int, SingleLayerTranscoder],
    input_token_ids: list[int],
    edge_top_k: int,
) -> list[GraphLink]:
    """Compute selected feature/input effects on each selected downstream feature."""

    links: list[GraphLink] = []
    if edge_top_k <= 0:
        return links
    if state.embedding is None:
        raise RuntimeError("Embedding activations were not captured")

    for index, downstream in enumerate(selected, start=1):
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"[INFO] circuit edges {index}/{len(selected)}")

        zero_captured_grads(state)
        mlp_input = state.mlp_inputs[downstream.layer]
        encoder_vec = transcoders[downstream.layer].W_enc[downstream.feature]
        grad = torch.zeros_like(mlp_input)
        grad[0, downstream.pos] = encoder_vec.to(device=mlp_input.device, dtype=mlp_input.dtype)
        mlp_input.backward(gradient=grad, retain_graph=True)

        candidates: list[tuple[float, str, float]] = []
        for source in selected:
            if source == downstream or source.layer >= downstream.layer:
                continue
            features = state.features[source.layer]
            feature_grads = features.grad
            if feature_grads is None:
                continue
            weight = float(
                (
                    features.detach()[0, source.pos, source.feature]
                    * feature_grads[0, source.pos, source.feature]
                ).item()
            )
            if weight:
                candidates.append((abs(weight), source.node_id, weight))

        embedding_grads = state.embedding.grad
        if embedding_grads is not None:
            embedding_weights = (state.embedding.detach() * embedding_grads).sum(dim=-1)[0]
            for pos, vocab_idx in enumerate(input_token_ids):
                weight = float(embedding_weights[pos].item())
                if weight:
                    candidates.append((abs(weight), embedding_node_id(vocab_idx, pos), weight))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, source_id, weight in candidates[:edge_top_k]:
            links.append(GraphLink(source=source_id, target=downstream.node_id, weight=weight))

    return links


def feature_nodes_from_selected(selected: list[SelectedFeature]) -> list[FeatureNode]:
    return [
        FeatureNode(
            layer=feature.layer,
            pos=feature.pos,
            feature=feature.feature,
            activation=feature.activation,
            clerp=feature.clerp,
            influence=abs(feature.logit_weight),
        )
        for feature in selected
    ]


def compute_feature_logits(
    *,
    selected: list[SelectedFeature],
    transcoders: dict[int, SingleLayerTranscoder],
    unembed: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int = DEFAULT_LOGITS_TOP_K,
) -> dict[tuple[int, int], tuple[list[str], list[str]]]:
    """Project each selected feature's decoder vector through the unembedding.

    Ignores the final LayerNorm, so these are an approximation of the per-feature
    direct logit attribution — useful as a quick "what does this feature push?"
    summary in the UI, not a faithful contribution measure.
    """

    if top_k <= 0:
        return {}

    unique_by_layer: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for feature in selected:
        key = (feature.layer, feature.feature)
        if key in seen:
            continue
        seen.add(key)
        unique_by_layer[feature.layer].append(feature.feature)

    out: dict[tuple[int, int], tuple[list[str], list[str]]] = {}
    for layer, feature_ids in unique_by_layer.items():
        decoder = transcoders[layer].W_dec
        idx = torch.tensor(feature_ids, dtype=torch.long, device=decoder.device)
        rows = decoder.index_select(0, idx).to(unembed.dtype)
        logits = rows @ unembed.t()
        cap = min(top_k, logits.shape[-1])
        _, top_idx = logits.topk(cap, dim=-1)
        _, bot_idx = logits.topk(cap, dim=-1, largest=False)
        top_idx_cpu = top_idx.detach().cpu().tolist()
        bot_idx_cpu = bot_idx.detach().cpu().tolist()
        unique_ids = sorted({int(tid) for row in top_idx_cpu + bot_idx_cpu for tid in row})
        decoded = dict(
            zip(unique_ids, tokenizer.batch_decode([[tid] for tid in unique_ids]), strict=True)
        )
        for feature_id, top_row, bot_row in zip(feature_ids, top_idx_cpu, bot_idx_cpu, strict=True):
            top_tokens = [decoded[int(token_id)] for token_id in top_row]
            bot_tokens = [decoded[int(token_id)] for token_id in bot_row]
            out[(layer, feature_id)] = (top_tokens, bot_tokens)
    return out


def window_to_frontend_example(rendered: str, value: float) -> dict[str, Any]:
    if "<<" in rendered and ">>" in rendered:
        before, rest = rendered.split("<<", 1)
        target, after = rest.split(">>", 1)
        tokens = [before, target, after]
        token_acts = [0.0, float(value), 0.0]
        train_token_ind = 1
    else:
        tokens = [rendered]
        token_acts = [float(value)]
        train_token_ind = 0

    return {
        "tokens": tokens,
        "tokens_acts_list": token_acts,
        "train_token_ind": train_token_ind,
        "is_repeated_datapoint": False,
        "value": float(value),
    }


def build_feature_examples(
    *,
    selected: list[SelectedFeature],
    labels: FeatureLabelMap,
    tokenizer: PreTrainedTokenizerBase,
    feature_logits: dict[tuple[int, int], tuple[list[str], list[str]]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Build local feature-example files from saved top-K windows when available."""

    examples: dict[int, dict[str, Any]] = {}
    features_by_layer: dict[int, set[int]] = {}
    for feature in selected:
        features_by_layer.setdefault(feature.layer, set()).add(feature.feature)

    for layer, feature_ids in sorted(features_by_layer.items()):
        topk_path = PROJECT_ROOT / "data" / "feature_topk" / f"topk_layer_{layer}.pt"
        if not topk_path.exists():
            print(f"[WARN] missing top-K file for layer {layer}: {topk_path}")
            continue
        try:
            layer_data = torch.load(topk_path, weights_only=False, map_location="cpu")
            needed_prompt_ids: set[int] = set()
            for feature_id in feature_ids:
                needed_prompt_ids.update(active_prompt_ids(layer_data, feature_id))
            text_by_prompt_id = collect_prompt_texts(
                str(layer_data["corpus_spec"]),
                needed_prompt_ids,
            )
        except Exception as exc:
            print(f"[WARN] could not load top-K windows for layer {layer}: {exc}")
            continue

        for feature_id in sorted(feature_ids):
            windows = get_windows(
                layer_data,
                feature_id,
                tokenizer,
                window=DEFAULT_WINDOW,
                text_by_prompt_id=text_by_prompt_id,
            )
            active_windows = [window for window in windows if window.active]
            frontend_windows = [
                window_to_frontend_example(window.rendered, window.value)
                for window in active_windows
            ]
            paired = paired_feature_index(layer, feature_id)
            label = get_feature_label(labels, layer, feature_id)
            top_tokens, bot_tokens = (
                feature_logits.get((layer, feature_id), ([], []))
                if feature_logits is not None
                else ([], [])
            )
            examples[paired] = make_feature_example_payload(
                feature_index=paired,
                label=label,
                windows=frontend_windows,
                act_max=max((window.value for window in active_windows), default=1.0),
                top_logits=top_tokens,
                bottom_logits=bot_tokens,
            )

    print(f"[INFO] wrote feature examples for {len(examples)} features")
    return examples


def resolve_pt_path(save_pt: str | None, slug: str) -> Path | None:
    if save_pt is None:
        return None
    if save_pt == "auto":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return DEFAULT_PT_DIR / f"attribution_{slug}_{timestamp}.pt"
    return Path(save_pt)


class BiologyAttributionRunner:
    """Lazy-loaded, lock-serialized Qwen attribution runner."""

    def __init__(
        self,
        *,
        layers: list[int] | None = None,
        model_id: str = MODEL_ID,
        graph_file_dir: Path | str = DEFAULT_GRAPH_DIR,
        preview_top_k: int = 5,
    ) -> None:
        self.layers = list(layers or DEFAULT_LAYERS)
        self.model_id = model_id
        self.graph_file_dir = Path(graph_file_dir)
        self.preview_top_k = preview_top_k
        self._lock = threading.RLock()
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: Any | None = None
        self._transcoders: dict[int, SingleLayerTranscoder] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        device, dtype = pick_device_dtype()
        print(f"[INFO] device={device} dtype={dtype} layers={self.layers}")
        transcoders = load_transcoders(self.layers, device=device, dtype=dtype)

        with timed("Loading model"):
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model.to(device).eval()
            for param in model.parameters():
                param.requires_grad_(False)

        actual_layers = len(model.model.layers)
        if actual_layers != NUM_LAYERS:
            print(f"[WARN] expected {NUM_LAYERS} layers, model has {actual_layers}")

        self._device = device
        self._dtype = dtype
        self._tokenizer = tokenizer
        self._model = model
        self._transcoders = transcoders

    def _loaded(self) -> tuple[Any, PreTrainedTokenizerBase, dict[int, SingleLayerTranscoder]]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._transcoders is not None
        return self._model, self._tokenizer, self._transcoders

    def _inputs_for_prompt(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
    ) -> tuple[Any, list[int], list[str]]:
        if self._device is None:
            raise RuntimeError("runner device is not loaded")
        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        input_token_ids = [
            int(token_id) for token_id in inputs.input_ids[0].detach().cpu().tolist()
        ]
        prompt_tokens = tokenizer.batch_decode([[token_id] for token_id in input_token_ids])
        print(f"[INFO] prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
        return inputs, input_token_ids, prompt_tokens

    def _register_hooks(
        self,
        model: Any,
        transcoders: dict[int, SingleLayerTranscoder],
        state: HookState,
    ) -> list[Any]:
        handles = [model.model.embed_tokens.register_forward_hook(make_embedding_hook(state))]
        for layer in self.layers:
            handles.append(
                model.model.layers[layer].mlp.register_forward_hook(
                    make_mlp_hook(layer, transcoders[layer], state)
                )
            )
        return handles

    def preview(
        self,
        prompt: str,
        *,
        slug: str | None = None,
        top_k: int | None = None,
    ) -> PreviewResult:
        """Run a hooked forward pass and return the default next-token target."""

        with self._lock:
            model, tokenizer, transcoders = self._loaded()
            resolved_slug = slug or slugify(prompt[:50])
            inputs, input_token_ids, prompt_tokens = self._inputs_for_prompt(tokenizer, prompt)
            state = HookState(features={}, mlp_inputs={})
            handles = self._register_hooks(model, transcoders, state)
            try:
                with timed("Preview forward pass"):
                    outputs = model(**inputs)
                    last_logits = outputs.logits[0, -1]
                    target_token_id, target_token_str, target_token_prob = select_target_token(
                        tokenizer,
                        last_logits,
                    )
                    top_tokens = top_token_candidates(
                        tokenizer,
                        last_logits,
                        top_k=top_k if top_k is not None else self.preview_top_k,
                    )
                    print(
                        f"[INFO] preview target token id={target_token_id} "
                        f"({target_token_str!r}) p={target_token_prob:.4f}"
                    )
            finally:
                for handle in handles:
                    handle.remove()

            return PreviewResult(
                prompt=prompt,
                slug=resolved_slug,
                prompt_tokens=prompt_tokens,
                input_token_ids=input_token_ids,
                target_token_id=target_token_id,
                target_token_str=target_token_str,
                target_token_prob=target_token_prob,
                top_tokens=top_tokens,
            )

    def generate_graph(
        self,
        prompt: str,
        *,
        slug: str | None = None,
        target_token_id: int | None = None,
        target_token: str | None = None,
        max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES,
        edge_top_k: int = DEFAULT_EDGE_TOP_K,
        graph_file_dir: Path | str | None = None,
        save_pt: str | None = None,
    ) -> GraphResult:
        """Generate and export a circuit-tracer-compatible graph JSON."""

        with self._lock:
            model, tokenizer, transcoders = self._loaded()
            resolved_slug = slug or slugify(prompt[:50])
            labels = load_feature_labels(set(self.layers))
            print(f"[INFO] loaded {len(labels)} feature labels")
            inputs, input_token_ids, prompt_tokens = self._inputs_for_prompt(tokenizer, prompt)

            state = HookState(features={}, mlp_inputs={})
            handles = self._register_hooks(model, transcoders, state)
            try:
                with timed("Forward pass"):
                    outputs = model(**inputs)
                    last_logits = outputs.logits[0, -1]
                    selected_token_id, target_token_str, target_token_prob = select_target_token(
                        tokenizer,
                        last_logits,
                        target_token_id=target_token_id,
                        target_token=target_token,
                    )
                    target_logit = last_logits[selected_token_id]
                    print(
                        f"[INFO] target token id={selected_token_id} "
                        f"({target_token_str!r}) p={target_token_prob:.4f} "
                        f"logit={target_logit.item():.3f}"
                    )

                with timed("Feature-to-logit attribution"):
                    target_logit.backward(retain_graph=True)
                    selected = select_seed_features(
                        state=state,
                        layers=self.layers,
                        labels=labels,
                        max_feature_nodes=max_feature_nodes,
                    )

                logit_id = logit_node_id(NUM_LAYERS, selected_token_id, len(prompt_tokens) - 1)
                seed_links = build_seed_links(selected, logit_id)

                with timed("Selected circuit edge attribution"):
                    circuit_links = build_circuit_links(
                        selected=selected,
                        state=state,
                        transcoders=transcoders,
                        input_token_ids=input_token_ids,
                        edge_top_k=edge_top_k,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            all_links = seed_links + circuit_links
            with timed("Per-feature direct-logit projection"), torch.no_grad():
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is None:
                    raise RuntimeError("model has no output embeddings; cannot project logits")
                feature_logits = compute_feature_logits(
                    selected=selected,
                    transcoders=transcoders,
                    unembed=output_embeddings.weight,
                    tokenizer=tokenizer,
                )

            feature_examples = build_feature_examples(
                selected=selected,
                labels=labels,
                tokenizer=tokenizer,
                feature_logits=feature_logits,
            )
            output_dir = Path(graph_file_dir) if graph_file_dir is not None else self.graph_file_dir
            graph_path = export_circuit_graph(
                output_dir=output_dir,
                slug=resolved_slug,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                input_token_ids=input_token_ids,
                num_layers=NUM_LAYERS,
                feature_nodes=feature_nodes_from_selected(selected),
                links=all_links,
                target_token_id=selected_token_id,
                target_token_str=target_token_str,
                target_token_prob=target_token_prob,
                feature_examples=feature_examples,
            )

            print(f"[SAVE] graph JSON: {graph_path}")
            print(f"[INFO] feature nodes={len(selected)} links={len(all_links)}")

            pt_path = resolve_pt_path(save_pt, resolved_slug)
            if pt_path is not None:
                pt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "prompt": prompt,
                        "slug": resolved_slug,
                        "target_token_id": selected_token_id,
                        "target_token_str": target_token_str,
                        "target_token_prob": target_token_prob,
                        "layers": self.layers,
                        "tokens": prompt_tokens,
                        "input_token_ids": input_token_ids,
                        "selected_features": [asdict(feature) for feature in selected],
                        "links": [asdict(link) for link in all_links],
                        "graph_path": str(graph_path),
                    },
                    pt_path,
                )
                print(f"[SAVE] compact attribution PT: {pt_path}")

            return GraphResult(
                prompt=prompt,
                slug=resolved_slug,
                graph_path=graph_path,
                target_token_id=selected_token_id,
                target_token_str=target_token_str,
                target_token_prob=target_token_prob,
                prompt_tokens=prompt_tokens,
                input_token_ids=input_token_ids,
                selected_features=selected,
                links=all_links,
                pt_path=pt_path,
            )
