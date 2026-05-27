"""Reusable Qwen3-4B attribution runner for scripts and the local server.

Only uses circuit-tracer's per-layer ``SingleLayerTranscoder`` class, loaded via
``load_relu_transcoder``. The attribution, graph construction, feature labels,
and frontend export are custom project code.
"""

from __future__ import annotations

import gc
import os
import re
import threading
import time
import types
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from circuit_tracer.transcoder.single_layer_transcoder import (
    load_transcoder as load_relu_transcoder,
)
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, repeat_kv

from biology_server_t_lens.memory_profile import (
    memory_checkpoint,
    memory_profile_call,
    memory_scope,
)
from circuit_graph_export import (
    FeatureNode,
    GraphLink,
    LogitNode,
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
DEFAULT_NODE_THRESHOLD = 0.8
DEFAULT_EDGE_THRESHOLD = 0.98
DEFAULT_LOGIT_PROB_THRESHOLD = 0.95
DEFAULT_MAX_LOGIT_NODES = 4
DEFAULT_BATCH_SIZE = 128
DEFAULT_UPDATE_INTERVAL = 1

CACHE_DIR = os.getenv("HF_HOME")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "ui_graphs"
DEFAULT_PT_DIR = PROJECT_ROOT / "data" / "attribution_graphs"
DEFAULT_TOPK_DIR = PROJECT_ROOT / "data" / "feature_topk" / "150k-pile"
DEFAULT_WINDOW = 10


@dataclass(slots=True)
class HookState:
    layers: list[int]
    transcoders: dict[int, SingleLayerTranscoder]
    mlp_inputs: dict[int, torch.Tensor]
    feature_values: dict[int, torch.Tensor]
    layer_features: dict[int, LayerFeatureData]
    output_grads: dict[int, torch.Tensor]
    embedding: torch.Tensor | None = None
    token_vectors: torch.Tensor | None = None
    embedding_grad: torch.Tensor | None = None
    final_hidden: torch.Tensor | None = None

    def clear_grads(self) -> None:
        self.output_grads.clear()
        self.embedding_grad = None
        if self.embedding is not None:
            self.embedding.grad = None


@dataclass(frozen=True, slots=True)
class LayerFeatureData:
    positions: torch.Tensor
    feature_ids: torch.Tensor
    activations: torch.Tensor
    encoder_vectors: torch.Tensor
    decoder_vectors: torch.Tensor
    start: int = 0

    @property
    def end(self) -> int:
        return self.start + int(self.feature_ids.numel())


@dataclass(frozen=True, slots=True)
class ActiveFeature:
    layer: int
    pos: int
    feature: int
    activation: float
    encoder_vector: torch.Tensor
    score_index: int

    @property
    def node_id(self) -> str:
        return feature_node_id(self.layer, self.feature, self.pos)


@dataclass(frozen=True, slots=True)
class SelectedFeature:
    layer: int
    pos: int
    feature: int
    activation: float
    logit_weight: float
    clerp: str
    encoder_vector: torch.Tensor | None = None
    score_index: int | None = None
    influence: float | None = None

    @property
    def node_id(self) -> str:
        return feature_node_id(self.layer, self.feature, self.pos)


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    token_id: int
    token: str
    prob: float


@dataclass(frozen=True, slots=True)
class LogitTarget:
    token_id: int
    token: str
    prob: float
    node_id: str


@dataclass(frozen=True, slots=True)
class LogitAttributionRow:
    target: LogitTarget
    feature_scores: torch.Tensor
    embedding_scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreviewResult:
    prompt: str
    slug: str
    use_chat_template: bool
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
    logit_targets: list[LogitTarget] = field(default_factory=list)
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
        state.token_vectors = embedding.detach().squeeze(0)
        embedding.register_hook(lambda grad: setattr(state, "embedding_grad", grad.detach()))
        return embedding

    return hook


def make_final_hidden_hook(state: HookState):
    def hook(_module, _inputs, output):
        state.final_hidden = output
        return output

    return hook


_FREEZE_FLAG_ATTR = "_attribution_freezes_installed"
_DETACHED_EAGER_KEY = "detached_eager"


def _detached_eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    """Eager attention with attention pattern detached.

    Mirrors transformers.models.qwen3.modeling_qwen3.eager_attention_forward but
    treats the softmaxed attention pattern as a constant. Gradients still flow
    through value_states (and thus W_V, W_O), giving the linearised OV-only
    attention Jacobian used by the attribution-graphs methodology.
    """

    del dropout  # never apply attention dropout during attribution
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = attn_weights.detach()
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _patched_rms_norm_forward(self, hidden_states):
    """RMSNorm forward with the 1/RMS scale detached.

    Forward output is numerically identical to the upstream Qwen3RMSNorm, but
    the rsqrt(variance + eps) factor is treated as an input-independent
    constant so backward only sees a linear projection through self.weight.
    """

    input_dtype = hidden_states.dtype
    hidden_states_f = hidden_states.to(torch.float32)
    variance = hidden_states_f.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + self.variance_epsilon).detach()
    hidden_states_f = hidden_states_f * scale
    return self.weight * hidden_states_f.to(input_dtype)


def install_freezes(model: Any) -> None:
    """Linearise the model for attribution: freeze attention pattern + RMSNorm scale.

    Replaces each Qwen3 attention layer's eager implementation with one that
    detaches the softmaxed pattern, and rewrites every Qwen3RMSNorm.forward so
    that its 1/RMS scale is also detached. Forward numerics are unchanged; only
    backward gradient flow is altered so that the model is exactly linear in
    the residual stream (as in the Anthropic local replacement model).
    """

    if getattr(model, _FREEZE_FLAG_ATTR, False):
        return

    # .register() writes to the class-level _global_mapping. transformers'
    # _preprocess_mask_arguments checks _global_mapping directly to decide
    # whether to skip mask construction, so an instance-level __setitem__
    # (which only updates _local_mapping) would cause create_causal_mask to
    # early-exit with None and break the forward pass numerics.
    ALL_ATTENTION_FUNCTIONS.register(_DETACHED_EAGER_KEY, _detached_eager_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(_DETACHED_EAGER_KEY, eager_mask)

    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = _DETACHED_EAGER_KEY
    for submodule in model.modules():
        sub_config = getattr(submodule, "config", None)
        if sub_config is not None and hasattr(sub_config, "_attn_implementation"):
            sub_config._attn_implementation = _DETACHED_EAGER_KEY

    for submodule in model.modules():
        if isinstance(submodule, Qwen3RMSNorm):
            submodule.forward = types.MethodType(_patched_rms_norm_forward, submodule)

    setattr(model, _FREEZE_FLAG_ATTR, True)


def make_mlp_hook(layer_idx: int, transcoder: SingleLayerTranscoder, state: HookState):
    """Replace an MLP and expose source vectors for attribution rows."""

    def hook(_module, inputs, output):
        mlp_input = inputs[0]
        state.mlp_inputs[layer_idx] = mlp_input
        with torch.no_grad():
            features = transcoder.encode(mlp_input).detach()
            layer_data = layer_feature_data(transcoder, features.squeeze(0))
        state.feature_values[layer_idx] = features
        state.layer_features[layer_idx] = layer_data

        feature_reconstruction = transcoder.decode(
            features.to(transcoder.W_dec.dtype),
            mlp_input if transcoder.W_skip is not None else None,
        ).to(output.dtype)
        if transcoder.W_skip is not None:
            skip = transcoder.compute_skip(mlp_input)
            replacement = skip + (feature_reconstruction - skip).detach()
        else:
            replacement = feature_reconstruction.detach().requires_grad_(True)
        replacement.register_hook(
            lambda grad, layer=layer_idx: state.output_grads.__setitem__(layer, grad.detach())
        )
        return replacement

    return hook


def layer_feature_data(
    transcoder: SingleLayerTranscoder,
    features: torch.Tensor,
) -> LayerFeatureData:
    sparse = features.to_sparse().coalesce()
    if sparse._nnz() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=features.device)
        empty_vec = torch.empty(0, transcoder.d_model, dtype=features.dtype, device=features.device)
        return LayerFeatureData(
            positions=empty_long,
            feature_ids=empty_long,
            activations=torch.empty(0, dtype=features.dtype, device=features.device),
            encoder_vectors=empty_vec,
            decoder_vectors=empty_vec,
        )

    positions, feature_ids = sparse.indices()
    activations = sparse.values()
    encoder_vectors = transcoder.W_enc.index_select(0, feature_ids).to(features.dtype)
    decoder_vectors = transcoder.W_dec.index_select(0, feature_ids).to(features.dtype)
    decoder_vectors = decoder_vectors * activations[:, None]
    return LayerFeatureData(
        positions=positions,
        feature_ids=feature_ids,
        activations=activations,
        encoder_vectors=encoder_vectors,
        decoder_vectors=decoder_vectors,
    )


def finalize_active_features(state: HookState) -> list[ActiveFeature]:
    active: list[ActiveFeature] = []
    offset = 0
    for layer in state.layers:
        data = state.layer_features[layer]
        state.layer_features[layer] = LayerFeatureData(
            positions=data.positions,
            feature_ids=data.feature_ids,
            activations=data.activations,
            encoder_vectors=data.encoder_vectors,
            decoder_vectors=data.decoder_vectors,
            start=offset,
        )
        for local_idx in range(data.feature_ids.numel()):
            active.append(
                ActiveFeature(
                    layer=layer,
                    pos=int(data.positions[local_idx].item()),
                    feature=int(data.feature_ids[local_idx].item()),
                    activation=float(data.activations[local_idx].item()),
                    encoder_vector=data.encoder_vectors[local_idx],
                    score_index=offset + local_idx,
                )
            )
        offset += int(data.feature_ids.numel())
    return active


def collect_feature_scores(
    state: HookState,
    n_features: int,
    *,
    layers: list[int] | None = None,
) -> torch.Tensor:
    scores: torch.Tensor | None = None
    score_layers = state.layers if layers is None else layers
    for layer in score_layers:
        data = state.layer_features[layer]
        grad = state.output_grads.get(layer)
        if grad is None:
            raise RuntimeError(f"output gradient not captured for layer {layer}")
        if data.feature_ids.numel() == 0:
            continue
        if scores is None:
            scores = torch.zeros(n_features, dtype=data.decoder_vectors.dtype, device=grad.device)
        assert scores is not None
        grad_at_positions = grad[0].index_select(0, data.positions.to(grad.device))
        layer_scores = (
            grad_at_positions.to(data.decoder_vectors.dtype) * data.decoder_vectors
        ).sum(dim=-1)
        scores[data.start : data.end] = layer_scores
    if scores is None:
        device = state.embedding.device if state.embedding is not None else torch.device("cpu")
        return torch.zeros(n_features, device=device)
    return scores


def collect_embedding_scores(state: HookState) -> torch.Tensor:
    if state.embedding_grad is None or state.token_vectors is None:
        raise RuntimeError("Embedding gradients were not captured")
    return (state.embedding_grad[0].to(state.token_vectors.dtype) * state.token_vectors).sum(dim=-1)


def demeaned_unembed_vector(
    unembed: torch.Tensor, token_id: int, dtype: torch.dtype
) -> torch.Tensor:
    row = unembed[token_id]
    return (row - unembed.mean(dim=0)).to(dtype)


def backward_from_final_hidden(state: HookState, vector: torch.Tensor) -> None:
    if state.final_hidden is None:
        raise RuntimeError("Final hidden states were not captured")
    state.clear_grads()
    gradient = torch.zeros_like(state.final_hidden)
    gradient[0, -1] = vector.to(device=gradient.device, dtype=gradient.dtype)
    state.final_hidden.backward(gradient=gradient, retain_graph=True)


def backward_from_feature(state: HookState, feature: SelectedFeature) -> None:
    """Backprop a downstream feature's encoder vector through the frozen model.

    With install_freezes applied, attention patterns, RMSNorm scales, and the
    non-linear part of every MLP write are detached. The model is therefore
    exactly linear in the residual stream, so injecting ``f_enc`` at
    ``(feature.layer, feature.pos)`` and calling ``mlp_input.backward(...)``
    produces residual-stream gradients whose dot product with upstream decoder
    vectors equals the direct effect described in Anthropic's Methods paper
    (eqns 7 and 8): same-token transcoder→transcoder edges plus cross-token
    edges mediated by frozen attention OV circuits.
    """

    if feature.encoder_vector is None:
        raise RuntimeError("Selected feature is missing its encoder vector")
    state.clear_grads()

    mlp_input = state.mlp_inputs[feature.layer]
    gradient = torch.zeros_like(mlp_input)
    gradient[0, feature.pos] = feature.encoder_vector.to(
        device=gradient.device, dtype=gradient.dtype
    )
    mlp_input.backward(gradient=gradient, retain_graph=True)


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


def select_logit_targets(
    tokenizer: PreTrainedTokenizerBase,
    last_logits: torch.Tensor,
    *,
    pos: int,
    prob_threshold: float = DEFAULT_LOGIT_PROB_THRESHOLD,
    max_logit_nodes: int = DEFAULT_MAX_LOGIT_NODES,
    target_token_id: int | None = None,
    target_token: str | None = None,
) -> list[LogitTarget]:
    """Select logit nodes using the graph-pruning probability-mass rule.

    By default this keeps the smallest top-probability prefix whose cumulative
    mass reaches ``prob_threshold``, capped at ``max_logit_nodes``. Explicit
    target-token arguments preserve the old CLI behaviour by requesting one
    particular output logit.
    """

    if max_logit_nodes <= 0:
        raise ValueError("max_logit_nodes must be positive")

    probs = torch.softmax(last_logits.float(), dim=-1)
    if target_token_id is not None or target_token is not None:
        selected_token_id, selected_token, selected_prob = select_target_token(
            tokenizer,
            last_logits,
            target_token_id=target_token_id,
            target_token=target_token,
        )
        return [
            LogitTarget(
                token_id=selected_token_id,
                token=selected_token,
                prob=selected_prob,
                node_id=logit_node_id(NUM_LAYERS, selected_token_id, pos),
            )
        ]

    cap = min(max_logit_nodes, probs.shape[-1])
    top_probs, top_idx = probs.topk(cap)
    probs_list = [float(prob) for prob in top_probs.detach().cpu().tolist()]
    ids_list = [int(token_id) for token_id in top_idx.detach().cpu().tolist()]
    tokens = tokenizer.batch_decode([[tid] for tid in ids_list])

    targets: list[LogitTarget] = []
    cumulative_prob = 0.0
    for token_id, token, prob in zip(ids_list, tokens, probs_list, strict=True):
        targets.append(
            LogitTarget(
                token_id=token_id,
                token=token,
                prob=prob,
                node_id=logit_node_id(NUM_LAYERS, token_id, pos),
            )
        )
        cumulative_prob += prob
        if cumulative_prob >= prob_threshold:
            break
    return targets


def selected_features_from_active(
    *,
    active_features: list[ActiveFeature],
    labels: FeatureLabelMap,
    logit_weights: dict[int, float] | None = None,
) -> list[SelectedFeature]:
    logit_weights = logit_weights or {}
    return [
        SelectedFeature(
            layer=active.layer,
            pos=active.pos,
            feature=active.feature,
            activation=active.activation,
            logit_weight=float(logit_weights.get(active.score_index, 0.0)),
            clerp=get_feature_label(labels, active.layer, active.feature),
            encoder_vector=active.encoder_vector,
            score_index=active.score_index,
        )
        for active in active_features
    ]


def logit_weights_from_dense_matrix(
    *,
    dense_edge_matrix: torch.Tensor,
    selected: list[ActiveFeature],
    logit_targets: list[LogitTarget],
    n_pos: int,
) -> dict[int, float]:
    """Signed direct logit contribution per selected feature score index."""

    weights: dict[int, float] = {}
    n_selected = len(selected)
    for col, feature in enumerate(selected):
        total = 0.0
        for logit_idx, target in enumerate(logit_targets):
            row = n_selected + n_pos + logit_idx
            total += float(target.prob) * float(dense_edge_matrix[row, col].item())
        weights[feature.score_index] = total
    return weights


def dense_edge_matrix_links(
    *,
    selected: list[SelectedFeature],
    input_token_ids: list[int],
    logit_targets: list[LogitTarget],
    dense_edge_matrix: torch.Tensor,
) -> list[GraphLink]:
    """Translate a packed ``A[target, source]`` matrix into signed links."""

    node_ids = [feature.node_id for feature in selected]
    node_ids.extend(
        embedding_node_id(vocab_idx, pos) for pos, vocab_idx in enumerate(input_token_ids)
    )
    node_ids.extend(target.node_id for target in logit_targets)

    if dense_edge_matrix.shape != (len(node_ids), len(node_ids)):
        raise ValueError(
            "dense_edge_matrix shape does not match packed node order: "
            f"{tuple(dense_edge_matrix.shape)} vs {(len(node_ids), len(node_ids))}"
        )

    links: list[GraphLink] = []
    matrix = dense_edge_matrix.detach().cpu()
    for target_idx, target_id in enumerate(node_ids):
        row = matrix[target_idx]
        source_indices = row.nonzero(as_tuple=False).flatten().tolist()
        for source_idx in source_indices:
            if source_idx == target_idx:
                continue
            weight = float(row[source_idx].item())
            if weight:
                links.append(
                    GraphLink(
                        source=node_ids[source_idx],
                        target=target_id,
                        weight=weight,
                    )
                )
    return links


def row_links(
    *,
    target_id: str,
    feature_scores: torch.Tensor,
    embedding_scores: torch.Tensor,
    selected: list[SelectedFeature],
    input_token_ids: list[int],
    edge_top_k: int,
) -> list[GraphLink]:
    candidates: list[tuple[float, str, float]] = []
    for feature in selected:
        if feature.score_index is None:
            continue
        weight = float(feature_scores[feature.score_index].detach().cpu().item())
        if weight and feature.node_id != target_id:
            candidates.append((abs(weight), feature.node_id, weight))
    for pos, vocab_idx in enumerate(input_token_ids):
        weight = float(embedding_scores[pos].detach().cpu().item())
        if weight:
            candidates.append((abs(weight), embedding_node_id(vocab_idx, pos), weight))

    if edge_top_k <= 0:
        return [
            GraphLink(source=source_id, target=target_id, weight=weight)
            for _, source_id, weight in candidates
        ]

    candidates.sort(key=lambda item: item[0], reverse=True)
    limit = min(edge_top_k, len(candidates))
    return [
        GraphLink(source=source_id, target=target_id, weight=weight)
        for _, source_id, weight in candidates[:limit]
    ]


def build_full_attribution_links(
    *,
    selected: list[SelectedFeature],
    state: HookState,
    active_count: int,
    logit_rows: list[LogitAttributionRow],
    input_token_ids: list[int],
) -> list[GraphLink]:
    """Build unpruned attribution rows for every selected feature and logit.

    Logit rows reuse the gradients captured during backward_from_final_hidden.
    For each selected feature we run a fresh backward_from_feature pass through
    the frozen model so that the residual-stream gradients pick up both the
    same-token transcoder→transcoder direct path (eqn 7) and cross-token
    contributions mediated by the frozen attention OV circuits (eqn 8).
    """

    links: list[GraphLink] = []
    for row in logit_rows:
        links.extend(
            row_links(
                target_id=row.target.node_id,
                feature_scores=row.feature_scores,
                embedding_scores=row.embedding_scores,
                selected=selected,
                input_token_ids=input_token_ids,
                edge_top_k=0,
            )
        )

    for index, downstream in enumerate(selected, start=1):
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"[INFO] full circuit rows {index}/{len(selected)}")

        backward_from_feature(state, downstream)

        source_layers = [layer for layer in state.layers if layer < downstream.layer]
        feature_scores = collect_feature_scores(state, active_count, layers=source_layers)
        embedding_scores = collect_embedding_scores(state)

        links.extend(
            row_links(
                target_id=downstream.node_id,
                feature_scores=feature_scores,
                embedding_scores=embedding_scores,
                selected=selected,
                input_token_ids=input_token_ids,
                edge_top_k=0,
            )
        )
    return links


def normalized_edge_weights(
    *,
    links: list[GraphLink],
    node_ids: set[str],
) -> dict[int, float]:
    row_sums: defaultdict[str, float] = defaultdict(float)
    for link in links:
        if link.source not in node_ids or link.target not in node_ids:
            continue
        row_sums[link.target] += abs(link.weight)

    normalized: dict[int, float] = {}
    for idx, link in enumerate(links):
        if link.source not in node_ids or link.target not in node_ids:
            continue
        total = max(row_sums[link.target], 1e-8)
        normalized[idx] = abs(link.weight) / total
    return normalized


def indirect_logit_influence(
    *,
    node_ids: set[str],
    links: list[GraphLink],
    logit_weights: dict[str, float],
) -> dict[str, float]:
    """Compute weighted logit-row path sums for the normalized adjacency.

    With ``A[target, source]`` this returns, for each source node, the weighted
    average over logit rows of ``A + A^2 + ...``. The current graph is acyclic,
    so this sparse reverse dynamic program is exact without forming a dense
    inverse.
    """

    normalized = normalized_edge_weights(links=links, node_ids=node_ids)
    outgoing: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for idx, weight in normalized.items():
        link = links[idx]
        outgoing[link.source].append((link.target, weight))

    cache: dict[str, float] = {}
    visiting: set[str] = set()

    def score(node_id: str) -> float:
        if node_id in cache:
            return cache[node_id]
        if node_id in visiting:
            raise ValueError("attribution graph contains a cycle")
        visiting.add(node_id)
        total = 0.0
        for target_id, edge_weight in outgoing.get(node_id, []):
            total += edge_weight * (logit_weights.get(target_id, 0.0) + score(target_id))
        visiting.remove(node_id)
        cache[node_id] = total
        return total

    return {node_id: score(node_id) for node_id in node_ids}


def cumulative_scores(
    *,
    scores: dict[str, float],
    candidate_ids: list[str],
) -> dict[str, float]:
    indexed_scores = [
        (idx, node_id, max(float(scores.get(node_id, 0.0)), 0.0))
        for idx, node_id in enumerate(candidate_ids)
    ]
    positive = [(idx, node_id, score) for idx, node_id, score in indexed_scores if score > 0]
    total = sum(score for _, _, score in positive)
    if total <= 0:
        return {}

    cumulative = 0.0
    out: dict[str, float] = {}
    for _, node_id, score in sorted(positive, key=lambda item: (-item[2], item[0])):
        cumulative += score / total
        out[node_id] = min(cumulative, 1.0)
    return out


def keep_feature_ids_by_threshold(
    *,
    cumulative_by_id: dict[str, float],
    feature_ids: list[str],
    threshold: float,
) -> set[str]:
    kept = {
        node_id
        for node_id in feature_ids
        if node_id in cumulative_by_id and cumulative_by_id[node_id] <= threshold
    }
    if kept or not cumulative_by_id:
        return kept

    best_id = min(cumulative_by_id, key=cumulative_by_id.__getitem__)
    return {best_id}


def prune_edges_by_thresholded_influence(
    *,
    links: list[GraphLink],
    node_ids: set[str],
    logit_weights: dict[str, float],
    threshold: float,
) -> list[GraphLink]:
    normalized = normalized_edge_weights(links=links, node_ids=node_ids)
    node_scores = indirect_logit_influence(
        node_ids=node_ids,
        links=links,
        logit_weights=logit_weights,
    )
    for node_id, weight in logit_weights.items():
        if node_id in node_ids:
            node_scores[node_id] = weight

    edge_scores = [
        (idx, normalized_weight * max(node_scores.get(links[idx].target, 0.0), 0.0))
        for idx, normalized_weight in normalized.items()
    ]
    positive = [(idx, score) for idx, score in edge_scores if score > 0]
    total = sum(score for _, score in positive)
    if total <= 0:
        return []

    cumulative = 0.0
    cutoff_score = 0.0
    for _idx, score in sorted(positive, key=lambda item: item[1], reverse=True):
        cumulative += score / total
        cutoff_score = score
        if cumulative >= threshold:
            break

    kept_indices = {idx for idx, score in positive if score >= cutoff_score}
    return [link for idx, link in enumerate(links) if idx in kept_indices]


def prune_graph_by_indirect_influence(
    *,
    selected: list[SelectedFeature],
    input_token_ids: list[int],
    links: list[GraphLink],
    logit_targets: list[LogitTarget],
    node_threshold: float,
    edge_threshold: float,
) -> tuple[list[SelectedFeature], list[GraphLink]]:
    feature_ids = [feature.node_id for feature in selected]
    embedding_ids = [
        embedding_node_id(vocab_idx, pos) for pos, vocab_idx in enumerate(input_token_ids)
    ]
    logit_weights = {target.node_id: target.prob for target in logit_targets}
    all_node_ids = set(feature_ids) | set(embedding_ids) | set(logit_weights)

    node_scores = indirect_logit_influence(
        node_ids=all_node_ids,
        links=links,
        logit_weights=logit_weights,
    )
    cumulative_by_id = cumulative_scores(scores=node_scores, candidate_ids=feature_ids)
    kept_feature_ids = keep_feature_ids_by_threshold(
        cumulative_by_id=cumulative_by_id,
        feature_ids=feature_ids,
        threshold=node_threshold,
    )
    kept_node_ids = set(embedding_ids) | set(logit_weights) | kept_feature_ids

    node_pruned_links = [
        link for link in links if link.source in kept_node_ids and link.target in kept_node_ids
    ]
    edge_pruned_links = prune_edges_by_thresholded_influence(
        links=node_pruned_links,
        node_ids=kept_node_ids,
        logit_weights=logit_weights,
        threshold=edge_threshold,
    )

    by_id = {feature.node_id: feature for feature in selected}
    ordered_kept_ids = sorted(
        kept_feature_ids,
        key=lambda node_id: cumulative_by_id.get(node_id, 1.0),
    )
    pruned_features = [
        replace(
            by_id[node_id],
            influence=min(cumulative_by_id.get(node_id, node_threshold), node_threshold),
        )
        for node_id in ordered_kept_ids
    ]
    print(
        "[INFO] pruned graph "
        f"features={len(pruned_features)}/{len(selected)} "
        f"links={len(edge_pruned_links)}/{len(links)}"
    )
    return pruned_features, edge_pruned_links


def feature_nodes_from_selected(selected: list[SelectedFeature]) -> list[FeatureNode]:
    return [
        FeatureNode(
            layer=feature.layer,
            pos=feature.pos,
            feature=feature.feature,
            activation=feature.activation,
            clerp=feature.clerp,
            influence=feature.influence
            if feature.influence is not None
            else abs(feature.logit_weight),
        )
        for feature in selected
    ]


def selected_feature_to_dict(feature: SelectedFeature) -> dict[str, Any]:
    return {
        "layer": feature.layer,
        "pos": feature.pos,
        "feature": feature.feature,
        "activation": feature.activation,
        "logit_weight": feature.logit_weight,
        "clerp": feature.clerp,
        "score_index": feature.score_index,
        "influence": feature.influence,
    }


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
        logits = rows @ unembed if unembed.shape[0] == rows.shape[-1] else rows @ unembed.t()
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
    topk_dir: Path,
    feature_logits: dict[tuple[int, int], tuple[list[str], list[str]]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Build local feature-example files from saved top-K windows.

    ``topk_dir`` is required and must contain ``topk_layer_<L>.pt`` for every
    selected layer; a missing file raises rather than silently producing empty
    feature panels.
    """

    examples: dict[int, dict[str, Any]] = {}
    features_by_layer: dict[int, set[int]] = {}
    for feature in selected:
        features_by_layer.setdefault(feature.layer, set()).add(feature.feature)

    for layer, feature_ids in sorted(features_by_layer.items()):
        topk_path = topk_dir / f"topk_layer_{layer}.pt"
        if not topk_path.exists():
            raise FileNotFoundError(f"missing top-K file for layer {layer}: {topk_path}")
        try:
            layer_data = torch.load(topk_path, weights_only=False, map_location="cpu")
            needed_prompt_ids: set[int] = set()
            for feature_id in feature_ids:
                needed_prompt_ids.update(active_prompt_ids(layer_data, feature_id))
            text_by_prompt_id = collect_prompt_texts(
                str(layer_data["corpus_spec"]),
                needed_prompt_ids,
                int(layer_data.get("num_parts", 1)),
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
    """Lazy-loaded, lock-serialized TransformerLens Qwen attribution runner."""

    def __init__(
        self,
        *,
        layers: list[int] | None = None,
        model_id: str = MODEL_ID,
        graph_file_dir: Path | str = DEFAULT_GRAPH_DIR,
        topk_dir: Path | str = DEFAULT_TOPK_DIR,
        preview_top_k: int = DEFAULT_LOGITS_TOP_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_feature_nodes: int = DEFAULT_MAX_FEATURE_NODES,
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
    ) -> None:
        self.layers = list(layers or DEFAULT_LAYERS)
        self.model_id = model_id
        self.graph_file_dir = Path(graph_file_dir)
        self.topk_dir = Path(topk_dir)
        self.preview_top_k = preview_top_k
        self.batch_size = batch_size
        self.max_feature_nodes = max_feature_nodes
        self.update_interval = update_interval
        self._lock = threading.RLock()
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._preview_model: Any | None = None
        self._model: Any | None = None
        self._transcoders: dict[int, SingleLayerTranscoder] | None = None

    def _ensure_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        if self._device is None or self._dtype is None:
            self._device, self._dtype = pick_device_dtype()
        return self._device, self._dtype

    def _ensure_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = self._tokenizer
        if tokenizer is not None:
            return tokenizer

        with memory_scope("tokenizer:from_pretrained"):
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        memory_checkpoint("tokenizer:ready")
        return tokenizer

    def _empty_device_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def _release_preview_model(self) -> None:
        if self._preview_model is None:
            return
        memory_checkpoint("preview_model:before release")
        self._preview_model = None
        gc.collect()
        self._empty_device_cache()
        memory_checkpoint("preview_model:after release")

    def _ensure_loaded(self, *, include_transcoders: bool = True) -> None:
        memory_checkpoint(f"ensure_loaded:start include_transcoders={include_transcoders}")
        device, dtype = self._ensure_device_dtype()
        print(f"[INFO] device={device} dtype={dtype} layers={self.layers}")
        memory_checkpoint("ensure_loaded:after device dtype")

        if self._model is None:
            from biology_server_t_lens.tl_model import load_replacement_model

            self._release_preview_model()
            tokenizer = self._ensure_tokenizer()
            with timed("Loading model"):
                model = memory_profile_call(
                    "tl_model:load_replacement_model",
                    load_replacement_model,
                    self.model_id,
                    device=device,
                    dtype=dtype,
                    cache_dir=CACHE_DIR,
                )

            actual_layers = model.cfg.n_layers
            if actual_layers != NUM_LAYERS:
                print(f"[WARN] expected {NUM_LAYERS} layers, model has {actual_layers}")

            self._tokenizer = tokenizer
            self._model = model
            memory_checkpoint("ensure_loaded:model stored")

        if include_transcoders and self._transcoders is None:
            self._transcoders = memory_profile_call(
                "transcoders:load",
                load_transcoders,
                self.layers,
                device=device,
                dtype=dtype,
            )
        memory_checkpoint(f"ensure_loaded:end include_transcoders={include_transcoders}")

    def _loaded(self) -> tuple[Any, PreTrainedTokenizerBase, dict[int, SingleLayerTranscoder]]:
        self._ensure_loaded(include_transcoders=True)
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._transcoders is not None
        return self._model, self._tokenizer, self._transcoders

    def _loaded_for_preview(self) -> tuple[Any, PreTrainedTokenizerBase]:
        memory_checkpoint("preview_load:start")
        device, dtype = self._ensure_device_dtype()
        print(f"[INFO] device={device} dtype={dtype} layers={self.layers}")
        tokenizer = self._ensure_tokenizer()
        if self._preview_model is None:
            with timed("Loading preview model"):
                model = memory_profile_call(
                    "preview_model:AutoModelForCausalLM.from_pretrained",
                    AutoModelForCausalLM.from_pretrained,
                    self.model_id,
                    cache_dir=CACHE_DIR,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                memory_checkpoint("preview_model:after from_pretrained")
                with memory_scope("preview_model:to device"):
                    model.to(device).eval()
                for param in model.parameters():
                    param.requires_grad_(False)
                self._preview_model = model
                self._empty_device_cache()
                memory_checkpoint("preview_model:ready")
        memory_checkpoint("preview_load:end")
        assert self._preview_model is not None
        return self._preview_model, tokenizer

    def _inputs_for_prompt(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        *,
        use_chat_template: bool = True,
    ) -> tuple[torch.Tensor, list[int], list[str]]:
        if self._device is None:
            raise RuntimeError("runner device is not loaded")
        text = prompt
        if use_chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(self._device)

        input_token_ids = [int(token_id) for token_id in input_ids[0].detach().cpu().tolist()]
        prompt_tokens = tokenizer.batch_decode([[token_id] for token_id in input_token_ids])
        print(f"[INFO] prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
        return input_ids, input_token_ids, prompt_tokens

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
        handles.append(model.model.norm.register_forward_hook(make_final_hidden_hook(state)))
        return handles

    def preview(
        self,
        prompt: str,
        *,
        slug: str | None = None,
        top_k: int | None = None,
        use_chat_template: bool = True,
    ) -> PreviewResult:
        """Run a lightweight inference forward and return the default next-token target."""

        with self._lock:
            memory_checkpoint("preview:start")
            model, tokenizer = self._loaded_for_preview()
            resolved_slug = slug or slugify(prompt[:50])
            with memory_scope("preview:tokenize"):
                input_ids, input_token_ids, prompt_tokens = self._inputs_for_prompt(
                    tokenizer, prompt, use_chat_template=use_chat_template
                )
            with timed("Preview inference forward pass"), torch.no_grad():
                outputs = memory_profile_call(
                    "preview:hf_model_forward",
                    lambda: model(input_ids, use_cache=False),
                )
                logits = outputs.logits
                memory_checkpoint("preview:after model_forward")
                last_logits = logits[0, -1]
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
                memory_checkpoint("preview:after top tokens")

            memory_checkpoint("preview:end")
            return PreviewResult(
                prompt=prompt,
                slug=resolved_slug,
                use_chat_template=use_chat_template,
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
        node_threshold: float = DEFAULT_NODE_THRESHOLD,
        edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
        logit_prob_threshold: float = DEFAULT_LOGIT_PROB_THRESHOLD,
        max_logit_nodes: int = DEFAULT_MAX_LOGIT_NODES,
        graph_file_dir: Path | str | None = None,
        save_pt: str | None = None,
        use_chat_template: bool = True,
    ) -> GraphResult:
        """Generate and export a circuit-tracer-compatible graph JSON."""

        with self._lock:
            model, tokenizer, transcoders = self._loaded()
            resolved_slug = slug or slugify(prompt[:50])
            labels = load_feature_labels(set(self.layers))
            print(f"[INFO] loaded {len(labels)} feature labels")
            input_ids, input_token_ids, prompt_tokens = self._inputs_for_prompt(
                tokenizer, prompt, use_chat_template=use_chat_template
            )

            from biology_server_t_lens.tl_attribution import (
                TargetSpec,
                run_attribution_from_context,
                setup_attribution,
            )

            batch_size = max(self.batch_size, max_logit_nodes)
            with timed("TL replacement forward pass"):
                ctx = setup_attribution(
                    model,
                    input_ids,
                    batch_size=batch_size,
                    transcoders=transcoders,
                    layers=self.layers,
                )
                last_logits = ctx.logits[0, -1]
                logit_targets = select_logit_targets(
                    tokenizer,
                    last_logits,
                    pos=len(prompt_tokens) - 1,
                    prob_threshold=logit_prob_threshold,
                    max_logit_nodes=max_logit_nodes,
                    target_token_id=target_token_id,
                    target_token=target_token,
                )
                primary_target = logit_targets[0]
                target_logit = last_logits[primary_target.token_id]
                print(
                    f"[INFO] primary logit id={primary_target.token_id} "
                    f"({primary_target.token!r}) p={primary_target.prob:.4f} "
                    f"logit={target_logit.item():.3f}; "
                    f"logit nodes={len(logit_targets)}; "
                    f"active features={ctx.n_features}"
                )

            logit_specs = [
                TargetSpec(
                    kind="logit",
                    pos=len(prompt_tokens) - 1,
                    token_id=target.token_id,
                    prob=target.prob,
                )
                for target in logit_targets
            ]
            with timed("TL iterative top-K attribution"):
                active_features, dense_edge_matrix = run_attribution_from_context(
                    ctx,
                    logit_targets=logit_specs,
                    max_feature_nodes=self.max_feature_nodes,
                    update_interval=self.update_interval,
                )
                logit_weights = logit_weights_from_dense_matrix(
                    dense_edge_matrix=dense_edge_matrix,
                    selected=active_features,
                    logit_targets=logit_targets,
                    n_pos=len(input_token_ids),
                )
                selected = selected_features_from_active(
                    active_features=active_features,
                    labels=labels,
                    logit_weights=logit_weights,
                )
                all_links = dense_edge_matrix_links(
                    selected=selected,
                    input_token_ids=input_token_ids,
                    logit_targets=logit_targets,
                    dense_edge_matrix=dense_edge_matrix,
                )
                print(
                    f"[INFO] selected feature rows={len(selected)} unpruned links={len(all_links)}"
                )

            with timed("Anthropic-style graph pruning"):
                selected, all_links = prune_graph_by_indirect_influence(
                    selected=selected,
                    input_token_ids=input_token_ids,
                    links=all_links,
                    logit_targets=logit_targets,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )

            with timed("Per-feature direct-logit projection"), torch.no_grad():
                feature_logits = compute_feature_logits(
                    selected=selected,
                    transcoders=transcoders,
                    unembed=model.W_U,
                    tokenizer=tokenizer,
                )

            feature_examples = build_feature_examples(
                selected=selected,
                labels=labels,
                tokenizer=tokenizer,
                topk_dir=self.topk_dir,
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
                target_token_id=primary_target.token_id,
                target_token_str=primary_target.token,
                target_token_prob=primary_target.prob,
                logit_nodes=[
                    LogitNode(
                        vocab_idx=target.token_id,
                        token=target.token,
                        token_prob=target.prob,
                    )
                    for target in logit_targets
                ],
                node_threshold=node_threshold,
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
                        "target_token_id": primary_target.token_id,
                        "target_token_str": primary_target.token,
                        "target_token_prob": primary_target.prob,
                        "logit_targets": [asdict(target) for target in logit_targets],
                        "layers": self.layers,
                        "tokens": prompt_tokens,
                        "input_token_ids": input_token_ids,
                        "selected_features": [
                            selected_feature_to_dict(feature) for feature in selected
                        ],
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
                target_token_id=primary_target.token_id,
                target_token_str=primary_target.token,
                target_token_prob=primary_target.prob,
                prompt_tokens=prompt_tokens,
                input_token_ids=input_token_ids,
                selected_features=selected,
                links=all_links,
                logit_targets=logit_targets,
                pt_path=pt_path,
            )
