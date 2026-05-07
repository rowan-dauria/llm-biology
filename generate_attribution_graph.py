"""
Custom mini-attribution for Qwen3-4B using a subset of per-layer transcoders.

Only depends on circuit-tracer's `SingleLayerTranscoder` (loaded via
`load_relu_transcoder`, the canonical factory for that class). No
ReplacementModel, no high-level `attribute()` API.

For each chosen layer:
  1. Hook the MLP module: replace its output with the transcoder reconstruction.
  2. Inside the hook, expose a leaf `features` tensor (requires_grad=True) so
     the autograd graph from logits flows back into per-feature gradients.
  3. After backward, attribution = features.detach() * features.grad.

This is a first-cut feature->logit attribution (gradient x activation). It
does *not* compute feature->feature edges. Once those are needed we extend.
"""

import os
import time
from pathlib import Path

import torch
from circuit_tracer.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    load_relu_transcoder,
)
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-4B"
TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"

# Qwen3-4B has 36 transformer layers. Sample 4 across depth.
NUM_LAYERS = 36
LAYERS_TO_HOOK = [
    2,  # near start
    NUM_LAYERS // 3,  # 12
    (2 * NUM_LAYERS) // 3,  # 24
    NUM_LAYERS - 3,  # 33, near end
]

PROMPT = "The biological function of hemoglobin is to"
TOP_K = 20

CACHE_DIR = os.getenv("HF_HOME")
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "data" / "attribution_graphs"


def pick_device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def timed(label):
    class T:
        def __enter__(self):
            self.start = time.time()
            print(f"[START] {label}")
            return self

        def __exit__(self, *_):
            print(f"[DONE]  {label} ({time.time() - self.start:.1f}s)")

    return T()


def make_mlp_hook(layer_idx: int, transcoder: SingleLayerTranscoder, captured: dict):
    """
    Forward hook that:
      - encodes the MLP input through the transcoder (no grad on encoder weights),
      - exposes the resulting feature vector as a fresh leaf w/ requires_grad=True,
      - replaces the MLP output with the transcoder's decoded reconstruction.
    """

    def hook(_module, inputs, output):
        mlp_input = inputs[0]
        with torch.no_grad():
            features_value = transcoder.encode(mlp_input)
        features = features_value.detach().clone().requires_grad_(True)
        captured[layer_idx] = features
        reconstruction = transcoder.decode(features, mlp_input)
        return reconstruction.to(output.dtype)

    return hook


def main():
    device, dtype = pick_device_dtype()
    print(f"Device: {device}  dtype: {dtype}  layers: {LAYERS_TO_HOOK}")

    # 1. Resolve + download transcoders for chosen layers only.
    # Done *before* the model load so the download has minimal memory pressure;
    # the repo also ships ~43 GB of `features/layer_*.bin` interpretability
    # files we don't use, hence the allow_patterns filter.
    wanted_files = [f"layer_{layer}.safetensors" for layer in LAYERS_TO_HOOK]
    with timed("Resolving transcoder paths"):
        transcoder_dir = Path(
            snapshot_download(
                TRANSCODER_REPO,
                allow_patterns=wanted_files,
            )
        )

    # 2. Tokenizer + model
    with timed("Loading model"):
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=dtype, trust_remote_code=True
        )
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

    actual_layers = len(model.model.layers)
    if actual_layers != NUM_LAYERS:
        print(f"Warning: expected {NUM_LAYERS} layers, model has {actual_layers}")

    transcoders: dict[int, SingleLayerTranscoder] = {}
    for layer in LAYERS_TO_HOOK:
        path = transcoder_dir / f"layer_{layer}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"No transcoder for layer {layer}: {path}")
        with timed(f"Loading transcoder layer {layer}"):
            tc = load_relu_transcoder(
                str(path),
                layer,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )
            for p in tc.parameters():
                p.requires_grad_(False)
            transcoders[layer] = tc
            print(
                f"  d_model={tc.d_model}  d_transcoder={tc.d_transcoder}  "
                f"skip={tc.W_skip is not None}"
            )

    # 3. Tokenize prompt
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    token_strs = tokenizer.convert_ids_to_tokens(inputs.input_ids[0].tolist())
    print(f"Prompt tokens ({len(token_strs)}): {token_strs}")

    # 4. Install hooks
    captured_features: dict[int, torch.Tensor] = {}
    handles = []
    for layer in LAYERS_TO_HOOK:
        target_module = model.model.layers[layer].mlp
        handles.append(
            target_module.register_forward_hook(
                make_mlp_hook(layer, transcoders[layer], captured_features)
            )
        )

    # 5. Forward + backward on top-1 next-token logit
    try:
        with timed("Forward + backward"):
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1]  # (vocab,)
            target_token = int(last_logits.argmax())
            target_logit = last_logits[target_token]
            print(
                f"Target token id={target_token} "
                f"({tokenizer.decode([target_token])!r})  "
                f"logit={target_logit.item():.3f}"
            )
            target_logit.backward()
    finally:
        for h in handles:
            h.remove()

    # 6. Compute + report per-layer attributions
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt_slug = PROMPT[:30].replace(" ", "_").replace("/", "-")
    save_path = OUTPUT_DIR / f"attribution_{prompt_slug}_{timestamp}.pt"

    attribution_per_layer: dict[int, torch.Tensor] = {}
    for layer in LAYERS_TO_HOOK:
        features = captured_features[layer]  # (1, seq, d_t)
        grads = features.grad
        if grads is None:
            print(f"Warning: no grad captured for layer {layer}")
            continue
        attribution = (features.detach() * grads).cpu().to(torch.float32)
        attribution_per_layer[layer] = attribution

        per_feature = attribution.squeeze(0).sum(dim=0)  # (d_t,)
        k = min(TOP_K, per_feature.numel())
        top_vals, top_idx = per_feature.abs().topk(k)
        print(f"\nLayer {layer}: top-{k} features by |sum-position attribution|")
        for v_abs, idx in zip(top_vals.tolist(), top_idx.tolist(), strict=True):
            signed = per_feature[idx].item()
            print(f"  feature {idx:6d}  signed={signed:+.4f}  |.|={v_abs:.4f}")

    torch.save(
        {
            "prompt": PROMPT,
            "target_token_id": target_token,
            "target_token_str": tokenizer.decode([target_token]),
            "layers": LAYERS_TO_HOOK,
            "tokens": token_strs,
            "attribution_per_layer": attribution_per_layer,
        },
        save_path,
    )
    print(f"\nSaved attribution to {save_path}")


if __name__ == "__main__":
    main()
