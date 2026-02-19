"""
Generate attribution graphs for Qwen3-4B using per-layer transcoders.

Uses circuit-tracer's load_transcoder_set and attribute functions with a
TransformerLens replacement model. Designed to run on CSD3 A100 GPUs.
"""

import time
from pathlib import Path

import torch

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-4B"
N_LAYERS = 36
# Local directory containing layer_0.safetensors through layer_35.safetensors.
# NOTE: Adjust this path to match where you've placed the transcoders on CSD3.
TRANSCODER_DIR = Path.home() / "rds" / "hpc-work" / "transcoders" / "qwen3-4b"
TRANSCODER_SCAN = "mwhanna/qwen3-4b-transcoders"

FEATURE_INPUT_HOOK = "mlp.hook_in"
FEATURE_OUTPUT_HOOK = "mlp.hook_out"

PROMPT = "The biological function of hemoglobin is to"

BATCH_SIZE = 256
MAX_FEATURE_NODES = 8192
OFFLOAD = "cpu"
DTYPE = torch.bfloat16

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "data" / "attribution_graphs"

# ── Helpers ──────────────────────────────────────────────────────────────────


def timed(label):
    """Context manager that prints elapsed time for a block."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"[START] {label}")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"[DONE]  {label} ({elapsed:.1f}s)")
    return Timer()


# ── Pipeline ─────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Prompt: {PROMPT!r}")
    print()

    # B. Resolve transcoder paths from local directory
    with timed("Resolving local transcoder paths"):
        if not TRANSCODER_DIR.is_dir():
            raise FileNotFoundError(
                f"Transcoder directory not found: {TRANSCODER_DIR}\n"
                f"Place layer_0.safetensors through layer_{N_LAYERS - 1}.safetensors there."
            )
        transcoder_paths = {}
        for layer in range(N_LAYERS):
            path = TRANSCODER_DIR / f"layer_{layer}.safetensors"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing transcoder for layer {layer}: {path}"
                )
            transcoder_paths[layer] = str(path)
        print(f"  Found {len(transcoder_paths)} transcoder files in {TRANSCODER_DIR}")

    # C. Load TranscoderSet
    with timed("Loading TranscoderSet"):
        transcoder_set = load_transcoder_set(
            transcoder_paths=transcoder_paths,
            scan=TRANSCODER_SCAN,
            feature_input_hook=FEATURE_INPUT_HOOK,
            feature_output_hook=FEATURE_OUTPUT_HOOK,
            device=device,
            dtype=DTYPE,
            special_load_fn=None,
            lazy_encoder=True,
            lazy_decoder=True,
        )
        print(f"  Layers: {transcoder_set.n_layers}, "
              f"d_transcoder: {transcoder_set.d_transcoder}, "
              f"skip_connection: {transcoder_set.skip_connection}")

    # D. Create ReplacementModel
    with timed("Creating TransformerLensReplacementModel"):
        model = TransformerLensReplacementModel.from_pretrained_and_transcoders(
            MODEL_NAME,
            transcoder_set,
            device=device,
            dtype=DTYPE,
        )
        print(f"  Model layers: {model.cfg.n_layers}, "
              f"d_model: {model.cfg.d_model}")

    # E. Run attribution
    with timed("Running attribution"):
        graph = attribute(
            prompt=PROMPT,
            model=model,
            batch_size=BATCH_SIZE,
            max_feature_nodes=MAX_FEATURE_NODES,
            offload=OFFLOAD,
            verbose=True,
        )
        print(f"  active_features: {graph.active_features.shape}")
        print(f"  adjacency_matrix: {graph.adjacency_matrix.shape}")

    # F. Save graph
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prompt_slug = PROMPT[:30].replace(" ", "_").replace("/", "-")
    filename = f"graph_{prompt_slug}_{timestamp}.pt"
    save_path = OUTPUT_DIR / filename

    with timed("Saving graph"):
        graph.to_pt(str(save_path))
        print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
