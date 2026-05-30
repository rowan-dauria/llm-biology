import os
from pathlib import Path

import torch
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_ID = "Qwen/Qwen3-4B"
CACHE_DIR = os.getenv("HF_HOME")
LAYERS_TO_HOOK = [10, 20]
SCRIPT_DIR = Path(__file__).parent.resolve()


def get_mlp_activations(model, tokenizer, prompts, layers_to_hook, batch_size=8):
    """
    Runs prompts through the model and captures MLP activations for specific layers.
    """
    activations = {layer: [] for layer in layers_to_hook}
    device = next(model.parameters()).device

    def get_hook(layer_idx):
        def hook(module, input, output):
            # MLP output is a tensor of shape (batch_size, sequence_length, hidden_dim)
            activations[layer_idx].append(output.detach().cpu())

        return hook

    handles = []
    for layer_idx in layers_to_hook:
        target_module = model.model.layers[layer_idx].mlp
        handle = target_module.register_forward_hook(get_hook(layer_idx))
        handles.append(handle)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                model(**inputs)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM - try reducing batch_size or number of layers")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Example Prompts (Replace with your synthetic dataset)
    prompts = ["The capital of France is Paris.", "The biological structure of a cell includes..."]

    captured_data = get_mlp_activations(model, tokenizer, prompts, LAYERS_TO_HOOK)

    # Save to Disk
    output_dir = SCRIPT_DIR / "data" / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, data_list in captured_data.items():
        save_path = output_dir / f"layer_{layer}_mlp.pt"
        # Save as list to preserve variable sequence lengths per batch
        torch.save(data_list, save_path)
        total_samples = sum(t.shape[0] for t in data_list)
        print(
            f"Saved layer {layer} activations to {save_path} ({total_samples} samples, {len(data_list)} batches)"
        )


if __name__ == "__main__":
    main()
