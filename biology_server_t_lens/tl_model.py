"""Load Qwen3 (or any HF model TL supports) as a linearised ``HookedTransformer``.

Wraps ``HookedTransformer.from_pretrained`` with the freeze settings the
biology_server attribution backend needs: ``fold_ln=False``,
``center_writing_weights=False``, ``center_unembed=False`` so unembed-vector
injection and residual stream line up with the transcoders expect. Then
applies :func:`install_freezes` so backward is linear.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from biology_server_t_lens.memory_profile import memory_checkpoint, memory_profile_call
from biology_server_t_lens.tl_freeze import install_freezes


def load_replacement_model(
    model_id: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | str | None = None,
    hf_model_id: str | None = None,
) -> HookedTransformer:
    """Load a HookedTransformer and install linearisation freezes.

    ``model_id`` is anything ``HookedTransformer.from_pretrained`` accepts (e.g.
    ``"Qwen/Qwen3-4B"``). ``cache_dir`` is forwarded as ``hf_model`` cache via
    ``HookedTransformer`` ignoring it — set ``HF_HOME`` in the environment for
    true cache control. Accepted here for symmetry with the legacy backend.

    If ``hf_model_id`` is set, load that Hugging Face model/folder first and
    pass it as ``hf_model`` while keeping ``model_id`` for TransformerLens's
    architecture/config lookup. This is useful for local fine-tuned or merged
    checkpoints whose architecture is still a TL-supported base model.
    """

    kwargs: dict = {
        # see https://transformer-circuits.pub/2021/framework/index.html#:~:text=Handling%20Layer%20Normalization
        # these settings can be used to make the model more interpretable, but the we need to disable
        # them because they would change the residual stream input to the transcoders, which would change the
        # behaviour of the replacement model.
        "fold_ln": False,
        "center_writing_weights": False,
        "center_unembed": False,
        "device": str(device) if isinstance(device, torch.device) else device,
        "dtype": dtype,
    }
    if cache_dir is not None:
        # HookedTransformer.from_pretrained doesn't take cache_dir directly;
        # set HF_HOME externally if needed. Kept here for signature symmetry.
        pass

    hf_model = None
    if hf_model_id is not None:
        hf_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if cache_dir is not None:
            hf_kwargs["cache_dir"] = str(cache_dir)
        memory_checkpoint("tl_model:before AutoModelForCausalLM.from_pretrained override")
        hf_model = memory_profile_call(
            "tl_model:AutoModelForCausalLM.from_pretrained override",
            AutoModelForCausalLM.from_pretrained,
            hf_model_id,
            **hf_kwargs,
        )
        for param in hf_model.parameters():
            param.requires_grad = False
        memory_checkpoint("tl_model:after AutoModelForCausalLM.from_pretrained override")

    memory_checkpoint("tl_model:before HookedTransformer.from_pretrained")
    model = memory_profile_call(
        "tl_model:HookedTransformer.from_pretrained",
        HookedTransformer.from_pretrained,
        model_id,
        hf_model=hf_model,
        **kwargs,
    )
    memory_checkpoint("tl_model:after HookedTransformer.from_pretrained")
    install_freezes(model)
    memory_checkpoint("tl_model:after install_freezes")
    model.eval()
    memory_checkpoint("tl_model:after eval")
    return model
