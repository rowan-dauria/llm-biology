"""Load Qwen3 (or any HF model TL supports) as a linearised ``HookedTransformer``.

Wraps ``HookedTransformer.from_pretrained`` with the freeze settings the
biology_server attribution backend needs: ``fold_ln=False``,
``center_writing_weights=False``, ``center_unembed=False`` so unembed-vector
injection and residual interpretation line up with the un-folded model. Then
applies :func:`install_freezes` so backward is linear.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from biology_server_t_lens.tl_freeze import install_freezes


def load_replacement_model(
    model_id: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | str | None = None,
) -> HookedTransformer:
    """Load a HookedTransformer and install linearisation freezes.

    ``model_id`` is anything ``HookedTransformer.from_pretrained`` accepts (e.g.
    ``"Qwen/Qwen3-4B"``). ``cache_dir`` is forwarded as ``hf_model`` cache via
    ``HookedTransformer`` ignoring it — set ``HF_HOME`` in the environment for
    true cache control. Accepted here for symmetry with the legacy backend.
    """
    kwargs: dict = {
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

    model = HookedTransformer.from_pretrained(model_id, **kwargs)
    install_freezes(model)
    model.eval()
    return model
