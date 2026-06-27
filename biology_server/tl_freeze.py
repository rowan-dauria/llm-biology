"""Linearisation freezes for a TransformerLens ``HookedTransformer``.

Installs permanent forward hooks that detach the non-linear pieces of the model
(attention pattern, RMS/LayerNorm scale, MLP output) so that ``.backward()`` from
the residual stream only flows through linear ops. The forward pass is
numerically unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

_MARKER_ATTR = "_biology_t_lens_freezes_installed"

BwdHook = tuple[str, Callable[[torch.Tensor, HookPoint], None]]


def _stop_gradient(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
    return acts.detach()


def _enable_gradient(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
    acts.requires_grad_(True)
    return acts


def _scale_hook_points(block) -> list[HookPoint]:
    points: list[HookPoint] = []
    for name in ("ln1", "ln2", "ln1_post", "ln2_post"):
        sub = getattr(block, name, None)
        if sub is None:
            continue
        hook = getattr(sub, "hook_scale", None)
        if isinstance(hook, HookPoint):
            points.append(hook)
    return points


def install_freezes(model: HookedTransformer) -> None:
    """Install permanent stop-gradient hooks that linearise the backward pass.

    Idempotent: a second call is a no-op (does not stack hooks).
    """
    if getattr(model, _MARKER_ATTR, False):
        return

    for block in model.blocks:
        # stop gradients passing through attention
        block.attn.hook_pattern.add_hook(_stop_gradient, is_permanent=True)
        # stop gradients passing through layernorms
        for scale_hook in _scale_hook_points(block):
            scale_hook.add_hook(_stop_gradient, is_permanent=True)
        # stop gradients through the mlp block
        block.hook_mlp_out.add_hook(_stop_gradient, is_permanent=True)

    final_scale = getattr(model.ln_final, "hook_scale", None)
    if final_scale is not None:
        assert isinstance(final_scale, HookPoint)
        final_scale.add_hook(_stop_gradient, is_permanent=True)

    for param in model.parameters():
        param.requires_grad = False

    model.hook_embed.add_hook(_enable_gradient, is_permanent=True)

    setattr(model, _MARKER_ATTR, True)


def _capture_grad(store: dict, name: str):
    def hook(grad: torch.Tensor, hook: HookPoint) -> None:  # noqa: ARG001
        store[name] = grad.detach()

    return hook


def verify_linearisation(model: HookedTransformer) -> None:
    """Forward + backward a random input; assert non-linear paths are dead.

    Linear paths (V, attention output ``z``, embeddings) must receive non-zero
    gradient; non-linear paths (Q, K, RMS scale, MLP internal pre/post) must
    not (their backward hooks either do not fire or fire with all-zero grads).
    Raises ``AssertionError`` on any violation.
    """
    if not getattr(model, _MARKER_ATTR, False):
        raise RuntimeError("verify_linearisation called before install_freezes")

    n_layers = model.cfg.n_layers
    seq_len = min(model.cfg.n_ctx, 4)
    device = next(model.parameters()).device

    grads: dict[str, torch.Tensor] = {}
    bwd_hooks: list[BwdHook] = []

    def add(name: str):
        bwd_hooks.append((name, _capture_grad(grads, name)))

    for layer in range(n_layers):
        add(f"blocks.{layer}.attn.hook_q")
        add(f"blocks.{layer}.attn.hook_k")
        add(f"blocks.{layer}.attn.hook_v")
        add(f"blocks.{layer}.attn.hook_z")
        add(f"blocks.{layer}.ln1.hook_scale")
        add(f"blocks.{layer}.ln2.hook_scale")
        add(f"blocks.{layer}.mlp.hook_pre")
        add(f"blocks.{layer}.mlp.hook_post")
    add("hook_embed")

    tokens = torch.randint(0, model.cfg.d_vocab, (1, seq_len), device=device)

    was_training = model.training
    model.eval()
    try:
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(tokens)
            loss = logits.sum()
            loss.backward()
    finally:
        if was_training:
            model.train()

    must_be_zero = []
    must_be_nonzero = []
    for layer in range(n_layers):
        must_be_zero.extend(
            [
                f"blocks.{layer}.attn.hook_q",
                f"blocks.{layer}.attn.hook_k",
                f"blocks.{layer}.ln1.hook_scale",
                f"blocks.{layer}.ln2.hook_scale",
                f"blocks.{layer}.mlp.hook_pre",
                f"blocks.{layer}.mlp.hook_post",
            ]
        )
        must_be_nonzero.extend(
            [
                f"blocks.{layer}.attn.hook_v",
                f"blocks.{layer}.attn.hook_z",
            ]
        )
    must_be_nonzero.append("hook_embed")

    for name in must_be_zero:
        grad = grads.get(name)
        if grad is None:
            continue
        if grad.abs().max().item() != 0.0:
            raise AssertionError(
                f"Linearisation violation: {name} received non-zero gradient "
                f"(max abs {grad.abs().max().item():.3e}); a non-linear path "
                "is still in the backward graph."
            )

    for name in must_be_nonzero:
        grad = grads.get(name)
        if grad is None:
            raise AssertionError(
                f"Linearisation violation: {name} received no gradient at all; "
                "linear path is broken."
            )
        if grad.abs().max().item() == 0.0:
            raise AssertionError(
                f"Linearisation violation: {name} received all-zero gradient; "
                "linear path is broken."
            )
