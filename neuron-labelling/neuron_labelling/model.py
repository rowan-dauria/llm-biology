"""Read and steer individual SwiGLU MLP neurons in a Hugging Face Qwen3 model.

A Qwen3 MLP neuron ``n`` in layer ``l`` computes

    act = SiLU(gate) * value,  gate = x @ W_gate[n],  value = x @ W_up[n]

and writes ``act * W_down[:, n]`` into the residual stream. We record ``gate``
and ``value`` separately so callers can tell gate-open firings from the small
sign-flipped "leak" a closed gate lets through (SiLU of a negative gate is a
small negative number, so a closed gate times a negative value is positive).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True, order=True)
class NeuronRef:
    layer: int
    index: int

    @classmethod
    def parse(cls, text: str) -> NeuronRef:
        """Parse ``L12/N345`` or ``12:345``."""
        match = re.fullmatch(r"\s*L?(\d+)\s*[/:]\s*N?(\d+)\s*", text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"cannot parse neuron {text!r}; expected L<layer>/N<index>")
        return cls(int(match.group(1)), int(match.group(2)))

    @property
    def slug(self) -> str:
        return f"L{self.layer}_N{self.index}"

    def __str__(self) -> str:
        return f"L{self.layer}/N{self.index}"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def load_model(model_id: str, device: str = "auto", dtype: str = "auto") -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=resolve_dtype(dtype, torch_device)
    )
    model.to(torch_device).eval()
    return model, tokenizer


def mlp_module(model: Any, layer: int) -> Any:
    return model.model.layers[layer].mlp


def check_neurons(model: Any, neurons: Sequence[NeuronRef]) -> None:
    n_layers = model.config.num_hidden_layers
    width = model.config.intermediate_size
    for neuron in neurons:
        if not (0 <= neuron.layer < n_layers and 0 <= neuron.index < width):
            raise ValueError(f"{neuron} is outside {n_layers} layers x {width} neurons")
    if model.config.hidden_act != "silu":
        raise ValueError(f"expected a SiLU-gated MLP, got hidden_act={model.config.hidden_act!r}")


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NeuronRecorder:
    """Record gate and value pre-activations for a fixed set of neurons.

    Hooks the MLP input and projects it onto the neurons' gate/up rows, so
    nothing of width ``intermediate_size`` is stored.
    """

    def __init__(self, model: Any, neurons: Sequence[NeuronRef]):
        check_neurons(model, neurons)
        self.model = model
        self.neurons = list(neurons)
        self._by_layer: dict[int, list[int]] = defaultdict(list)
        for column, neuron in enumerate(self.neurons):
            self._by_layer[neuron.layer].append(column)
        self._handles: list[Any] = []
        self._gate = [torch.empty(0)] * len(self.neurons)
        self._value = [torch.empty(0)] * len(self.neurons)

    def _hook(self, layer: int, columns: list[int]):
        mlp = mlp_module(self.model, layer)
        index = torch.tensor([self.neurons[c].index for c in columns])
        w_gate = mlp.gate_proj.weight[index].detach().float()
        w_up = mlp.up_proj.weight[index].detach().float()

        def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0].float()
            gate = x @ w_gate.T
            value = x @ w_up.T
            for k, column in enumerate(columns):
                self._gate[column] = gate[..., k].cpu()
                self._value[column] = value[..., k].cpu()

        return hook

    def __enter__(self) -> NeuronRecorder:
        for layer, columns in self._by_layer.items():
            module = mlp_module(self.model, layer)
            self._handles.append(module.register_forward_pre_hook(self._hook(layer, columns)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``gate, value`` of shape ``[n_neurons, batch, seq]`` from the last forward."""
        gate = torch.stack(self._gate).numpy()
        value = torch.stack(self._value).numpy()
        return gate, value


class NeuronSteerer:
    """Add ``delta`` to one neuron's activation at every position."""

    def __init__(self, model: Any, neuron: NeuronRef, delta: float):
        check_neurons(model, [neuron])
        self.down_proj = mlp_module(model, neuron.layer).down_proj
        self.neuron = neuron
        self.delta = delta
        self._handle: Any = None

    def _hook(self, _module: Any, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        hidden = args[0].clone()
        hidden[..., self.neuron.index] += self.delta
        return (hidden, *args[1:])

    def __enter__(self) -> NeuronSteerer:
        self._handle = self.down_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _pad(batch: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, ids in enumerate(batch):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = 1
    return input_ids, mask


def pad_id(tokenizer: Tokenizer) -> int:
    for candidate in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if candidate is not None:
            return candidate
    return 0


@dataclass
class TextActivations:
    tokens: list[int]
    gate: np.ndarray  # [n_neurons, len(tokens)]
    value: np.ndarray

    @property
    def act(self) -> np.ndarray:
        return silu(self.gate) * self.value


class Probe:
    """Run arbitrary texts through the model and read the chosen neurons."""

    def __init__(
        self, model: Any, tokenizer: Tokenizer, neurons: Sequence[NeuronRef], batch_size: int = 8
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.neurons = list(neurons)
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def column(self, neuron: NeuronRef) -> int:
        return self.neurons.index(neuron)

    @torch.no_grad()
    def measure(self, texts: Sequence[str]) -> list[TextActivations]:
        results: list[TextActivations] = []
        encoded = [self.tokenizer.encode(text, add_special_tokens=False) or [0] for text in texts]
        for start in range(0, len(encoded), self.batch_size):
            batch = encoded[start : start + self.batch_size]
            input_ids, mask = _pad(batch, pad_id(self.tokenizer))
            with NeuronRecorder(self.model, self.neurons) as recorder:
                self.model.model(  # decoder only; the LM head is not needed
                    input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)
                )
                gate, value = recorder.read()
            for row, ids in enumerate(batch):
                n = len(ids)
                results.append(TextActivations(list(ids), gate[:, row, :n], value[:, row, :n]))
        return results

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        neuron: NeuronRef | None = None,
        delta: float = 0.0,
        max_new_tokens: int = 40,
    ) -> list[str]:
        """Greedy continuations, optionally with ``delta`` added to ``neuron``."""
        outputs: list[str] = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([ids], device=self.device)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": pad_id(self.tokenizer),
                "attention_mask": torch.ones_like(input_ids),
            }
            if neuron is None or delta == 0.0:
                generated = self.model.generate(input_ids, **kwargs)
            else:
                with NeuronSteerer(self.model, neuron, delta):
                    generated = self.model.generate(input_ids, **kwargs)
            outputs.append(self.tokenizer.decode(generated[0, len(ids) :].tolist()))
        return outputs


@torch.no_grad()
def logit_effects(
    model: Any, tokenizer: Tokenizer, neuron: NeuronRef, k: int = 15
) -> dict[str, list[tuple[str, float]]]:
    """Tokens a positive activation pushes up and down via the direct path.

    Projects ``W_down[:, n]``, scaled by the final RMSNorm gain, onto the
    unembedding. A negative activation flips both lists.
    """
    check_neurons(model, [neuron])
    direction = mlp_module(model, neuron.layer).down_proj.weight[:, neuron.index].float()
    direction = direction * model.model.norm.weight.float()
    logits = model.get_output_embeddings().weight.float() @ direction
    top = torch.topk(logits, k)
    bottom = torch.topk(-logits, k)

    def named(values: torch.Tensor, ids: torch.Tensor, sign: float) -> list[tuple[str, float]]:
        return [
            (tokenizer.decode([int(i)]), round(sign * float(v), 4))
            for v, i in zip(values, ids, strict=True)
        ]

    return {
        "promotes": named(top.values, top.indices, 1.0),
        "suppresses": named(bottom.values, bottom.indices, -1.0),
    }
