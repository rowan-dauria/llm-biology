"""Turn collected activations into per-pole statistics and readable examples.

A neuron has two poles: strong positive and strong negative activations. Each
is described separately, using only gate-open firings (``gate > 0``). Firings
with a closed gate are the sign-flipped SwiGLU leak; they are counted and shown
apart so they do not pass for the neuron's job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache

import numpy as np

from neuron_labelling.model import Tokenizer, silu

POLES = ("positive", "negative")
SIGN = {"positive": 1.0, "negative": -1.0}


def pole_strength(gate: np.ndarray, value: np.ndarray, pole: str, gate_open: bool) -> np.ndarray:
    """Activation in the pole's direction (>= 0), restricted to an open or closed gate."""
    signed = SIGN[pole] * silu(gate) * value
    mask = gate > 0 if gate_open else gate <= 0
    return np.where(mask, np.maximum(signed, 0.0), 0.0)


@dataclass
class PoleStats:
    max_act: float
    density: float  # share of tokens above 10% of max_act
    leak_share_top: float  # share of the raw top-k (ignoring the gate) that is leak
    leak_max_ratio: float  # strongest leak / strongest gate-open firing


def pole_stats(
    gate: np.ndarray, value: np.ndarray, pole: str, valid: np.ndarray, top_k: int = 50
) -> PoleStats:
    open_strength = np.where(valid, pole_strength(gate, value, pole, True), 0.0)
    leak_strength = np.where(valid, pole_strength(gate, value, pole, False), 0.0)
    max_act = float(open_strength.max())
    raw = np.maximum(open_strength, leak_strength).ravel()
    k = min(top_k, raw.size)
    top = np.argpartition(-raw, k - 1)[:k]
    top = top[raw[top] > 0]
    leak_share = float((leak_strength.ravel()[top] > 0).mean()) if top.size else 0.0
    return PoleStats(
        max_act=max_act,
        density=float((open_strength > 0.1 * max_act).sum() / max(valid.sum(), 1))
        if max_act > 0
        else 0.0,
        leak_share_top=leak_share,
        leak_max_ratio=float(leak_strength.max() / max_act) if max_act > 0 else 0.0,
    )


@dataclass
class Example:
    chunk: int
    position: int
    act: float  # raw activation
    gate: float
    value: float
    score: int  # 0-10, relative to the pole's gate-open maximum
    text: str  # context with the firing token marked [[like this]]

    def to_dict(self) -> dict:
        return asdict(self)


def make_decoder(tokenizer: Tokenizer) -> Callable[[int], str]:
    @cache
    def decode(token_id: int) -> str:
        return tokenizer.decode([token_id]).replace("\n", "⏎")

    return decode


def render(
    tokens: np.ndarray,
    strength: np.ndarray,
    position: int,
    max_act: float,
    decode: Callable[[int], str],
    left: int = 32,
    right: int = 8,
) -> str:
    """Context window; the firing token is ``[[marked]]`` and others score as ``tok⟨n⟩``."""
    start, stop = max(0, position - left), min(len(tokens), position + right + 1)
    parts = []
    for pos in range(start, stop):
        piece = decode(int(tokens[pos]))
        score = round(10 * strength[pos] / max_act) if max_act > 0 else 0
        if pos == position:
            piece = f"[[{piece}]]⟨{score}⟩"
        elif score >= 2:
            piece = f"{piece}⟨{score}⟩"
        parts.append(piece)
    return "".join(parts)


def select_positions(
    strength: np.ndarray,
    k: int,
    lo: float = 0.0,
    hi: float = float("inf"),
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """At most one position per chunk: the top ``k`` chunks, or a random ``k`` in a band.

    With ``rng`` set, samples chunks whose peak lies in ``[lo, hi)`` instead of
    taking the strongest.
    """
    peak_pos = strength.argmax(axis=1)
    peak = strength[np.arange(strength.shape[0]), peak_pos]
    candidates = np.flatnonzero((peak > 0) & (peak >= lo) & (peak < hi))
    if rng is None:
        chosen = candidates[np.argsort(-peak[candidates], kind="stable")][:k]
    else:
        chosen = rng.choice(candidates, size=min(k, candidates.size), replace=False)
        chosen = chosen[np.argsort(-peak[chosen], kind="stable")]
    return [(int(c), int(peak_pos[c])) for c in chosen]


@dataclass
class PoleEvidence:
    pole: str
    stats: PoleStats
    top: list[Example]
    weaker: list[Example]  # sampled from 20-50% of max_act
    leaks: list[Example]  # strongest gate-closed firings in this direction


def pole_evidence(
    tokens: np.ndarray,
    gate: np.ndarray,
    value: np.ndarray,
    pole: str,
    decode: Callable[[int], str],
    skip_positions: int = 4,
    n_top: int = 20,
    n_weaker: int = 8,
    n_leaks: int = 3,
    seed: int = 0,
) -> PoleEvidence:
    """Statistics and examples for one pole of one neuron (arrays are ``[chunks, seq]``)."""
    valid = np.zeros_like(gate, dtype=bool)
    valid[:, skip_positions:] = True
    stats = pole_stats(gate, value, pole, valid)
    act = silu(gate) * value
    open_strength = np.where(valid, pole_strength(gate, value, pole, True), 0.0)
    leak_strength = np.where(valid, pole_strength(gate, value, pole, False), 0.0)

    def examples(strength: np.ndarray, positions: list[tuple[int, int]]) -> list[Example]:
        return [
            Example(
                chunk=c,
                position=p,
                act=float(act[c, p]),
                gate=float(gate[c, p]),
                value=float(value[c, p]),
                score=round(10 * strength[c, p] / stats.max_act) if stats.max_act > 0 else 0,
                text=render(tokens[c], strength[c], p, stats.max_act, decode),
            )
            for c, p in positions
        ]

    rng = np.random.default_rng(seed)
    band = (0.2 * stats.max_act, 0.5 * stats.max_act)
    return PoleEvidence(
        pole=pole,
        stats=stats,
        top=examples(open_strength, select_positions(open_strength, n_top)),
        weaker=examples(open_strength, select_positions(open_strength, n_weaker, *band, rng=rng)),
        leaks=examples(leak_strength, select_positions(leak_strength, n_leaks)),
    )
