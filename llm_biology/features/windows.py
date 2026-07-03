"""Shared helpers for decoding and formatting top-K feature windows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from .corpus import iter_texts
except ImportError:
    from corpus import iter_texts


@dataclass(slots=True)
class Window:
    rank: int
    value: float
    prompt_id: int
    token_pos: int
    rendered: str
    active: bool


@dataclass(slots=True)
class FeatureSummary:
    feature: int
    max_activation: float
    n_distinct_prompts: int
    active_prompt_ids: frozenset[int]


def _feature_vectors(
    layer_data: Mapping[str, Any], feature_idx: int
) -> tuple[list[float], list[int]]:
    vals = layer_data["topk_vals"][:, feature_idx].tolist()
    prompt_ids = layer_data["topk_prompt_id"][:, feature_idx].tolist()
    return [float(val) for val in vals], [int(pid) for pid in prompt_ids]


def active_prompt_ids(layer_data: Mapping[str, Any], feature_idx: int) -> frozenset[int]:
    vals, prompt_ids = _feature_vectors(layer_data, feature_idx)
    return frozenset(
        pid for pid, val in zip(prompt_ids, vals, strict=True) if math.isfinite(val) and val > 0
    )


def select_features(
    layer_data: Mapping[str, Any],
    *,
    top_n: int,
    diversity: int,
) -> list[FeatureSummary]:
    topk_vals = layer_data["topk_vals"]
    ranked = torch.argsort(torch.amax(topk_vals, dim=0), descending=True).tolist()

    selected: list[FeatureSummary] = []
    for feature_idx in ranked:
        vals, prompt_ids = _feature_vectors(layer_data, feature_idx)
        active_ids = frozenset(
            pid for pid, val in zip(prompt_ids, vals, strict=True) if math.isfinite(val) and val > 0
        )
        # note: diversity drops features whose top-K windows pile into too few prompts —
        # those are usually single-doc artefacts (e.g. fires on every comma in one file)
        # rather than generalisable concepts, and the LLM label would be misleading.
        # Caveat on small corpora (~1k Pile docs): a genuinely niche feature (e.g. fires
        # on "Habsburg") may only have 1-2 source docs and get filtered out too. Lower
        # `--diversity` (e.g. 2) on small corpora, or scale the corpus to ~10k docs.
        if len(active_ids) < diversity:
            continue
        selected.append(
            FeatureSummary(
                feature=feature_idx,
                max_activation=max(vals),
                n_distinct_prompts=len(active_ids),
                active_prompt_ids=active_ids,
            )
        )
        if len(selected) >= top_n:
            break
    return selected


def collect_prompt_texts(
    corpus_spec: str, needed_prompt_ids: set[int], num_parts: int = 1
) -> dict[int, str]:
    if not needed_prompt_ids:
        return {}

    # prompt_ids encode their shard (pid = local * num_parts + worker), so recover
    # per worker: re-stream that worker's shard and stop once its locals are covered.
    by_worker: dict[int, set[int]] = {}
    for pid in needed_prompt_ids:
        by_worker.setdefault(pid % num_parts, set()).add(pid)

    text_by_id: dict[int, str] = {}
    for worker, ids in by_worker.items():
        max_local = max(pid // num_parts for pid in ids)
        for pid, text in iter_texts(corpus_spec, part_idx=worker, num_parts=num_parts):
            if pid in ids:
                text_by_id[pid] = text
            if pid // num_parts >= max_local:
                break
    return text_by_id


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    local_dir = snapshot_download(model_id, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_windows(
    layer_data: Mapping[str, Any],
    feature_idx: int,
    tokenizer: PreTrainedTokenizerBase,
    window: int = 10,
    *,
    text_by_prompt_id: Mapping[int, str] | None = None,
) -> list[Window]:
    vals, prompt_ids = _feature_vectors(layer_data, feature_idx)
    token_pos = layer_data["topk_token_pos"][:, feature_idx].tolist()
    token_pos = [int(pos) for pos in token_pos]
    max_seq_len = int(layer_data["max_seq_len"])

    if text_by_prompt_id is None:
        text_by_prompt_id = collect_prompt_texts(
            str(layer_data["corpus_spec"]),
            {
                pid
                for pid, val in zip(prompt_ids, vals, strict=True)
                if math.isfinite(val) and val > 0
            },
            int(layer_data.get("num_parts", 1)),
        )

    windows: list[Window] = []
    for rank, (val, pid, tpos) in enumerate(zip(vals, prompt_ids, token_pos, strict=True)):
        if not math.isfinite(val) or val <= 0:
            windows.append(
                Window(
                    rank=rank,
                    value=val,
                    prompt_id=pid,
                    token_pos=tpos,
                    rendered="(no activation)",
                    active=False,
                )
            )
            continue

        text = text_by_prompt_id.get(pid)
        if text is None:
            windows.append(
                Window(
                    rank=rank,
                    value=val,
                    prompt_id=pid,
                    token_pos=tpos,
                    rendered=f"(prompt {pid} missing from stream)",
                    active=False,
                )
            )
            continue

        enc = tokenizer(text, truncation=True, max_length=max_seq_len, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        if tpos >= len(ids):
            windows.append(
                Window(
                    rank=rank,
                    value=val,
                    prompt_id=pid,
                    token_pos=tpos,
                    rendered=f"(token position {tpos} out of range for prompt {pid})",
                    active=False,
                )
            )
            continue

        lo = max(0, tpos - window)
        hi = min(len(ids), tpos + window + 1)
        before = tokenizer.decode(ids[lo:tpos])
        target = tokenizer.decode([ids[tpos]])
        after = tokenizer.decode(ids[tpos + 1 : hi])
        windows.append(
            Window(
                rank=rank,
                value=val,
                prompt_id=pid,
                token_pos=tpos,
                rendered=f"{before}<<{target}>>{after}",
                active=True,
            )
        )
    return windows


def format_windows_for_prompt(windows: Sequence[Window]) -> str:
    active_windows = [window for window in windows if window.active]
    return "\n".join(
        f"#{index}: {window.rendered}" for index, window in enumerate(active_windows, start=1)
    )
