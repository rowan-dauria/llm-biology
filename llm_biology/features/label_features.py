"""Auto-label a filtered subset of transcoder features with an LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from .windows import (
        FeatureSummary,
        Window,
        active_prompt_ids,
        collect_prompt_texts,
        get_windows,
        load_tokenizer,
        select_features,
    )
except ImportError:
    from windows import (  # type: ignore[no-redef]
        FeatureSummary,
        Window,
        active_prompt_ids,
        collect_prompt_texts,
        get_windows,
        load_tokenizer,
        select_features,
    )

DEFAULT_DIR = Path(__file__).parent.parent / "data" / "feature_labels"
DEFAULT_WINDOW = 10
DEFAULT_TOP_N = 200
DEFAULT_DIVERSITY = 4
DEFAULT_CONCURRENCY = 8
DEFAULT_PROVIDER = "transformers"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_TRANSFORMERS_MODEL = "google/gemma-3-12b-it"
DEFAULT_MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """You are explaining the behavior of a neuron in a neural network. Your response should be a very concise explanation (1-6 words) that captures what the neuron detects or predicts by finding patterns in lists.

To determine the explanation, you are given two lists:

- MAX_ACTIVATING_TOKENS, which are the top activating tokens in the top activating texts.
- TOP_ACTIVATING_TEXTS, which are top activating texts.

You should look for a pattern by trying the following methods in order. Once you find a pattern, stop and return that pattern. Do not proceed to the later methods.
Method 1: Look at MAX_ACTIVATING_TOKENS. If they share something specific in common, or are all the same token or a variation of the same token (like different cases or conjugations), respond with that token.
Method 2: Look at TOP_ACTIVATING_TEXTS and make a best guess by describing the broad theme or context, ignoring the max activating tokens.

Rules:
- Keep your explanation extremely concise (1-6 words, mostly 1-3 words).
- Do not add unnecessary phrases like "words related to", "concepts related to", or "variations of the word".
- Do not mention "tokens" or "patterns" in your explanation.
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".
- If you absolutely cannot make any guesses, return the first token in MAX_ACTIVATING_TOKENS.

Respond with strict JSON only: {"label": "...", "rationale": "..."}.
The label is the concise explanation. The rationale is one sentence that names the method number used and the reason for the label."""

_TRIGGER_RE = re.compile(r"<<(.*?)>>", re.DOTALL)


@dataclass(slots=True)
class LabelOutcome:
    summary: FeatureSummary
    record: dict[str, Any] | None
    error: str | None = None


def _output_path(layer: int) -> Path:
    return DEFAULT_DIR / f"layer_{layer}.jsonl"


def _load_topk(layer: int, topk_dir: Path) -> dict[str, Any]:
    topk_path = topk_dir / f"topk_layer_{layer}.pt"
    return torch.load(topk_path, weights_only=False)


def _read_existing_features(path: Path) -> set[int]:
    if not path.exists():
        return set()

    existing: set[int] = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            existing.add(int(obj["feature"]))
    return existing


def _build_user_prompt(layer: int, summary: FeatureSummary, windows: list[Window]) -> str:
    active_windows = [window for window in windows if window.active]
    max_tokens = []
    top_texts = []
    for window in active_windows:
        match = _TRIGGER_RE.search(window.rendered)
        token = match.group(1) if match else ""
        max_tokens.append(f"#{len(max_tokens) + 1}: {token}")
        top_texts.append(f"#{len(top_texts) + 1}: {window.rendered}")

    lines = [
        f"Layer {layer} feature {summary.feature}",
        f"Top windows: max_activation={summary.max_activation:.3f} distinct_prompts={summary.n_distinct_prompts}",
        "",
        "MAX_ACTIVATING_TOKENS:",
        "\n".join(max_tokens) if max_tokens else "(none)",
        "",
        "TOP_ACTIVATING_TEXTS:",
    ]
    lines.append("\n".join(top_texts) if top_texts else "(no activating windows)")
    return "\n".join(lines)


def _validate_response(text: str) -> dict[str, str]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Response was not a JSON object")

    label = payload.get("label")
    rationale = payload.get("rationale")
    if not isinstance(label, str) or not isinstance(rationale, str):
        raise TypeError("Response must contain string 'label' and 'rationale'")

    label = label.strip()
    rationale = rationale.strip()
    if not label:
        raise ValueError("Label is empty")
    if len(label.split()) > 6:
        raise ValueError("Label exceeds 6 words")
    if not rationale:
        raise ValueError("Rationale is empty")
    if "\n" in rationale:
        raise ValueError("Rationale must be one sentence")
    return {"label": label, "rationale": rationale}


_TRANSFORMERS_STATE: dict[str, Any] = {}
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_block(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else text


def _load_transformers(model_id: str) -> dict[str, Any]:
    if _TRANSFORMERS_STATE.get("model_id") == model_id:
        return _TRANSFORMERS_STATE

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use AutoTokenizer (text-only) rather than AutoProcessor: Gemma 4's processor
    # pulls in Gemma4VideoProcessor which requires torchvision, and torchvision is
    # deliberately not installed in qwen-sae (conflicts with this torch build).
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _TRANSFORMERS_STATE.clear()
    _TRANSFORMERS_STATE.update(
        {
            "model_id": model_id,
            "tokenizer": tokenizer,
            "model": model,
            "device": next(model.parameters()).device,
        }
    )
    return _TRANSFORMERS_STATE


def _build_chat_text(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate_batch(state: dict[str, Any], prompts: list[str]) -> list[str]:
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]
    enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    new_tokens = out[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _build_record(
    layer: int,
    summary: FeatureSummary,
    payload: dict[str, str],
    source_tag: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "layer": layer,
        "feature": summary.feature,
        "label": payload["label"],
        "rationale": payload["rationale"],
        "max_activation": summary.max_activation,
        "n_distinct_prompts": summary.n_distinct_prompts,
    }
    if source_tag:
        record["source"] = source_tag
    return record


def _run_transformers_labels(
    model_id: str,
    layer: int,
    payloads: list[tuple[FeatureSummary, str]],
    output_path: Path,
    batch_size: int,
    *,
    source_tag: str | None = None,
) -> None:
    from tqdm import tqdm  # type: ignore

    state = _load_transformers(model_id)
    tokenizer = state["tokenizer"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    reminder = "\n\nReply with valid JSON only."

    def _label_one_text(user_prompt: str) -> dict[str, str]:
        prompts = [_build_chat_text(tokenizer, SYSTEM_PROMPT, user_prompt)]
        text = _generate_batch(state, prompts)[0]
        return _validate_response(_extract_json_block(text).strip())

    with output_path.open("a", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(payloads), batch_size), desc="batches"):
            chunk = payloads[start : start + batch_size]
            chat_texts = [_build_chat_text(tokenizer, SYSTEM_PROMPT, up) for _, up in chunk]
            try:
                outputs = _generate_batch(state, chat_texts)
            except Exception as exc:
                for summary, _ in chunk:
                    failures.append(f"feature {summary.feature}: batch failure: {exc}")
                continue

            for (summary, user_prompt), raw in zip(chunk, outputs, strict=True):
                payload: dict[str, str] | None = None
                try:
                    payload = _validate_response(_extract_json_block(raw).strip())
                except Exception:
                    try:
                        payload = _label_one_text(user_prompt + reminder)
                    except Exception as exc:
                        failures.append(f"feature {summary.feature}: {exc}")
                        continue

                record = _build_record(layer, summary, payload, source_tag)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    if failures:
        raise RuntimeError("Some features failed to label: " + "; ".join(failures))


async def _anthropic_complete(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str,
) -> str:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("Anthropic SDK is not installed") from exc

    client = AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
    )

    chunks = []
    for block in response.content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type != "text":
            continue
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
        chunks.append(text)
    return "".join(chunks)


async def _openai_complete(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("OpenAI SDK is not installed") from exc

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    choice = response.choices[0]
    message = choice.message.content
    if message is None:
        raise RuntimeError("OpenAI response did not include message content")
    return message


def _post_ollama_chat(host: str, payload: dict[str, Any]) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed for {url}") from exc

    obj = json.loads(body)
    error = obj.get("error")
    if error:
        raise RuntimeError(f"Ollama error: {error}")
    message = obj.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama response did not include a message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama response did not include message content")
    return content


async def _ollama_complete(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    host: str,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 256,
        },
    }
    return await asyncio.to_thread(_post_ollama_chat, host, payload)


async def _call_model(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None,
    ollama_host: str,
) -> str:
    if provider == "anthropic":
        if api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return await _anthropic_complete(model, system_prompt, user_prompt, api_key=api_key)
    if provider == "openai":
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return await _openai_complete(model, system_prompt, user_prompt, api_key=api_key)
    if provider == "ollama":
        return await _ollama_complete(model, system_prompt, user_prompt, host=ollama_host)
    raise ValueError(f"Unknown provider: {provider!r}")


async def _label_one(
    provider: str,
    model: str,
    api_key: str | None,
    summary: FeatureSummary,
    user_prompt: str,
    ollama_host: str,
) -> LabelOutcome:
    reminder = "\n\nReply with valid JSON only."
    for attempt in range(2):
        prompt = user_prompt if attempt == 0 else f"{user_prompt}{reminder}"
        try:
            raw = await _call_model(
                provider,
                model,
                SYSTEM_PROMPT,
                prompt,
                api_key=api_key,
                ollama_host=ollama_host,
            )
            payload = _validate_response(raw.strip())
            return LabelOutcome(summary=summary, record=payload)
        except Exception as exc:
            if attempt == 0:
                continue
            return LabelOutcome(summary=summary, record=None, error=str(exc))
    return LabelOutcome(summary=summary, record=None, error="unreachable")


async def _run_api_labels(
    provider: str,
    model: str,
    api_key: str | None,
    layer: int,
    payloads: list[tuple[FeatureSummary, str]],
    output_path: Path,
    concurrency: int,
    ollama_host: str,
    *,
    source_tag: str | None = None,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    async def run_one(summary: FeatureSummary, user_prompt: str) -> LabelOutcome:
        async with semaphore:
            return await _label_one(
                provider,
                model,
                api_key,
                summary,
                user_prompt,
                ollama_host,
            )

    tasks = [
        asyncio.create_task(run_one(summary, user_prompt)) for summary, user_prompt in payloads
    ]
    with output_path.open("a", encoding="utf-8") as handle:
        from tqdm import tqdm  # type: ignore

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="features"):
            outcome = await task
            if outcome.record is None:
                failures.append(f"feature {outcome.summary.feature}: {outcome.error}")
                continue
            record = _build_record(layer, outcome.summary, outcome.record, source_tag)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    if failures:
        raise RuntimeError("Some features failed to label: " + "; ".join(failures))


def _build_payloads(
    layer: int,
    layer_data: dict[str, Any],
    summaries: list[FeatureSummary],
    *,
    tokenizer: Any,
    corpus_spec: str,
) -> list[tuple[FeatureSummary, str]]:
    needed_prompt_ids: set[int] = set()
    for summary in summaries:
        needed_prompt_ids.update(summary.active_prompt_ids)

    text_by_prompt_id = collect_prompt_texts(
        corpus_spec, needed_prompt_ids, int(layer_data.get("num_parts", 1))
    )
    payloads: list[tuple[FeatureSummary, str]] = []
    for summary in summaries:
        windows = get_windows(
            layer_data,
            summary.feature,
            tokenizer,
            window=DEFAULT_WINDOW,
            text_by_prompt_id=text_by_prompt_id,
        )
        payloads.append((summary, _build_user_prompt(layer, summary, windows)))
    return payloads


def _print_dry_run(layer: int, payloads: list[tuple[FeatureSummary, str]]) -> None:
    print(SYSTEM_PROMPT)
    for summary, user_prompt in payloads:
        print()
        print(f"=== layer {layer} feature {summary.feature} ===")
        print(user_prompt)


async def _async_main(args: argparse.Namespace) -> None:
    layer_data = _load_topk(args.layer, args.topk_dir)
    summaries = select_features(
        layer_data,
        top_n=args.top_n,
        diversity=args.diversity,
    )
    if not summaries:
        print(f"No features in layer {args.layer} passed the top_n/diversity filter.")
        return

    output_path = _output_path(args.layer)
    existing = _read_existing_features(output_path)
    summaries = [summary for summary in summaries if summary.feature not in existing]
    if not summaries:
        print(f"All selected features for layer {args.layer} are already present in {output_path}.")
        return

    tokenizer = load_tokenizer(str(layer_data["model_id"]))
    corpus_spec = args.corpus_spec or str(layer_data["corpus_spec"])
    payloads = _build_payloads(
        args.layer,
        layer_data,
        summaries,
        tokenizer=tokenizer,
        corpus_spec=corpus_spec,
    )
    if args.dry_run:
        _print_dry_run(args.layer, payloads)
        return

    if args.provider == "transformers":
        model = args.model or DEFAULT_TRANSFORMERS_MODEL
        _run_transformers_labels(
            model,
            args.layer,
            payloads,
            output_path,
            args.concurrency,
            source_tag="layer_batch",
        )
        return

    api_key = None
    if args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    if args.provider not in {"ollama", "transformers"} and not api_key:
        raise RuntimeError(f"{args.provider.upper()}_API_KEY is not set")

    model = args.model or (
        DEFAULT_ANTHROPIC_MODEL
        if args.provider == "anthropic"
        else DEFAULT_OPENAI_MODEL
        if args.provider == "openai"
        else DEFAULT_OLLAMA_MODEL
    )
    ollama_host = args.ollama_host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST

    await _run_api_labels(
        args.provider,
        model,
        api_key,
        args.layer,
        payloads,
        output_path,
        args.concurrency,
        ollama_host,
        source_tag="layer_batch",
    )


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _summary_for_feature(layer_data: dict[str, Any], feature: int) -> FeatureSummary | None:
    active = active_prompt_ids(layer_data, feature)
    if not active:
        return None
    vals = layer_data["topk_vals"][:, feature].tolist()
    max_activation = 0.0
    for raw in vals:
        value = float(raw)
        if math.isfinite(value) and value > max_activation:
            max_activation = value
    return FeatureSummary(
        feature=feature,
        max_activation=max_activation,
        n_distinct_prompts=len(active),
        active_prompt_ids=active,
    )


def label_features_subset(
    layer: int,
    feature_indices: Iterable[int],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    ollama_host: str | None = None,
    corpus_spec_override: str | None = None,
    source_tag: str = "on_demand",
    topk_dir: Path,
) -> dict[str, int]:
    """Label an explicit list of ``(layer, feature)`` pairs.

    Uses the saved ``topk_layer_<L>.pt`` windows. Features already present in
    ``data/feature_labels/layer_<L>.jsonl`` are skipped. The graph-surfaced
    diversity filter is *not* applied — features with sparse activations are
    still labelled but emit a one-line caveat.

    Returns counts: ``{requested, skipped_existing, no_active_windows,
    written, failed}``.
    """

    requested = list(dict.fromkeys(int(feature) for feature in feature_indices))
    output_path = _output_path(layer)
    existing = _read_existing_features(output_path)

    pending = [feature for feature in requested if feature not in existing]
    counts: dict[str, int] = {
        "requested": len(requested),
        "skipped_existing": len(requested) - len(pending),
        "no_active_windows": 0,
        "written": 0,
        "failed": 0,
    }
    if not pending:
        return counts

    layer_data = _load_topk(layer, topk_dir=topk_dir)
    summaries: list[FeatureSummary] = []
    for feature in pending:
        summary = _summary_for_feature(layer_data, feature)
        if summary is None:
            counts["no_active_windows"] += 1
            print(f"[CAVEAT] layer {layer} feature {feature}: no active windows; skipped")
            continue
        if summary.n_distinct_prompts < DEFAULT_DIVERSITY:
            print(
                f"[CAVEAT] layer {layer} feature {feature}: "
                f"{summary.n_distinct_prompts} active prompts "
                f"(< diversity threshold {DEFAULT_DIVERSITY}); labelling anyway"
            )
        summaries.append(summary)

    if not summaries:
        return counts

    tokenizer = load_tokenizer(str(layer_data["model_id"]))
    corpus_spec = corpus_spec_override or str(layer_data["corpus_spec"])
    payloads = _build_payloads(
        layer,
        layer_data,
        summaries,
        tokenizer=tokenizer,
        corpus_spec=corpus_spec,
    )

    before_count = _count_lines(output_path)

    try:
        if provider == "transformers":
            run_model = model or DEFAULT_TRANSFORMERS_MODEL
            _run_transformers_labels(
                run_model,
                layer,
                payloads,
                output_path,
                concurrency,
                source_tag=source_tag,
            )
        else:
            api_key = None
            if provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            if provider not in {"ollama"} and not api_key:
                raise RuntimeError(f"{provider.upper()}_API_KEY is not set")
            run_model = model or (
                DEFAULT_ANTHROPIC_MODEL
                if provider == "anthropic"
                else DEFAULT_OPENAI_MODEL
                if provider == "openai"
                else DEFAULT_OLLAMA_MODEL
            )
            host = ollama_host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
            asyncio.run(
                _run_api_labels(
                    provider,
                    run_model,
                    api_key,
                    layer,
                    payloads,
                    output_path,
                    concurrency,
                    host,
                    source_tag=source_tag,
                )
            )
    except RuntimeError as exc:
        print(f"[WARN] layer {layer}: {exc}")

    after_count = _count_lines(output_path)
    counts["written"] = max(0, after_count - before_count)
    counts["failed"] = len(payloads) - counts["written"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--topk-dir",
        type=Path,
        required=True,
        help="Directory holding topk_layer_<L>.pt windows (e.g. data/feature_topk/150k-pile).",
    )
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--diversity", type=int, default=DEFAULT_DIVERSITY)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--provider",
        choices=("transformers", "ollama", "openai", "anthropic"),
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider default model.",
    )
    parser.add_argument(
        "--corpus_spec",
        default=None,
        help="Override the corpus spec from the saved top-K file.",
    )
    parser.add_argument(
        "--ollama_host",
        default=None,
        help=f"Ollama host URL; defaults to OLLAMA_HOST or {DEFAULT_OLLAMA_HOST}.",
    )
    parser.add_argument("--dry_run", action="store_true")
    asyncio.run(_async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
