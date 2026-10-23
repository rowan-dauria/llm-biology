"""Minimal client for OpenAI-compatible chat-completions endpoints.

Works with OpenRouter (default), Gemini's OpenAI-compatible endpoint, OpenAI,
or a local vLLM server. Configure with ``LLM_API_BASE``, ``LLM_API_KEY`` and
``LLM_MODEL`` or the matching CLI flags.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

Message = dict[str, str]


class ChatClient(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        api_base: str = DEFAULT_API_BASE,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        timeout: float = 180.0,
        retries: int = 4,
    ):
        self.model = model
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_env(
        cls, model: str | None = None, api_base: str | None = None
    ) -> OpenAICompatibleClient:
        model = model or os.environ.get("LLM_MODEL")
        if not model:
            raise ValueError("set --llm-model or LLM_MODEL (e.g. an OpenRouter model slug)")
        return cls(
            model=model,
            api_base=api_base or os.environ.get("LLM_API_BASE", DEFAULT_API_BASE),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
        )

    def complete(self, messages: list[Message]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload: dict[str, Any] = json.load(response)
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt == self.retries - 1:
                    detail = error.read().decode(errors="replace")[:500]
                    raise RuntimeError(f"LLM request failed ({error.code}): {detail}") from error
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
            time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")


def parse_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a reply, tolerating prose and code fences."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON object in LLM reply: {text[:300]!r}")
