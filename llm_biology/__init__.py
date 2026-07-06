"""Local tooling for Qwen biology attribution graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_biology.attribution.attribution import BiologyAttributionRunner
    from llm_biology.viewer.server import serve

__all__ = ["BiologyAttributionRunner", "serve"]


def __getattr__(name: str) -> Any:
    if name == "BiologyAttributionRunner":
        from llm_biology.attribution.attribution import BiologyAttributionRunner

        return BiologyAttributionRunner
    if name == "serve":
        from llm_biology.viewer.server import serve

        return serve
    raise AttributeError(name)
