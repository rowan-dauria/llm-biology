"""Local tooling for Qwen biology attribution graphs."""

from __future__ import annotations

__all__ = ["BiologyAttributionRunner", "serve"]


def __getattr__(name: str):
    if name == "BiologyAttributionRunner":
        from llm_biology.attribution.attribution import BiologyAttributionRunner

        return BiologyAttributionRunner
    if name == "serve":
        from llm_biology.viewer.server import serve

        return serve
    raise AttributeError(name)
