"""Compatibility imports for circuit-tracer runtime drift."""

from __future__ import annotations

from typing import Any

load_transcoder: Any
try:
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder
except ImportError:
    from circuit_tracer.transcoder.single_layer_transcoder import (
        load_relu_transcoder as load_transcoder,
    )

__all__ = ["load_transcoder"]
