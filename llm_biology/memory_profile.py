"""Opt-in memory profiling helpers for local biology-server debugging."""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_WARNED_MEMORY_PROFILER = False


def memory_profile_enabled() -> bool:
    """Return whether ``BIOLOGY_SERVER_PROFILE_MEMORY`` requests profiling output."""
    return os.getenv("BIOLOGY_SERVER_PROFILE_MEMORY", "").lower() in _TRUE_VALUES


def _maybe_collect() -> None:
    """Force a GC pass when ``BIOLOGY_SERVER_PROFILE_GC`` is set, for cleaner RSS readings."""
    if os.getenv("BIOLOGY_SERVER_PROFILE_GC", "").lower() in _TRUE_VALUES:
        gc.collect()


def current_rss_mib() -> float | None:
    """Return current RSS in MiB using memory-profiler when available."""
    global _WARNED_MEMORY_PROFILER
    try:
        from memory_profiler import memory_usage

        value = memory_usage(
            -1,
            interval=0.01,
            timeout=0.05,
            max_usage=True,
            include_children=True,
        )
        if isinstance(value, list):
            value = value[0]
        return float(value)
    except Exception as exc:  # pragma: no cover - fallback is for ad hoc debugging
        if not _WARNED_MEMORY_PROFILER:
            print(f"[MEM] memory_profiler unavailable ({exc!r}); using maxrss fallback")
            _WARNED_MEMORY_PROFILER = True

    try:
        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return maxrss / (1024 * 1024)
        return maxrss / 1024
    except Exception:
        return None


def memory_checkpoint(label: str) -> None:
    """Print current RSS tagged with ``label`` if memory profiling is enabled; no-op otherwise."""
    if not memory_profile_enabled():
        return
    _maybe_collect()
    rss = current_rss_mib()
    if rss is None:
        print(f"[MEM] {label}: rss=?")
    else:
        print(f"[MEM] {label}: rss={rss:.1f} MiB")


@contextmanager
def memory_scope(label: str) -> Iterator[None]:
    """Log RSS before/after the wrapped block, tagged with ``label``, when profiling is enabled."""
    if not memory_profile_enabled():
        yield
        return

    _maybe_collect()
    start = current_rss_mib()
    start_time = time.time()
    if start is None:
        print(f"[MEM] {label}: start rss=?")
    else:
        print(f"[MEM] {label}: start rss={start:.1f} MiB")
    try:
        yield
    finally:
        _maybe_collect()
        end = current_rss_mib()
        elapsed = time.time() - start_time
        if start is None or end is None:
            print(f"[MEM] {label}: end rss=? elapsed={elapsed:.2f}s")
        else:
            print(
                f"[MEM] {label}: end rss={end:.1f} MiB "
                f"delta={end - start:+.1f} MiB elapsed={elapsed:.2f}s"
            )


def memory_profile_call(label: str, func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run ``func`` while sampling peak RSS when profiling is enabled."""
    if not memory_profile_enabled():
        return func(*args, **kwargs)

    try:
        from memory_profiler import memory_usage

        _maybe_collect()
        start = current_rss_mib()
        start_time = time.time()
        print(
            f"[MEM] {label}: call start rss={start:.1f} MiB"
            if start is not None
            else f"[MEM] {label}: call start rss=?"
        )
        samples, retval = memory_usage(
            (func, args, kwargs),
            interval=0.05,
            retval=True,
            include_children=True,
            max_usage=False,
        )
        end = current_rss_mib()
        peak = max(float(sample) for sample in samples) if samples else None
        elapsed = time.time() - start_time
        if start is None or end is None or peak is None:
            print(f"[MEM] {label}: call end rss=? peak=? elapsed={elapsed:.2f}s")
        else:
            print(
                f"[MEM] {label}: call end rss={end:.1f} MiB "
                f"delta={end - start:+.1f} MiB peak={peak:.1f} MiB "
                f"peak_delta={peak - start:+.1f} MiB elapsed={elapsed:.2f}s"
            )
        return retval
    except Exception as exc:  # pragma: no cover - ad hoc profiler fallback
        print(f"[MEM] {label}: memory_profile_call fallback ({exc!r})")
        with memory_scope(label):
            return func(*args, **kwargs)
