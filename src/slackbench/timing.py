from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import torch


@dataclass
class DuplexUnitTiming:
    prefill: dict[str, Any]
    generated: dict[str, Any] | None
    prefill_host_ms: float
    generate_host_ms: float
    total_wall_ms: float
    prefill_gpu_ms: float
    generate_gpu_ms: float
    total_gpu_ms: float


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_duplex_unit(
    prefill_fn: Callable[[], dict[str, Any]],
    generate_fn: Callable[[], dict[str, Any]],
) -> DuplexUnitTiming:
    """Measure one duplex unit without inserting a synchronization between stages.

    The primary latency is ``total_wall_ms``. CUDA is synchronized once before the
    unit and once after it, so the measured wall time includes all GPU work launched
    by both ``streaming_prefill`` and ``streaming_generate`` while preserving their
    normal back-to-back execution. Per-stage host durations are diagnostic only;
    CUDA event times are the stage-level GPU measurements.
    """
    if not torch.cuda.is_available():
        wall_start = time.perf_counter()
        prefill_start = time.perf_counter()
        prefill = prefill_fn()
        prefill_host_ms = (time.perf_counter() - prefill_start) * 1000.0
        generated: dict[str, Any] | None = None
        generate_host_ms = float("nan")
        if prefill.get("success", False):
            generate_start = time.perf_counter()
            generated = generate_fn()
            generate_host_ms = (time.perf_counter() - generate_start) * 1000.0
        return DuplexUnitTiming(
            prefill=prefill,
            generated=generated,
            prefill_host_ms=prefill_host_ms,
            generate_host_ms=generate_host_ms,
            total_wall_ms=(time.perf_counter() - wall_start) * 1000.0,
            prefill_gpu_ms=float("nan"),
            generate_gpu_ms=float("nan"),
            total_gpu_ms=float("nan"),
        )

    synchronize_cuda()
    unit_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    unit_end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    unit_start.record()

    prefill_host_start = time.perf_counter()
    prefill = prefill_fn()
    prefill_host_ms = (time.perf_counter() - prefill_host_start) * 1000.0
    prefill_end.record()

    generated: dict[str, Any] | None = None
    generate_host_ms = float("nan")
    if prefill.get("success", False):
        generate_host_start = time.perf_counter()
        generated = generate_fn()
        generate_host_ms = (time.perf_counter() - generate_host_start) * 1000.0

    unit_end.record()
    synchronize_cuda()
    total_wall_ms = (time.perf_counter() - wall_start) * 1000.0

    prefill_gpu_ms = float(unit_start.elapsed_time(prefill_end))
    total_gpu_ms = float(unit_start.elapsed_time(unit_end))
    generate_gpu_ms = (
        float(prefill_end.elapsed_time(unit_end)) if prefill.get("success", False) else float("nan")
    )

    return DuplexUnitTiming(
        prefill=prefill,
        generated=generated,
        prefill_host_ms=prefill_host_ms,
        generate_host_ms=generate_host_ms,
        total_wall_ms=total_wall_ms,
        prefill_gpu_ms=prefill_gpu_ms,
        generate_gpu_ms=generate_gpu_ms,
        total_gpu_ms=total_gpu_ms,
    )


def sleep_until(target_monotonic_s: float) -> float:
    """Sleep until target monotonic time and return actual wake time."""
    remaining = target_monotonic_s - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    return time.perf_counter()
