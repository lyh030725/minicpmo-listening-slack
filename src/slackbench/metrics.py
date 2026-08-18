from __future__ import annotations

from typing import Iterable
import math

import numpy as np


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
