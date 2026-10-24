"""Summary statistics of error magnitudes, shared by the evaluators."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def error_statistics(values: list[float] | NDArray[np.float64]) -> dict[str, Any]:
    """RMS, mean, 95th percentile, maximum, and final value of error magnitudes."""
    array = np.asarray(values, float)
    if array.size == 0:
        return dict(count=0)
    return dict(
        count=int(array.size),
        rmse=float(np.sqrt(np.mean(array**2))),
        mean=float(np.mean(array)),
        p95=float(np.percentile(array, 95)),
        max=float(np.max(array)),
        final=float(array[-1]),
    )
