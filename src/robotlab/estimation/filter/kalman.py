"""The Kalman measurement update shared by recursive filters.

``kalman_update`` is the gated linear(ized) correction every filter needs:
innovation covariance S = H P Hᵀ + R, normalized innovation squared
NIS = rᵀ S⁻¹ r (chi-square with dim(r) degrees of freedom for a consistent
filter), gain K = P Hᵀ S⁻¹, error-state correction δx = K r, and the
Joseph-form posterior (I − K H) P (I − K H)ᵀ + K R Kᵀ, which stays positive
semidefinite for any gain (Bar-Shalom, Li, and Kirubarajan, *Estimation
with Applications to Tracking and Navigation* (2001), eqs. 5.2.3-18 and
5.2.3-19). The filter applies δx to its own state (adding for vector
blocks, ⊕ for rotations) and, for an error-state filter, resets the
covariance; this module never touches the state itself.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.measurements import readonly_array
from ..interface import Innovation
from .models import LinearizedMeasurement


@dataclass(frozen=True)
class KalmanCorrection:
    """Result of one ``kalman_update``.

    ``delta`` is the state correction K r, None when the gate rejected the
    measurement; ``covariance`` is the posterior when accepted and the
    unchanged prior otherwise.
    """

    innovation: Innovation
    delta: NDArray[np.float64] | None
    covariance: NDArray[np.float64]


def kalman_update(
    covariance: NDArray[np.float64],
    measurement: LinearizedMeasurement,
    gate_squared: float = np.inf,
) -> KalmanCorrection:
    """Gated Kalman correction of ``covariance`` with ``measurement``
    (module docs).

    The measurement is rejected when its NIS exceeds ``gate_squared`` (a
    chi-square quantile, e.g. 25 ≈ 5 σ; ``inf`` disables gating). Raises
    ValueError for mismatched sizes, a nonpositive gate, or a non-finite
    NIS (non-finite residual or singular S).
    """
    P = covariance
    H, R, residual = measurement.jacobian, measurement.covariance, measurement.residual
    if P.shape != (H.shape[1], H.shape[1]):
        raise ValueError(
            f"{measurement.name}: Jacobian has {H.shape[1]} columns, "
            f"covariance is {P.shape}"
        )
    if not gate_squared > 0:
        raise ValueError("Gate must be positive")
    S = H @ P @ H.T + R
    nis = float(residual @ np.linalg.solve(S, residual))
    if not np.isfinite(nis):
        raise ValueError(f"{measurement.name}: non-finite NIS")
    if nis > gate_squared:
        return KalmanCorrection(
            Innovation(measurement.name, nis, len(residual), False), None, P
        )
    K = np.linalg.solve(S, H @ P).T
    A = np.eye(len(P)) - K @ H
    posterior = A @ P @ A.T + K @ R @ K.T
    return KalmanCorrection(
        Innovation(measurement.name, nis, len(residual), True), K @ residual, posterior
    )


def checked_covariance(value: ArrayLike, size: int) -> NDArray[np.float64]:
    """A writable ``size`` × ``size`` copy of ``value``; raises ValueError
    unless it is symmetric positive definite."""
    result = readonly_array(value, (size, size)).copy()
    if (
        not np.allclose(result, result.T, atol=1e-12, rtol=0)
        or np.linalg.eigvalsh(result).min() <= 0
    ):
        raise ValueError("Covariance must be symmetric positive definite")
    return result
