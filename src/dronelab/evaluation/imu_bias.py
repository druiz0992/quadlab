"""Effective IMU bias of what the estimator actually receives (evaluation only).

An estimator's bias states describe the error of the readings it is given:
raw readings, or driver-calibrated ones when a calibration is active. So the
truth to score them against is the effective bias

    b_eff = EMA_τ( R_nominal · delivered − R_true · ideal )      (body axes)

with ``ideal`` the error-free Gazebo reading in the true sensor axes and
``delivered`` the reading after all error models (and calibration), rotated
with the nominal mount as the estimator does. The exponential moving average
(time constant ``time_constant_s``) removes white noise; what remains is
offset, drift, and signal-dependent errors (scale, misalignment, mount
error) at the current motion, which is what the filter must absorb.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


class EffectiveImuBias:
    """Running effective accelerometer and gyro bias (see module docs)."""

    def __init__(
        self,
        nominal_mount: Rotation,
        true_mount: Rotation,
        time_constant_s: float = 1.0,
    ) -> None:
        self._nominal, self._true = nominal_mount, true_mount
        self._tau = time_constant_s
        self._bias: NDArray[np.float64] | None = None
        self._previous_ns: int | None = None

    def update(
        self,
        sim_time_ns: int,
        ideal_force: ArrayLike,
        ideal_rate: ArrayLike,
        delivered_force: ArrayLike,
        delivered_rate: ArrayLike,
    ) -> None:
        error = np.concatenate(
            (
                self._nominal.apply(delivered_force) - self._true.apply(ideal_force),
                self._nominal.apply(delivered_rate) - self._true.apply(ideal_rate),
            )
        )
        if self._bias is None:
            self._bias = error
        else:
            dt = max(sim_time_ns - self._previous_ns, 0) * 1e-9
            self._bias += (1 - math.exp(-dt / self._tau)) * (error - self._bias)
        self._previous_ns = sim_time_ns

    def current(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(accelerometer, gyro) effective bias in body axes; zeros before data."""
        if self._bias is None:
            return np.zeros(3), np.zeros(3)
        return self._bias[:3].copy(), self._bias[3:].copy()
