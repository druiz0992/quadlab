"""Estimation accuracy and consistency against simulator truth.

For each estimate (decimated to ``rate_hz``), truth is interpolated to the
estimate's own timestamp (``TruthHistory``) and errors are formed in the
error-state convention of the GPS/IMU EKF (truth = estimate ⊕ error):

    δp = p_true − p_est,  δv = v_true − v_est,  δθ = Log(R_estᵀ R_true)

(right attitude error, rad), plus bias errors when true biases are known.
Consistency uses the normalized estimation error squared
NEES = eᵀ P⁻¹ e over the states with known truth (9 navigation states, or
all 15 with biases) and, when an estimator reports it, the normalized
innovation squared (NIS) of its measurement updates. For a consistent
filter both are χ²-distributed with mean equal to their degrees of freedom;
we report the mean and the fraction of samples inside the two-sided 95 %
χ² interval (Bar-Shalom, Li & Kirubarajan 2001, ch. 5.4).
"""

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from ..core.measurements import EstimatedState
from .truth import TruthHistory

BiasSource = Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]]


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


def chi2_consistency(values: list[float], dof: int) -> dict[str, Any]:
    """Mean and fraction inside the two-sided 95 % χ²(dof) interval."""
    if not values:
        return dict(dof=dof, count=0)
    array = np.asarray(values, float)
    low, high = chi2.ppf([0.025, 0.975], dof)
    return dict(
        dof=dof,
        count=int(array.size),
        mean=float(np.mean(array)),
        expected_mean=float(dof),
        fraction_inside_95=float(np.mean((array >= low) & (array <= high))),
        interval_95=[float(low), float(high)],
    )


class EstimationMetrics:
    """Online estimation scoring; feed truth with ``add_truth``.

    ``true_biases`` (optional) returns the simulator's current accelerometer
    and gyro biases in the estimator's bias axes, for evaluation only.
    ``start_s`` skips initial convergence (seconds after the first estimate).
    """

    def __init__(
        self,
        truth: TruthHistory,
        *,
        rate_hz: float = 50.0,
        start_s: float = 0.0,
        true_biases: BiasSource | None = None,
    ) -> None:
        self._truth = truth
        self._period_ns = round(1e9 / rate_hz)
        self._start_ns = round(start_s * 1e9)
        self._true_biases = true_biases
        self._pending: deque[tuple[int, EstimatedState]] = deque()
        self._first_ns: int | None = None
        self._last_ns: int | None = None
        self._errors: dict[str, list[float]] = {
            key: []
            for key in (
                "position_m",
                "horizontal_m",
                "vertical_m",
                "velocity_mps",
                "attitude_deg",
                "yaw_deg",
                "accel_bias_mps2",
                "gyro_bias_rad_s",
            )
        }
        self._axis_sq = np.zeros(3)
        self._nees: list[float] = []
        self._nees_dof = 15 if true_biases is not None else 9
        self._nis: dict[str, list[float]] = {}
        self._nis_dof: dict[str, int] = {}

    def add_estimate(self, sim_time_ns: int, state: EstimatedState) -> None:
        """Queue an estimate; it is scored once truth covers its time."""
        if self._first_ns is None:
            self._first_ns = sim_time_ns
        if self._last_ns is not None and sim_time_ns - self._last_ns < self._period_ns:
            return
        if sim_time_ns - self._first_ns < self._start_ns:
            return
        self._last_ns = sim_time_ns
        self._pending.append((sim_time_ns, state))

    def add_nis(self, name: str, value: float, dof: int = 3) -> None:
        """Record one normalized innovation squared (``dof`` measurement terms)."""
        self._nis.setdefault(name, []).append(float(value))
        self._nis_dof[name] = dof

    def process(self) -> list[dict[str, float]]:
        """Score queued estimates that truth now covers; return their errors."""
        scored = []
        latest = self._truth.latest_ns
        while self._pending and latest is not None and self._pending[0][0] <= latest:
            stamp, state = self._pending.popleft()
            truth = self._truth.at(stamp)
            if truth is not None:
                scored.append(self._score(stamp, state, truth))
        return scored

    def _score(self, stamp, state, truth) -> dict[str, float]:
        dp = truth.position_m - state.position_m
        dv = truth.velocity_mps - state.velocity_mps
        r_est, r_true = Rotation.from_quat(state.quaternion_xyzw), Rotation.from_quat(
            truth.quaternion_xyzw
        )
        dtheta = (r_est.inv() * r_true).as_rotvec()
        yaw = r_true.as_euler("ZYX")[0] - r_est.as_euler("ZYX")[0]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        error = np.concatenate((dp, dv, dtheta))
        if self._true_biases is not None:
            accel_bias, gyro_bias = self._true_biases()
            db_a = np.asarray(accel_bias) - state.accel_bias_mps2
            db_g = np.asarray(gyro_bias) - state.gyro_bias_rad_s
            error = np.concatenate((error, db_a, db_g))
            self._errors["accel_bias_mps2"].append(float(np.linalg.norm(db_a)))
            self._errors["gyro_bias_rad_s"].append(float(np.linalg.norm(db_g)))
        covariance = np.asarray(state.covariance)[: error.size, : error.size]
        nees = float(error @ np.linalg.solve(covariance, error))
        self._nees.append(nees)
        self._axis_sq += dp**2
        values = dict(
            position_m=float(np.linalg.norm(dp)),
            horizontal_m=float(np.hypot(dp[0], dp[1])),
            vertical_m=float(abs(dp[2])),
            velocity_mps=float(np.linalg.norm(dv)),
            attitude_deg=float(np.degrees(np.linalg.norm(dtheta))),
            yaw_deg=float(np.degrees(abs(yaw))),
        )
        for key, value in values.items():
            self._errors[key].append(value)
        return dict(values, nees=nees, sim_time_s=stamp * 1e-9)

    def summary(self) -> dict[str, Any]:
        count = len(self._nees)
        return dict(
            samples=count,
            rate_hz=1e9 / self._period_ns,
            skipped_initial_s=self._start_ns * 1e-9,
            errors={k: error_statistics(v) for k, v in self._errors.items() if v},
            position_rmse_axes_m=(
                np.sqrt(self._axis_sq / count).tolist() if count else None
            ),
            nees=chi2_consistency(self._nees, self._nees_dof),
            nis={
                name: chi2_consistency(v, self._nis_dof[name])
                for name, v in self._nis.items()
            },
            unscored_at_close=len(self._pending),
        )
