"""Estimation accuracy and consistency against simulator truth.

For each estimate (decimated to ``rate_hz``), truth is interpolated to the
estimate's own timestamp (``TruthHistory``) and errors are formed in the
error-state convention of the GPS/IMU EKF (truth = estimate ⊕ error):

    δp = p_true − p_est,  δv = v_true − v_est,  δθ = Log(R_estᵀ R_true)

(right attitude error, rad), plus bias errors when true biases are known.
Only the blocks an estimate contains are scored (``EstimatedState`` may
omit velocity, attitude, position, or biases). Consistency uses the
normalized estimation error squared NEES = eᵀ P⁻¹ e over the present
states with known truth (9 navigation states, or all 15 with biases; P is
the reported covariance restricted to them), when a covariance is
reported, and, when an estimator reports it, the normalized
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


def _attitude_errors(
    estimated_xyzw: NDArray[np.float64], true_xyzw: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Right attitude error Log(R_estᵀ R_true) (rad) and wrapped yaw error."""
    r_est, r_true = Rotation.from_quat(estimated_xyzw), Rotation.from_quat(true_xyzw)
    yaw = r_true.as_euler("ZYX")[0] - r_est.as_euler("ZYX")[0]
    return (r_est.inv() * r_true).as_rotvec(), float(
        (yaw + np.pi) % (2 * np.pi) - np.pi
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
        self._scored = 0
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
        self._position_count = 0
        # NEES samples by degrees of freedom (the blocks scored can change,
        # e.g. attitude-only before a position fix).
        self._nees: dict[int, list[float]] = {}
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
                self._scored += 1
        return scored

    def _score(self, stamp, state, truth) -> dict[str, float]:
        """Errors of the present blocks; NEES over those with known truth."""
        errors: dict[str, NDArray[np.float64]] = {}
        values: dict[str, float] = {}
        if state.position_m is not None:
            dp = truth.position_m - state.position_m
            errors["position_m"] = dp
            self._axis_sq += dp**2
            self._position_count += 1
            values.update(
                position_m=float(np.linalg.norm(dp)),
                horizontal_m=float(np.hypot(dp[0], dp[1])),
                vertical_m=float(abs(dp[2])),
            )
        if state.velocity_mps is not None:
            dv = truth.velocity_mps - state.velocity_mps
            errors["velocity_mps"] = dv
            values["velocity_mps"] = float(np.linalg.norm(dv))
        if state.quaternion_xyzw is not None:
            dtheta, yaw = _attitude_errors(state.quaternion_xyzw, truth.quaternion_xyzw)
            errors["quaternion_xyzw"] = dtheta
            values.update(
                attitude_deg=float(np.degrees(np.linalg.norm(dtheta))),
                yaw_deg=float(np.degrees(abs(yaw))),
            )
        values.update(self._bias_errors(state, errors))
        for key, value in values.items():
            self._errors[key].append(value)
        nees = self._nees_of(state, errors)
        result = dict(values, sim_time_s=stamp * 1e-9)
        if nees is not None:
            result["nees"] = nees
        return result

    def _bias_errors(self, state, errors) -> dict[str, float]:
        """Bias error norms for estimated biases, when true biases are known."""
        if self._true_biases is None:
            return {}
        values = {}
        for name, true in zip(
            ("accel_bias_mps2", "gyro_bias_rad_s"), self._true_biases()
        ):
            estimated = getattr(state, name)
            if estimated is not None:
                errors[name] = np.asarray(true) - estimated
                values[name] = float(np.linalg.norm(errors[name]))
        return values

    def _nees_of(self, state, errors) -> float | None:
        """NEES over the blocks with known truth, from the reported
        covariance restricted to them (marginal); None without one."""
        if state.covariance is None or not errors:
            return None
        rows = np.concatenate(
            [np.arange(3) + 3 * state.blocks.index(name) for name in errors]
        )
        error = np.concatenate(list(errors.values()))
        covariance = np.asarray(state.covariance)[np.ix_(rows, rows)]
        nees = float(error @ np.linalg.solve(covariance, error))
        self._nees.setdefault(error.size, []).append(nees)
        return nees

    def summary(self) -> dict[str, Any]:
        positions = self._position_count
        return dict(
            samples=self._scored,
            rate_hz=1e9 / self._period_ns,
            skipped_initial_s=self._start_ns * 1e-9,
            errors={k: error_statistics(v) for k, v in self._errors.items() if v},
            position_rmse_axes_m=(
                np.sqrt(self._axis_sq / positions).tolist() if positions else None
            ),
            nees=self._nees_summary(),
            nis={
                name: chi2_consistency(v, self._nis_dof[name])
                for name, v in self._nis.items()
            },
            unscored_at_close=len(self._pending),
        )

    def _nees_summary(self) -> dict[str, Any]:
        """χ² statistics of NEES: one dof (the usual case, 9 or 15), or one
        entry per dof under ``by_dof`` when the scored blocks changed."""
        if len(self._nees) <= 1:
            default = 15 if self._true_biases is not None else 9
            dof, values = next(iter(self._nees.items()), (default, []))
            return chi2_consistency(values, dof)
        return dict(
            by_dof={
                str(dof): chi2_consistency(values, dof)
                for dof, values in sorted(self._nees.items())
            }
        )
