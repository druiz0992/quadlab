"""Control performance against the mission reference and simulator truth.

Tracking errors compare the true state with the reference sampled at the
same time: e = p_ref − p_true (and likewise velocity and yaw). Step and
doublet missions also get classic step-response figures on the step axis:

- rise time: 10 % → 90 % of the step,
- overshoot: peak beyond the final value, % of the step,
- settling time: last time outside a band of ``max(2 % of step, 0.05 m)``
  around the final value, measured from the step,
- steady-state error: mean error over the last 20 % of the window.

Effort: mean total thrust relative to weight, RMS deviation of total thrust
from weight (N), mean absolute rotor-thrust slew (N/s, smoothness), and the
fraction of commands that saturated the allocation.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ..control.references import ReferencePoint
from .estimation import error_statistics


def step_response(
    times_s: Sequence[float],
    values: Sequence[float],
    *,
    step_time_s: float,
    initial: float,
    final: float,
    end_time_s: float | None = None,
) -> dict[str, Any]:
    """Rise time, overshoot, settling time, and steady-state error of a step.

    ``values`` is the response on the step axis; the window ends at
    ``end_time_s`` (default: last sample). Missing figures are None.
    """
    t, y = np.asarray(times_s, float), np.asarray(values, float)
    end = t[-1] if end_time_s is None else end_time_s
    window = (t >= step_time_s) & (t <= end)
    t, y = t[window] - step_time_s, y[window]
    size = final - initial
    if len(t) < 2 or size == 0:
        return dict(rise_time_s=None, overshoot_pct=None, settling_time_s=None)
    progress = (y - initial) / size  # 0 → 1 along the step
    rise = None
    above10, above90 = np.nonzero(progress >= 0.1)[0], np.nonzero(progress >= 0.9)[0]
    if len(above10) and len(above90):
        rise = float(t[above90[0]] - t[above10[0]])
    overshoot = max(0.0, float(np.max(progress) - 1.0) * 100)
    band = max(0.02 * abs(size), 0.05)
    outside = np.nonzero(np.abs(y - final) > band)[0]
    if len(outside) == 0:
        settling = 0.0
    elif outside[-1] == len(t) - 1:
        settling = None  # never settled within the window
    else:
        settling = float(t[outside[-1] + 1])
    tail = t >= t[-1] - 0.2 * t[-1]
    return dict(
        rise_time_s=rise,
        overshoot_pct=overshoot,
        settling_time_s=settling,
        steady_state_error_m=float(np.mean(final - y[tail])),
        band_m=band,
    )


class ControlMetrics:
    """Online control scoring fed at the state (odometry) rate.

    ``weight_n`` is the vehicle weight used to normalize thrust.
    """

    def __init__(self, weight_n: float) -> None:
        self._weight = weight_n
        self._times: list[float] = []
        self._positions: list[NDArray[np.float64]] = []
        self._errors: dict[str, list[float]] = {
            k: []
            for k in (
                "position_m",
                "horizontal_m",
                "vertical_m",
                "velocity_mps",
                "yaw_deg",
            )
        }
        self._max_tilt_deg = 0.0
        self._thrust_total: list[float] = []
        self._slew: list[float] = []
        self._previous: tuple[float, NDArray[np.float64]] | None = None
        self._commands = 0
        self._saturated = 0

    def add_state(
        self,
        time_s: float,
        position_m: ArrayLike,
        velocity_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
        reference: ReferencePoint | None,
    ) -> dict[str, float] | None:
        """Record truth (and the reference at the same time); return errors."""
        position = np.asarray(position_m, float)
        rotation = Rotation.from_quat(quaternion_xyzw)
        tilt = np.degrees(np.arccos(np.clip(rotation.apply([0, 0, 1])[2], -1, 1)))
        self._max_tilt_deg = max(self._max_tilt_deg, float(tilt))
        self._times.append(time_s)
        self._positions.append(position)
        if reference is None:
            return None
        dp = reference.position_m - position
        dv = reference.velocity_mps - np.asarray(velocity_mps, float)
        yaw = reference.yaw_rad - rotation.as_euler("ZYX")[0]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        values = dict(
            position_m=float(np.linalg.norm(dp)),
            horizontal_m=float(np.hypot(dp[0], dp[1])),
            vertical_m=float(abs(dp[2])),
            velocity_mps=float(np.linalg.norm(dv)),
            yaw_deg=float(np.degrees(abs(yaw))),
        )
        for key, value in values.items():
            self._errors[key].append(value)
        return values

    def add_command(self, time_s: float, thrust_n: ArrayLike, saturated: bool) -> None:
        """Record one released rotor-thrust command."""
        thrust = np.asarray(thrust_n, float)
        self._commands += 1
        self._saturated += int(saturated)
        self._thrust_total.append(float(thrust.sum()))
        if self._previous is not None and time_s > self._previous[0]:
            dt = time_s - self._previous[0]
            self._slew.append(float(np.mean(np.abs(thrust - self._previous[1])) / dt))
        self._previous = (time_s, thrust)

    def step(
        self,
        axis: int,
        step_time_s: float,
        initial: float,
        final: float,
        end_time_s: float | None = None,
    ) -> dict[str, Any]:
        """Step-response figures on one world axis (0 x, 1 y, 2 z)."""
        if not self._times:
            return {}
        values = [p[axis] for p in self._positions]
        return step_response(
            self._times,
            values,
            step_time_s=step_time_s,
            initial=initial,
            final=final,
            end_time_s=end_time_s,
        )

    def summary(self) -> dict[str, Any]:
        thrust = np.asarray(self._thrust_total, float)
        effort = None
        if thrust.size:
            effort = dict(
                mean_thrust_to_weight=float(thrust.mean() / self._weight),
                rms_thrust_deviation_n=float(
                    np.sqrt(np.mean((thrust - self._weight) ** 2))
                ),
                mean_rotor_slew_n_per_s=(
                    float(np.mean(self._slew)) if self._slew else None
                ),
            )
        return dict(
            tracking={k: error_statistics(v) for k, v in self._errors.items() if v},
            max_tilt_deg=self._max_tilt_deg,
            effort=effort,
            commands=self._commands,
            saturated_fraction=(
                self._saturated / self._commands if self._commands else None
            ),
        )
