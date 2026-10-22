"""Reference trajectories for testing controllers (world ENU frame, SI units).

Each reference maps mission time t (s, from the mission start) to a
``ReferencePoint`` with analytic position, velocity, acceleration, yaw, and
yaw rate, so controllers can use feed-forward terms and metrics can compare
against the exact reference. All references are deterministic.

| Type | Motion |
| --- | --- |
| ``hold`` | Stay at ``center_m`` |
| ``step`` | Hold ``center_m``, then jump by ``step_m`` at ``start_s`` |
| ``waypoints`` | Stop-and-go minimum-jerk segments through ``waypoints_m``, ``laps`` times |
| ``circle`` | Constant-speed circle of ``radius_m`` around ``center_m`` |
| ``figure_eight`` | Lemniscate of Gerono, ``radius_m`` half-width, period from ``speed_mps`` |
| ``chirp`` | Sinusoid on ``axis`` sweeping linearly from ``frequency_start_hz`` to ``frequency_end_hz`` over ``duration_s`` |
| ``doublet`` | +``amplitude_m`` then −``amplitude_m`` on ``axis``, each ``pulse_s`` long |

Minimum-jerk segments follow s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ (Flash & Hogan 1985),
whose peak speed is 1.875·d/T, so a segment of length d lasts
T = 1.875·d / ``speed_mps``. With ``yaw_mode: path`` the vehicle first turns
to face the next segment (minimum-jerk in yaw at ``yaw_rate_dps``) and then
flies it; ``fixed`` keeps ``yaw_deg``. Position steps and doublets are
discontinuous on purpose: they measure a controller's step response. The
chirp and doublet excite the closed loop for frequency-response or system-
identification experiments.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

MISSION_TYPES = (
    "none",
    "hold",
    "step",
    "waypoints",
    "circle",
    "figure_eight",
    "chirp",
    "doublet",
)
AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class ReferencePoint:
    """Desired state at one instant (world ENU; yaw about +z from +x)."""

    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    acceleration_mps2: NDArray[np.float64]
    yaw_rad: float
    yaw_rate_rad_s: float = 0.0


class Reference(Protocol):
    """Anything that gives a reference point at mission time t (s)."""

    duration_s: float

    def sample(self, t_s: float) -> ReferencePoint: ...


def _point(position, velocity=None, acceleration=None, yaw=0.0, yaw_rate=0.0):
    zero = np.zeros(3)
    return ReferencePoint(
        np.asarray(position, float),
        zero if velocity is None else np.asarray(velocity, float),
        zero if acceleration is None else np.asarray(acceleration, float),
        float(yaw),
        float(yaw_rate),
    )


def minimum_jerk(tau: float) -> tuple[float, float, float]:
    """Normalized minimum-jerk profile s, ds/dτ, d²s/dτ² for τ in [0, 1]."""
    tau = min(max(tau, 0.0), 1.0)
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    ds = 30 * tau**2 * (1 - tau) ** 2
    dds = 60 * tau * (1 - tau) * (1 - 2 * tau)
    return s, ds, dds


class Hold:
    """Hold a position and yaw indefinitely."""

    def __init__(self, center_m: ArrayLike, yaw_rad: float = 0.0) -> None:
        self._center = np.asarray(center_m, float)
        self._yaw = yaw_rad
        self.duration_s = 0.0

    def sample(self, t_s: float) -> ReferencePoint:
        return _point(self._center, yaw=self._yaw)


class Step:
    """Hold ``center_m``; at ``start_s`` the setpoint jumps by ``step_m``."""

    def __init__(
        self, center_m: ArrayLike, step_m: ArrayLike, start_s: float, yaw_rad: float
    ) -> None:
        self._center = np.asarray(center_m, float)
        self._step = np.asarray(step_m, float)
        self.start_s = start_s
        self._yaw = yaw_rad
        self.duration_s = start_s

    def sample(self, t_s: float) -> ReferencePoint:
        offset = self._step if t_s >= self.start_s else 0.0
        return _point(self._center + offset, yaw=self._yaw)


@dataclass(frozen=True)
class _Phase:
    """One piece of a waypoint mission: a turn, a flight segment, or a dwell."""

    start_s: float
    duration_s: float
    kind: str  # "turn", "fly", or "dwell"
    position_from: NDArray[np.float64]
    position_to: NDArray[np.float64]
    yaw_from: float
    yaw_to: float


class Waypoints:
    """Stop-and-go minimum-jerk flight through waypoints (see module docs)."""

    def __init__(
        self,
        start_m: ArrayLike,
        waypoints_m: Sequence[Sequence[float]],
        *,
        speed_mps: float,
        dwell_s: float,
        laps: int,
        yaw_mode: str,
        yaw_rad: float,
        yaw_rate_rad_s: float,
    ) -> None:
        points = [np.asarray(p, float) for p in waypoints_m] * laps
        self._phases: list[_Phase] = []
        position, yaw, time_s = np.asarray(start_m, float), yaw_rad, 0.0
        for target in points:
            delta = target - position
            if yaw_mode == "path" and np.hypot(*delta[:2]) > 1e-6:
                heading = math.atan2(delta[1], delta[0])
                turn = _wrap(heading - yaw)
                if abs(turn) > 1e-6:
                    duration = 1.875 * abs(turn) / yaw_rate_rad_s
                    time_s = self._add(
                        time_s, duration, "turn", position, position, yaw, yaw + turn
                    )
                    yaw = yaw + turn
            distance = float(np.linalg.norm(delta))
            if distance > 1e-9:
                time_s = self._add(
                    time_s,
                    1.875 * distance / speed_mps,
                    "fly",
                    position,
                    target,
                    yaw,
                    yaw,
                )
            if dwell_s > 0:
                time_s = self._add(time_s, dwell_s, "dwell", target, target, yaw, yaw)
            position = target
        self._end = _point(position, yaw=yaw)
        self.duration_s = time_s

    def _add(self, start, duration, kind, p0, p1, yaw0, yaw1) -> float:
        self._phases.append(_Phase(start, duration, kind, p0, p1, yaw0, yaw1))
        return start + duration

    def sample(self, t_s: float) -> ReferencePoint:
        for phase in self._phases:
            if t_s < phase.start_s + phase.duration_s:
                return _phase_point(phase, max(t_s - phase.start_s, 0.0))
        return self._end

    @property
    def phases(self) -> tuple[_Phase, ...]:
        return tuple(self._phases)


def _phase_point(phase: _Phase, elapsed_s: float) -> ReferencePoint:
    """Minimum-jerk interpolation in position and yaw within one phase."""
    T = phase.duration_s
    s, ds, dds = minimum_jerk(elapsed_s / T)
    delta = phase.position_to - phase.position_from
    turn = phase.yaw_to - phase.yaw_from
    return _point(
        phase.position_from + s * delta,
        ds / T * delta,
        dds / T**2 * delta,
        _wrap(phase.yaw_from + s * turn),
        ds / T * turn,
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Circle:
    """Counter-clockwise circle in the horizontal plane at constant speed."""

    def __init__(
        self,
        center_m: ArrayLike,
        radius_m: float,
        speed_mps: float,
        yaw_mode: str,
        yaw_rad: float,
    ) -> None:
        self._center = np.asarray(center_m, float)
        self._radius = radius_m
        self._omega = speed_mps / radius_m
        self._yaw_mode = yaw_mode
        self._yaw = yaw_rad
        self.duration_s = 2 * math.pi / self._omega  # one lap

    def sample(self, t_s: float) -> ReferencePoint:
        r, w = self._radius, self._omega
        c, s = math.cos(w * t_s), math.sin(w * t_s)
        position = self._center + r * np.array([c, s, 0.0])
        velocity = r * w * np.array([-s, c, 0.0])
        acceleration = -r * w**2 * np.array([c, s, 0.0])
        if self._yaw_mode == "path":
            return _point(
                position, velocity, acceleration, _wrap(w * t_s + math.pi / 2), w
            )
        return _point(position, velocity, acceleration, self._yaw)


class FigureEight:
    """Lemniscate of Gerono: x = r sin(ωt), y = (r/2) sin(2ωt)."""

    def __init__(
        self, center_m: ArrayLike, radius_m: float, speed_mps: float, yaw_rad: float
    ) -> None:
        self._center = np.asarray(center_m, float)
        self._radius = radius_m
        # Peak speed r·ω·√2 (at the crossing) equals speed_mps.
        self._omega = speed_mps / (radius_m * math.sqrt(2))
        self._yaw = yaw_rad
        self.duration_s = 2 * math.pi / self._omega

    def sample(self, t_s: float) -> ReferencePoint:
        r, w = self._radius, self._omega
        p = np.array([r * math.sin(w * t_s), r / 2 * math.sin(2 * w * t_s), 0.0])
        v = np.array([r * w * math.cos(w * t_s), r * w * math.cos(2 * w * t_s), 0.0])
        a = np.array(
            [-r * w**2 * math.sin(w * t_s), -2 * r * w**2 * math.sin(2 * w * t_s), 0.0]
        )
        return _point(self._center + p, v, a, self._yaw)


class Chirp:
    """Linear-frequency sweep on one axis: A·sin(2π(f₀t + k t²/2)), k = (f₁−f₀)/T."""

    def __init__(
        self,
        center_m: ArrayLike,
        axis: str,
        amplitude_m: float,
        f0_hz: float,
        f1_hz: float,
        duration_s: float,
        yaw_rad: float,
    ) -> None:
        self._center = np.asarray(center_m, float)
        self._axis = AXES[axis]
        self._amplitude = amplitude_m
        self._f0, self._k = f0_hz, (f1_hz - f0_hz) / duration_s
        self._yaw = yaw_rad
        self.duration_s = duration_s

    def frequency_hz(self, t_s: float) -> float:
        """Instantaneous frequency at mission time t."""
        return self._f0 + self._k * min(max(t_s, 0.0), self.duration_s)

    def sample(self, t_s: float) -> ReferencePoint:
        if not 0 <= t_s <= self.duration_s:
            return _point(self._center, yaw=self._yaw)
        phase = 2 * math.pi * (self._f0 * t_s + self._k * t_s**2 / 2)
        rate = 2 * math.pi * (self._f0 + self._k * t_s)  # dφ/dt
        A = self._amplitude
        unit = np.eye(3)[self._axis]
        position = self._center + A * math.sin(phase) * unit
        velocity = A * rate * math.cos(phase) * unit
        acceleration = (
            A
            * (2 * math.pi * self._k * math.cos(phase) - rate**2 * math.sin(phase))
            * unit
        )
        return _point(position, velocity, acceleration, self._yaw)


class Doublet:
    """+A for ``pulse_s``, then −A for ``pulse_s``, starting at ``start_s``."""

    def __init__(
        self,
        center_m: ArrayLike,
        axis: str,
        amplitude_m: float,
        pulse_s: float,
        start_s: float,
        yaw_rad: float,
    ) -> None:
        self._center = np.asarray(center_m, float)
        self._unit = np.eye(3)[AXES[axis]]
        self._amplitude, self._pulse = amplitude_m, pulse_s
        self.start_s = start_s
        self._yaw = yaw_rad
        self.duration_s = start_s + 2 * pulse_s

    def sample(self, t_s: float) -> ReferencePoint:
        elapsed = t_s - self.start_s
        if 0 <= elapsed < self._pulse:
            sign = 1.0
        elif self._pulse <= elapsed < 2 * self._pulse:
            sign = -1.0
        else:
            sign = 0.0
        return _point(self._center + sign * self._amplitude * self._unit, yaw=self._yaw)


def build_reference(mission: Mapping[str, Any], start_m: ArrayLike) -> Reference | None:
    """Reference for the validated ``mission`` section; None for ``none``.

    ``start_m`` is where the mission begins (the vehicle's position when it
    starts), used by ``waypoints`` as its first point.
    """
    kind = mission["type"]
    yaw = math.radians(mission["yaw_deg"])
    center = mission["center_m"]
    if kind == "none":
        return None
    if kind == "hold":
        return Hold(center, yaw)
    if kind == "step":
        return Step(center, mission["step_m"], mission["step_time_s"], yaw)
    if kind == "waypoints":
        return Waypoints(
            start_m,
            mission["waypoints_m"],
            speed_mps=mission["speed_mps"],
            dwell_s=mission["dwell_s"],
            laps=mission["laps"],
            yaw_mode=mission["yaw_mode"],
            yaw_rad=yaw,
            yaw_rate_rad_s=math.radians(mission["yaw_rate_dps"]),
        )
    if kind == "circle":
        return Circle(
            center, mission["radius_m"], mission["speed_mps"], mission["yaw_mode"], yaw
        )
    if kind == "figure_eight":
        return FigureEight(center, mission["radius_m"], mission["speed_mps"], yaw)
    if kind == "chirp":
        return Chirp(
            center,
            mission["axis"],
            mission["amplitude_m"],
            mission["frequency_start_hz"],
            mission["frequency_end_hz"],
            mission["duration_s"],
            yaw,
        )
    return Doublet(
        center,
        mission["axis"],
        mission["amplitude_m"],
        mission["pulse_s"],
        mission["step_time_s"],
        yaw,
    )


def validate_mission(mission: Mapping[str, Any]) -> None:
    """Cross-field checks beyond the generic config validation."""
    kind = mission["type"]
    if kind not in MISSION_TYPES:
        raise ValueError(f"mission.type must be one of {MISSION_TYPES}")
    if mission["yaw_mode"] not in ("fixed", "path"):
        raise ValueError("mission.yaw_mode must be fixed or path")
    if mission["axis"] not in AXES:
        raise ValueError("mission.axis must be x, y, or z")
    if kind == "waypoints" and not mission["waypoints_m"]:
        raise ValueError("mission type waypoints needs waypoints_m")
    if kind == "figure_eight" and mission["yaw_mode"] == "path":
        raise ValueError("figure_eight supports yaw_mode fixed only")
    if mission["frequency_end_hz"] < mission["frequency_start_hz"]:
        raise ValueError("mission chirp frequency_end_hz must be >= frequency_start_hz")
    if type(mission["laps"]) is not int:
        raise ValueError("mission.laps must be an integer")
