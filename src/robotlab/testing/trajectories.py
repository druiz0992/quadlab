"""Truth trajectories for synthetic estimator scenarios (world ENU, SI).

A trajectory maps time (s) to a ``TruthPoint``: position, velocity, and
acceleration of the body origin, the body-to-world attitude, and body-axis
angular rates. They must be mutually consistent (velocity is the derivative
of position, attitude rates match the body rates); all built-in ones are
exact closed forms:

- ``hovering``, ``straight``: level, constant velocity (any vehicle).
- ``circling``: level circle, nose along the path: a ground vehicle (the
  tyres supply the centripetal force), not a multirotor.
- ``flown``: a multirotor on a ``HarmonicPath``. Its attitude follows from
  the path by differential flatness: thrust along body z must supply
  ``a − g``, so the body tilts into every turn and the accelerometer reads
  ``(0, 0, |a − g|)`` (no drag); yaw is fixed, constant-rate, or along the
  path.
- ``tumbling``: stationary body rotating through large roll, pitch, and
  yaw (a rate table or hand-held unit), for attitude-only filters.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TruthPoint:
    """True state at one time (world ENU, body-to-world quaternion xyzw)."""

    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    acceleration_mps2: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    body_rate_rad_s: NDArray[np.float64]


Trajectory = Callable[[float], TruthPoint]


def _level(yaw_rad: float) -> NDArray[np.float64]:
    return Rotation.from_euler("z", yaw_rad).as_quat()


def hovering(position_m: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> Trajectory:
    """Level, still, facing east."""
    point = TruthPoint(
        np.array(position_m, float), np.zeros(3), np.zeros(3), _level(0.0), np.zeros(3)
    )
    return lambda t: point


def straight(
    velocity_mps: tuple[float, float, float],
    start_m: tuple[float, float, float] = (0.0, 0.0, 1.0),
    yaw_rad: float = 0.0,
) -> Trajectory:
    """Constant velocity, level, fixed heading ``yaw_rad`` (from east)."""
    velocity = np.array(velocity_mps, float)
    start = np.array(start_m, float)
    attitude = _level(yaw_rad)
    return lambda t: TruthPoint(
        start + velocity * t, velocity, np.zeros(3), attitude, np.zeros(3)
    )


def circling(radius_m: float, speed_mps: float, height_m: float = 1.0) -> Trajectory:
    """Counter-clockwise level circle about the origin, nose along the path.

    Starts at (radius, 0) heading north. In body axes the specific force
    (0, v²/r, g) and the yaw rate v/r are constant.
    """
    if radius_m <= 0 or speed_mps <= 0:
        raise ValueError("circle radius and speed must be positive")
    rate = speed_mps / radius_m

    def at(t: float) -> TruthPoint:
        angle = rate * t
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        return TruthPoint(
            radius_m * radial + [0.0, 0.0, height_m],
            speed_mps * tangent,
            -rate * speed_mps * radial,
            _level(angle + np.pi / 2),
            np.array([0.0, 0.0, rate]),
        )

    return at


# Multirotor flight (differential flatness) ----------------------------------------


@dataclass(frozen=True)
class Harmonic:
    """One term ``amplitude · sin(frequency · t + phase)`` on world ``axis``
    (0 east, 1 north, 2 up)."""

    axis: int
    amplitude_m: float
    frequency_rad_s: float
    phase_rad: float = 0.0


@dataclass(frozen=True)
class HarmonicPath:
    """``p(t) = offset + velocity · t + Σ harmonics``: smooth, with exact
    derivatives of every order."""

    offset_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    harmonics: tuple[Harmonic, ...] = ()

    def __post_init__(self) -> None:
        if any(h.axis not in (0, 1, 2) for h in self.harmonics):
            raise ValueError("harmonic axis must be 0, 1, or 2")

    def derivatives(self, t: float) -> list[NDArray[np.float64]]:
        """Position, velocity, acceleration, and jerk at ``t``."""
        result = [np.array(self.offset_m, float) + np.multiply(self.velocity_mps, t)]
        result += [np.array(self.velocity_mps, float), np.zeros(3), np.zeros(3)]
        for h in self.harmonics:
            for order in range(4):
                # d^k/dt^k sin(wt + phi) = w^k sin(wt + phi + k pi/2)
                angle = h.frequency_rad_s * t + h.phase_rad + order * math.pi / 2
                value = h.amplitude_m * h.frequency_rad_s**order * math.sin(angle)
                result[order][h.axis] += value
        return result


def figure_eight(
    half_width_m: float,
    period_s: float,
    height_m: float = 2.0,
    climb_m: float = 0.0,
) -> HarmonicPath:
    """Lemniscate of Gerono ``(r sin wt, (r/2) sin 2wt)`` about the origin,
    with an optional vertical oscillation ``climb_m · sin wt``."""
    w = 2 * math.pi / period_s
    terms = [Harmonic(0, half_width_m, w), Harmonic(1, half_width_m / 2, 2 * w)]
    if climb_m:
        terms.append(Harmonic(2, climb_m, w))
    return HarmonicPath((0.0, 0.0, height_m), harmonics=tuple(terms))


def flown(
    path: HarmonicPath,
    yaw_rad: float = 0.0,
    yaw_rate_rad_s: float = 0.0,
    yaw_along_path: bool = False,
) -> Trajectory:
    """Multirotor flying ``path`` (see module docs).

    Heading is ``yaw_rad + yaw_rate_rad_s · t``, or the horizontal velocity
    direction when ``yaw_along_path`` (the path must never stop moving
    horizontally). Attitude and body rates follow Faessler, Franchi and
    Scaramuzza, "Differential flatness of quadrotor dynamics subject to
    rotor drag", RA-L 2018 (with zero drag): ``z_B = (a − g)/|a − g|``,
    ``x_B ∥ y_C × z_B`` with ``y_C = (−sin ψ, cos ψ, 0)``, and rates from
    the jerk. Raises ``ValueError`` in free fall (``a = g``).
    """

    def at(t: float) -> TruthPoint:
        position, velocity, acceleration, jerk = path.derivatives(t)
        if yaw_along_path:
            yaw, yaw_rate = _path_heading(velocity, acceleration)
        else:
            yaw, yaw_rate = yaw_rad + yaw_rate_rad_s * t, yaw_rate_rad_s
        rotation, rates = _flat_attitude(acceleration, jerk, yaw, yaw_rate)
        return TruthPoint(position, velocity, acceleration, rotation.as_quat(), rates)

    return at


def _path_heading(
    velocity: NDArray[np.float64], acceleration: NDArray[np.float64]
) -> tuple[float, float]:
    """Heading of the horizontal velocity and its rate (rad, rad/s)."""
    speed_squared = velocity[0] ** 2 + velocity[1] ** 2
    if speed_squared < 1e-12:
        raise ValueError("yaw_along_path needs horizontal motion at every time")
    rate = (velocity[0] * acceleration[1] - velocity[1] * acceleration[0]) / (
        speed_squared
    )
    return math.atan2(velocity[1], velocity[0]), float(rate)


def _flat_attitude(
    acceleration: NDArray[np.float64],
    jerk: NDArray[np.float64],
    yaw: float,
    yaw_rate: float,
) -> tuple[Rotation, NDArray[np.float64]]:
    """Body-to-world rotation and body rates from the flat outputs."""
    thrust = acceleration + np.array([0.0, 0.0, 9.81])  # a − g, g = (0, 0, −9.81)
    c = float(np.linalg.norm(thrust))
    if c < 1e-6:
        raise ValueError("free fall: the thrust direction is undefined")
    z_b = thrust / c
    x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_c = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    cross = np.cross(y_c, z_b)
    x_b = cross / np.linalg.norm(cross)
    y_b = np.cross(z_b, x_b)
    rate_x = -float(y_b @ jerk) / c
    rate_y = float(x_b @ jerk) / c
    rate_z = (yaw_rate * float(x_c @ x_b) + rate_y * float(y_c @ z_b)) / float(
        np.linalg.norm(cross)
    )
    matrix = np.column_stack([x_b, y_b, z_b])
    return Rotation.from_matrix(matrix), np.array([rate_x, rate_y, rate_z])


# Attitude-only motion ---------------------------------------------------------------


def tumbling(
    roll_amplitude_rad: float = math.radians(60),
    pitch_amplitude_rad: float = math.radians(45),
    yaw_rate_rad_s: float = 0.5,
    period_s: float = 7.0,
    position_m: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Trajectory:
    """Stationary body with ZYX Euler angles ``ψ = yaw_rate · t``,
    ``θ = pitch · sin(wt · 1.3)``, ``φ = roll · sin(wt)`` (w = 2π/period;
    incommensurate so all orientations mix). Body rates from the Euler
    rates: p = φ̇ − ψ̇ sin θ, q = θ̇ cos φ + ψ̇ cos θ sin φ,
    r = −θ̇ sin φ + ψ̇ cos θ cos φ."""
    if abs(pitch_amplitude_rad) >= math.pi / 2:
        raise ValueError("pitch amplitude must stay below 90° (Euler singularity)")
    w = 2 * math.pi / period_s
    position = np.array(position_m, float)

    def at(t: float) -> TruthPoint:
        roll = roll_amplitude_rad * math.sin(w * t)
        pitch = pitch_amplitude_rad * math.sin(1.3 * w * t)
        yaw = yaw_rate_rad_s * t
        roll_rate = roll_amplitude_rad * w * math.cos(w * t)
        pitch_rate = pitch_amplitude_rad * 1.3 * w * math.cos(1.3 * w * t)
        rates = np.array(
            [
                roll_rate - yaw_rate_rad_s * math.sin(pitch),
                pitch_rate * math.cos(roll)
                + yaw_rate_rad_s * math.cos(pitch) * math.sin(roll),
                -pitch_rate * math.sin(roll)
                + yaw_rate_rad_s * math.cos(pitch) * math.cos(roll),
            ]
        )
        attitude = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
        return TruthPoint(position, np.zeros(3), np.zeros(3), attitude, rates)

    return at
