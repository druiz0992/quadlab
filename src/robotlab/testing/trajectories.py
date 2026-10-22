"""Truth trajectories for synthetic estimator scenarios (world ENU, SI).

A trajectory maps time (s) to a ``TruthPoint``: position, velocity, and
acceleration of the body origin, the body-to-world attitude, and body-axis
angular rates. They must be mutually consistent (velocity is the derivative
of position, and so on); the built-in ones are exact closed forms.
"""

from collections.abc import Callable
from dataclasses import dataclass

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
