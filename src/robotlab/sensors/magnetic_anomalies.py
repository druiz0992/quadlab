"""Magnetic anomalies from nearby structures, as point dipoles.

Steel structures, vehicles, rebar, and buried pipes distort the Earth field
near them. Each anomaly is a magnetic dipole with moment m (A·m², world
axes) at a world position; its field at offset r from the dipole is

    B = μ₀/(4π) · (3 r̂ (m·r̂) − m) / |r|³        (tesla)

(e.g. a 1000 A·m² moment, a parked car, gives about 2 µT at 5 m). Distances
below ``MIN_DISTANCE_M`` are clamped, as the dipole model fails inside the
object. Dipoles come from scene obstacles with ``magnetic_moment_am2``
(placed at the obstacle centre) and from ``sensors.magnetometer.dipoles``.
The field is evaluated at the sensor's true position and rotated into
sensor axes with the true attitude: physical field synthesis, applied before
the sensor's own errors.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

MU0_OVER_4PI = 1e-7  # T·m/A
MIN_DISTANCE_M = 0.5


class MagneticAnomalies:
    """Sum of dipole fields at a point (see module docs)."""

    def __init__(self, dipoles: Iterable[Mapping[str, Any]]) -> None:
        dipoles = list(dipoles)
        self._positions = np.array([d["position_m"] for d in dipoles], float).reshape(
            -1, 3
        )
        self._moments = np.array([d["moment_am2"] for d in dipoles], float).reshape(
            -1, 3
        )
        self.max_field_ut = 0.0

    @property
    def enabled(self) -> bool:
        return len(self._positions) > 0

    def field_world(self, position_m: ArrayLike) -> NDArray[np.float64]:
        """Anomaly field (tesla, world axes) at a world position."""
        if not self.enabled:
            return np.zeros(3)
        r = np.asarray(position_m, float) - self._positions
        distance = np.maximum(np.linalg.norm(r, axis=1), MIN_DISTANCE_M)
        unit = r / np.linalg.norm(r, axis=1, keepdims=True).clip(min=1e-12)
        dot = np.sum(self._moments * unit, axis=1, keepdims=True)
        field = MU0_OVER_4PI * (3 * unit * dot - self._moments) / distance[:, None] ** 3
        total = field.sum(axis=0)
        self.max_field_ut = max(self.max_field_ut, float(np.linalg.norm(total)) * 1e6)
        return total


def scene_dipoles(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Dipoles at scene obstacles that declare ``magnetic_moment_am2``."""
    return [
        dict(position_m=o["position"], moment_am2=o["magnetic_moment_am2"])
        for o in scene.get("obstacles", [])
        if "magnetic_moment_am2" in o
    ]
