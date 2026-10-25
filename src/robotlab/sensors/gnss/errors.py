"""Seeded, time-correlated GPS position error in local east/north/up metres.

Real GPS position error is dominated by slowly varying components (satellite
geometry, atmosphere, multipath), not independent per-fix noise. This model
adds, per ENU axis:

    error = correlated(t) + white(t)

- ``correlated``: first-order Gauss-Markov process with stationary standard
  deviation ``*_std_m`` and correlation time ``correlation_time_s``. It is
  updated exactly over the actual elapsed simulation interval and starts in
  its stationary distribution, so errors exist from the first fix.
- ``white``: independent per-fix Gaussian noise, ``white_*_std_m``.

Horizontal settings apply equally to east and north, so the error is
isotropic in metres (unlike Gazebo's native degree-based horizontal noise).
The ENU offset is mapped to latitude/longitude/altitude with WGS84 radii of
curvature at the fix; this linearization is accurate to well below a
millimetre for metre-level offsets. Velocity fields are not modified.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.measurements import GpsReading

_EQUATORIAL_RADIUS_M = 6378137.0
_ECCENTRICITY_SQUARED = 6.6943799901413165e-3


def offset_fix(fix: GpsReading, offset_enu_m: NDArray[np.float64]) -> GpsReading:
    """Move a WGS84 fix by a small local east/north/up offset in metres."""
    latitude = math.radians(fix.latitude_deg)
    denominator = 1 - _ECCENTRICITY_SQUARED * math.sin(latitude) ** 2
    # Prime-vertical (east-west) and meridian (north-south) radii of curvature.
    prime_vertical_m = _EQUATORIAL_RADIUS_M / math.sqrt(denominator)
    meridian_m = _EQUATORIAL_RADIUS_M * (1 - _ECCENTRICITY_SQUARED) / denominator**1.5
    east, north, up = map(float, offset_enu_m)
    d_latitude = north / (meridian_m + fix.altitude_m)
    d_longitude = east / ((prime_vertical_m + fix.altitude_m) * math.cos(latitude))
    return GpsReading(
        fix.latitude_deg + math.degrees(d_latitude),
        fix.longitude_deg + math.degrees(d_longitude),
        fix.altitude_m + up,
    )


class GpsErrorModel:
    """Apply seeded correlated plus white ENU position error to ordered fixes.

    ``settings`` is the validated ``sensors.gps.error_model`` section. Call
    ``apply`` for every received fix, including fixes later removed by the
    dropout model, so the random sequence does not depend on dropouts.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        horizontal, vertical = settings["horizontal_std_m"], settings["vertical_std_m"]
        self._correlated_std = np.array([horizontal, horizontal, vertical], dtype=float)
        white_h, white_v = (
            settings["white_horizontal_std_m"],
            settings["white_vertical_std_m"],
        )
        self._white_std = np.array([white_h, white_h, white_v], dtype=float)
        self._tau_s: float = settings["correlation_time_s"]
        self._correlated = np.zeros(3)
        self._previous_time_ns: int | None = None
        self._samples = 0
        self._peak_horizontal_m = 0.0

    def apply(self, fix: GpsReading, sim_time_ns: int) -> GpsReading:
        """Return the fix moved by the current error; identity when disabled."""
        if not self.enabled:
            return fix
        self._advance(sim_time_ns)
        error = self._correlated + self._white_std * self._rng.normal(size=3)
        self._samples += 1
        self._peak_horizontal_m = max(
            self._peak_horizontal_m, float(np.hypot(error[0], error[1]))
        )
        return offset_fix(fix, error)

    def _advance(self, sim_time_ns: int) -> None:
        """Propagate the Gauss-Markov state exactly to sim_time_ns."""
        previous = self._previous_time_ns
        if sim_time_ns < 0 or (previous is not None and sim_time_ns <= previous):
            raise ValueError(
                "GPS error timestamps must be nonnegative and strictly increasing"
            )
        self._previous_time_ns = sim_time_ns
        if previous is None:
            # Start stationary: the receiver has been running before the mission.
            self._correlated = self._correlated_std * self._rng.normal(size=3)
            return
        dt_s = (sim_time_ns - previous) * 1e-9
        decay = math.exp(-dt_s / self._tau_s)
        # Exact discretization keeps the stationary std for any interval.
        drive_std = self._correlated_std * math.sqrt(
            -math.expm1(-2 * dt_s / self._tau_s)
        )
        self._correlated = decay * self._correlated + drive_std * self._rng.normal(
            size=3
        )

    def summary(self) -> dict[str, Any]:
        """Diagnostics for summary.json, for evaluation only."""
        return dict(
            enabled=self.enabled,
            samples=self._samples,
            final_correlated_error_enu_m=self._correlated.tolist(),
            peak_horizontal_error_m=self._peak_horizontal_m,
        )
