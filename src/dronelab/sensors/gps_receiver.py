"""GNSS receiver models beyond position error: velocity, status, RTK, heading.

- ``GpsVelocityErrors`` (``sensors.gps.velocity_error``): Doppler velocity
  error per ENU axis = first-order Gauss–Markov (stationary σ
  ``correlated_std_mps``, correlation ``correlation_time_s``) + white
  (``white_horizontal_std_mps``, ``white_vertical_std_mps``).
- ``RtkErrorModel`` (``sensors.gps.rtk``): carrier-phase RTK replacing the
  standard position error model. "rtk_float" with Gauss–Markov error of σ
  ``float_std_m`` (τ 10 s) until ``convergence_s`` after the first fix, then
  "rtk_fixed" with white error ``fixed_std_m`` (horizontal) and
  ``fixed_vertical_std_m``. While GPS multipath is active the solution falls
  back to float; it re-fixes ``convergence_s`` / 3 after multipath ends.
- ``GpsStatusModel`` (``sensors.gps.receiver``): what the receiver reports
  about itself. Satellites: ``satellites`` ± a bounded random walk of
  ``satellite_variation``, minus ``multipath_satellite_loss`` during
  multipath. HDOP = ``hdop`` · √(satellites₀ / satellites). Reported 1-σ
  accuracies are the receiver's estimates, not the true error: horizontal =
  HDOP · ``uere_m`` (RTK: the float/fixed σ), vertical = 1.6 × horizontal,
  speed = ``speed_accuracy_mps``.
- ``GpsHeading`` (``sensors.gps.heading``): dual-antenna heading at each
  delivered fix: true yaw + ``mount_yaw_deg`` + N(0, ``noise_std_deg``²),
  wrapped to (−π, π]. Typical accuracy ≈ 0.2° per metre of baseline.

The truth these models read (true yaw for the heading) is used only to
synthesize the measurement. Seeded local RNGs.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.measurements import GpsHeadingReading, GpsReading, GpsStatus
from .gps_errors import offset_fix


def _gauss_markov(previous, sigma, tau_s, dt_s, rng, size=3):
    decay = math.exp(-dt_s / tau_s)
    return decay * previous + math.sqrt(1 - decay**2) * sigma * rng.normal(size=size)


class GpsVelocityErrors:
    """Correlated plus white ENU velocity error (see module docs)."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._sigma = settings["correlated_std_mps"]
        self._tau = settings["correlation_time_s"]
        self._white = np.array(
            [settings["white_horizontal_std_mps"]] * 2
            + [settings["white_vertical_std_mps"]]
        )
        self._state = self._rng.normal(size=3) * self._sigma
        self._previous_ns: int | None = None

    def apply(
        self, velocity_enu_mps: ArrayLike, sim_time_ns: int
    ) -> NDArray[np.float64]:
        velocity = np.asarray(velocity_enu_mps, float)
        if not self.enabled:
            return velocity
        if self._previous_ns is not None and sim_time_ns > self._previous_ns:
            dt = (sim_time_ns - self._previous_ns) * 1e-9
            self._state = _gauss_markov(
                self._state, self._sigma, self._tau, dt, self._rng
            )
        self._previous_ns = sim_time_ns
        return velocity + self._state + self._white * self._rng.normal(size=3)


class RtkErrorModel:
    """RTK float/fixed position error (see module docs)."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._convergence_ns = round(settings["convergence_s"] * 1e9)
        self._float = settings["float_std_m"]
        self._fixed = np.array(
            [settings["fixed_std_m"]] * 2 + [settings["fixed_vertical_std_m"]]
        )
        self._float_error = self._rng.normal(size=3) * self._float
        self._fixed_at_ns: int | None = None
        self._previous_ns: int | None = None
        self.state = "rtk_float"
        self.fixed_fixes = 0

    def apply(self, fix: GpsReading, sim_time_ns: int, degraded: bool) -> GpsReading:
        if not self.enabled:
            return fix
        if self._fixed_at_ns is None:
            self._fixed_at_ns = sim_time_ns + self._convergence_ns
        if degraded:  # multipath: lose the fix, re-converge faster afterwards
            self._fixed_at_ns = max(
                self._fixed_at_ns, sim_time_ns + self._convergence_ns // 3
            )
        if self._previous_ns is not None and sim_time_ns > self._previous_ns:
            dt = (sim_time_ns - self._previous_ns) * 1e-9
            self._float_error = _gauss_markov(
                self._float_error, self._float, 10.0, dt, self._rng
            )
        self._previous_ns = sim_time_ns
        if sim_time_ns >= self._fixed_at_ns:
            self.state = "rtk_fixed"
            self.fixed_fixes += 1
            offset = self._fixed * self._rng.normal(size=3)
        else:
            self.state = "rtk_float"
            offset = self._float_error
        return offset_fix(fix, offset)

    @property
    def accuracy_m(self) -> float:
        return float(self._fixed[0]) if self.state == "rtk_fixed" else self._float


class GpsStatusModel:
    """Receiver-reported fix quality (see module docs)."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self._settings = settings
        self._rng = np.random.default_rng(settings["seed"])
        self._walk = 0
        self.status: GpsStatus | None = None

    def update(
        self, multipath_active: bool, rtk: RtkErrorModel | None = None
    ) -> GpsStatus:
        s = self._settings
        step = int(self._rng.integers(-1, 2))
        variation = s["satellite_variation"]
        self._walk = int(np.clip(self._walk + step, -variation, variation))
        satellites = s["satellites"] + self._walk
        if multipath_active:
            satellites -= s["multipath_satellite_loss"]
        satellites = max(4, satellites)
        hdop = s["hdop"] * math.sqrt(s["satellites"] / satellites)
        if rtk is not None and rtk.enabled:
            fix_type, horizontal = rtk.state, rtk.accuracy_m
        else:
            fix_type, horizontal = "3d", hdop * s["uere_m"]
        self.status = GpsStatus(
            fix_type,
            satellites,
            hdop,
            horizontal,
            1.6 * horizontal,
            s["speed_accuracy_mps"],
        )
        return self.status


class GpsHeading:
    """Dual-antenna heading at delivered fixes (see module docs)."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._mount = math.radians(settings["mount_yaw_deg"])
        self._noise = math.radians(settings["noise_std_deg"])
        self.samples = 0

    def sample(self, true_yaw_rad: float) -> GpsHeadingReading:
        self.samples += 1
        heading = true_yaw_rad + self._mount + self._noise * self._rng.normal()
        wrapped = math.atan2(math.sin(heading), math.cos(heading))
        return GpsHeadingReading(wrapped, self._noise)
