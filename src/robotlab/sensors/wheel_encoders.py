"""Wheel encoders and steering-angle sensors for wheeled robots.

Each wheel has an incremental encoder with ``counts_per_revolution``
counts: the count is floor(θ · cpr / 2π) of the true wheel angle θ, and the
reported speed is the count difference over the time since the previous
reading, so it carries the encoder's quantization (2π / cpr rad per
interval) and averages over the interval, like a real speed estimate from
counts. Steering angles are the true actuator angles plus seeded white
noise of ``steering_noise_std_rad`` (per sample). Readings are taken at
``rate_hz`` on the odometry periods; the controller gets the latest one and
the sensor bus carries every one as ``sensors/wheel_encoders`` (estimator
input). The first reading reports zero speed. Units: rad, rad/s, s.
Reading arrays are read-only (bus samples are immutable).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ..core.measurements import EncoderReading


class WheelEncoderModel:
    """Quantized wheel speed and noisy steering angle (see module docs).

    Disabled: speeds and angles are reported exactly (truth pass-through),
    with counts left at zero.
    """

    def __init__(self, settings: Mapping[str, Any], wheel_count: int) -> None:
        self.enabled = bool(settings["enabled"])
        self.counts_per_revolution = int(settings["counts_per_revolution"])
        self.steering_noise_std_rad = float(settings["steering_noise_std_rad"])
        self._rng = np.random.default_rng(settings["seed"])
        self._previous: tuple[int, NDArray[np.int64]] | None = None
        self._count = wheel_count
        self.readings = 0

    def read(
        self,
        sim_time_ns: int,
        wheel_angle_rad: NDArray[np.float64],
        wheel_speed_rad_s: NDArray[np.float64],
        steer_rad: NDArray[np.float64],
    ) -> EncoderReading:
        """Reading from the true wheel angles, speeds, and steering angles."""
        self.readings += 1
        if not self.enabled:
            return EncoderReading(
                sim_time_ns,
                (0,) * self._count,
                _frozen(wheel_speed_rad_s),
                _frozen(steer_rad),
            )
        scale = self.counts_per_revolution / (2 * np.pi)
        counts = np.floor(np.asarray(wheel_angle_rad) * scale).astype(np.int64)
        speed = np.zeros(self._count)
        if self._previous is not None:
            previous_ns, previous_counts = self._previous
            dt = (sim_time_ns - previous_ns) * 1e-9
            if dt > 0:
                speed = (counts - previous_counts) / scale / dt
        self._previous = (sim_time_ns, counts)
        noise = self._rng.normal(0.0, self.steering_noise_std_rad, self._count)
        steer = np.asarray(steer_rad, float) + (
            noise if self.steering_noise_std_rad > 0 else 0.0
        )
        return EncoderReading(
            sim_time_ns, tuple(int(c) for c in counts), _frozen(speed), _frozen(steer)
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            enabled=self.enabled,
            counts_per_revolution=self.counts_per_revolution,
            steering_noise_std_rad=self.steering_noise_std_rad,
            readings=self.readings,
        )


def _frozen(values: Any) -> NDArray[np.float64]:
    """Read-only float copy (readings are shared through the sensor bus)."""
    array = np.array(values, dtype=float)
    array.flags.writeable = False
    return array
