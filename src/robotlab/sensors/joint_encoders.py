"""Joint encoders for bipeds: quantized servo positions and velocities.

Each actuated joint reports its angle as an absolute encoder with
``counts_per_revolution`` counts: count = round((θ + n) · cpr / 2π), with
n seeded white noise of ``position_noise_std_rad`` per sample (added before
quantization, so it shows as count flicker as on a real encoder), and the
reported angle is count · 2π / cpr. Velocity is the true joint velocity
quantized to multiples of ``velocity_resolution_rad_s`` (0 disables), like
a servo that reports speed in fixed units (Dynamixel XL330: 0.229 rpm).
Readings are taken at ``rate_hz`` and published on the sensor bus as
``sensors/joint_encoders``. Disabled: positions and velocities pass
through exactly, with counts left at zero. Units: rad, rad/s (m and m/s
for slide joints). Reading arrays are read-only (bus samples are immutable).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class JointStateReading:
    """One reading per actuated joint (``names`` order) at ``sim_time_ns``."""

    sim_time_ns: int
    names: tuple[str, ...]
    counts: tuple[int, ...]
    position_rad: NDArray[np.float64]
    velocity_rad_s: NDArray[np.float64]


class JointEncoderModel:
    """Quantized, optionally noisy joint readings (see module docs)."""

    def __init__(self, settings: Mapping[str, Any], names: tuple[str, ...]) -> None:
        self.enabled = bool(settings["enabled"])
        self.counts_per_revolution = int(settings["counts_per_revolution"])
        self.velocity_resolution_rad_s = float(settings["velocity_resolution_rad_s"])
        self.position_noise_std_rad = float(settings["position_noise_std_rad"])
        self._rng = np.random.default_rng(settings["seed"])
        self._names = names
        self.readings = 0

    def read(
        self, sim_time_ns: int, position_rad: ArrayLike, velocity_rad_s: ArrayLike
    ) -> JointStateReading:
        """Reading from the true joint positions and velocities; raises
        ValueError when their length differs from the joint names."""
        position = np.asarray(position_rad, dtype=float)
        velocity = np.asarray(velocity_rad_s, dtype=float)
        if position.shape != (len(self._names),) or velocity.shape != position.shape:
            raise ValueError(
                f"joint encoders: expected {len(self._names)} positions and velocities"
            )
        self.readings += 1
        if not self.enabled:
            return JointStateReading(
                sim_time_ns,
                self._names,
                (0,) * len(self._names),
                _frozen(position),
                _frozen(velocity),
            )
        if self.position_noise_std_rad > 0:
            position = position + self._rng.normal(
                0.0, self.position_noise_std_rad, position.shape
            )
        scale = self.counts_per_revolution / (2 * np.pi)
        counts = np.round(position * scale).astype(np.int64)
        resolution = self.velocity_resolution_rad_s
        if resolution > 0:
            velocity = np.round(velocity / resolution) * resolution
        return JointStateReading(
            sim_time_ns,
            self._names,
            tuple(int(c) for c in counts),
            _frozen(counts / scale),
            _frozen(velocity),
        )

    def summary(self) -> dict[str, Any]:
        """Settings and reading count for summary.json."""
        return dict(
            enabled=self.enabled,
            counts_per_revolution=self.counts_per_revolution,
            velocity_resolution_rad_s=self.velocity_resolution_rad_s,
            position_noise_std_rad=self.position_noise_std_rad,
            readings=self.readings,
        )


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Read-only float copy (readings are shared through the sensor bus)."""
    array = np.array(values, dtype=float)
    array.flags.writeable = False
    return array
