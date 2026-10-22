"""Synthetic downward optical-flow sensor over flat ground.

Gazebo has no optical-flow sensor, so this model synthesizes one from the
vehicle's true motion, like other Python sensor models: truth generates the
measurement, the estimator only ever sees the noisy output.

Geometry. The sensor's optical axis is its +x axis (the same convention as
Gazebo cameras and ray sensors); the default mount pitches it 90° to look
down. For the ground point on the optical axis at distance d, with sensor
linear velocity v and angular velocity ω (both in sensor axes), the point's
apparent velocity is ``ṗ = -v - ω × (d, 0, 0)``. The sensor reports the
angular image rates along its image axes y and z:

    flow_rad_s = (ṗ_y / d, ṗ_z / d) = (-v_y / d - ω_z,  -v_z / d + ω_y)

This is raw flow, including body rotation, as real sensors report it;
compensating with the gyro is the estimator's job. The ground is the plane
z = 0, or the scene terrain when one is supplied; obstacles under the
vehicle are ignored, unlike the Gazebo rangefinder. A sample is invalid (``valid = False``, flow NaN) when the axis
does not point at the ground, the distance is outside the configured range,
or either rate exceeds ``max_rate_rad_s``. Samples lie on a fixed
simulation-time grid at ``rate_hz``; each uses the latest truth state.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class OpticalFlowReading:
    """One flow sample; flow is NaN when invalid."""

    sim_time_ns: int
    flow_rad_s: NDArray[np.float64]  # Image-axis angular rates (sensor y, z)
    distance_m: float  # Along the optical axis to the ground (NaN if none)
    valid: bool


class OpticalFlowModel:
    """Synthesize flow samples from truth at a fixed rate.

    ``settings`` is the validated ``sensors.optical_flow`` section.
    """

    def __init__(self, settings: Mapping[str, Any], terrain: Any = None) -> None:
        # Optional simulation.terrain.Terrain; flat z = 0 ground without it.
        self._terrain = terrain
        self.enabled: bool = settings["enabled"]
        mount = settings["mount"]
        self._offset_m = np.asarray(mount["position_m"], dtype=float)
        # Fixed-axis roll/pitch/yaw, the same convention as Gazebo SDF poses.
        self._mount = Rotation.from_euler("xyz", mount["rotation_deg"], degrees=True)
        self._period_ns = round(1e9 / settings["rate_hz"])
        self._min_m: float = settings["min_distance_m"]
        self._max_m: float = settings["max_distance_m"]
        self._max_rate: float = settings["max_rate_rad_s"]
        self._noise_std: float = settings["noise_std_rad_s"]
        self._rng = np.random.default_rng(settings["seed"])
        self._last_tick = -1
        self.samples = 0
        self.invalid = 0

    def sample(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        velocity_world_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
        body_rates_rad_s: ArrayLike,
    ) -> OpticalFlowReading | None:
        """Return a reading when a grid time is reached, else None."""
        tick = sim_time_ns // self._period_ns
        if not self.enabled or tick <= self._last_tick:
            return None
        self._last_tick = tick
        body = Rotation.from_quat(quaternion_xyzw)
        sensor = body * self._mount
        rates_body = np.asarray(body_rates_rad_s, dtype=float)
        # Sensor velocity includes the lever arm: v + ω × r, in world axes.
        velocity_world = np.asarray(velocity_world_mps, dtype=float) + body.apply(
            np.cross(rates_body, self._offset_m)
        )
        origin = np.asarray(position_m, dtype=float) + body.apply(self._offset_m)
        axis_z = sensor.apply([1.0, 0.0, 0.0])[2]
        if self._terrain is not None:
            distance = self._terrain.ray_distance(
                origin, sensor.apply([1.0, 0.0, 0.0]), self._max_m
            )
        else:
            distance = -origin[2] / axis_z if axis_z < 0 else math.nan
        v = sensor.inv().apply(velocity_world)
        w = self._mount.inv().apply(rates_body)
        self.samples += 1
        # Draw every sample so the sequence does not depend on validity.
        noise = self._noise_std * self._rng.normal(size=2)
        if not self._min_m <= distance <= self._max_m:
            return self._invalid(sim_time_ns, distance)
        flow = np.array([-v[1] / distance - w[2], -v[2] / distance + w[1]]) + noise
        if np.any(np.abs(flow) > self._max_rate):
            return self._invalid(sim_time_ns, distance)
        return OpticalFlowReading(sim_time_ns, flow, distance, True)

    def _invalid(self, sim_time_ns: int, distance: float) -> OpticalFlowReading:
        self.invalid += 1
        return OpticalFlowReading(sim_time_ns, np.full(2, math.nan), distance, False)

    def summary(self) -> dict[str, Any]:
        return dict(enabled=self.enabled, samples=self.samples, invalid=self.invalid)
