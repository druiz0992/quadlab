"""Lidar realism Gazebo's gpu_lidar lacks.

Applied in Python to each decoded scan, in the lidar's own axes, before
recording. ``process_scan`` first applies the per-beam return physics of
``lidar_returns`` (beam-spread mixed pixels, weather clutter, and detection
from surface reflectivity and atmospheric extinction), then ``process``:

- **Range noise**: σ(r) = ``noise_std_m`` + ``noise_per_m`` · r, along the ray.
- **Dropouts**: each return is lost with probability
  p(r) = p₀ + (1 − p₀)·(r / r_max)^n (``dropout_probability`` p₀,
  ``far_dropout_exponent`` n; n = 0 means no range dependence), standing in
  for weak returns and missed pulses.
- **Motion distortion**: a real lidar sweeps over ``scan_time_s`` while the
  vehicle moves, but Gazebo captures the whole scan at once. Point i is
  assigned time offset Δᵢ = (aᵢ − ½)·T before or after mid-scan, aᵢ in [0, 1)
  being its azimuth sweep fraction, and is expressed in the sensor frame at
  that moment: ``p' = R(−ω·Δᵢ) (p − v·Δᵢ)`` with v, ω the sensor's
  translational and angular velocity in sensor axes. Removing it again
  ("deskewing") is left to perception code, as on real hardware.

Seeded and local RNG; inputs come from simulator truth (scan and motion),
outputs only from the noisy model.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .lidar_returns import (
    BeamReturns,
    LidarScan,
    _mix,
    beam_points,
    decode_scan,
    detected,
    edge_split,
    extinction_per_m,
    scan_points,
    select_returns,
    weather_clutter,
)

CLEAR_AIR = {
    "visibility_m": None,
    "lidar_clutter_per_m": 0.0,
    "lidar_clutter_max_range_m": 10.0,
    "lidar_clutter_reflectivity": 0.05,
}


class LidarEffects:
    """Apply configured effects to scans (``sensors.lidar``).

    ``weather`` is the ``environment.weather`` section (clear air if None).
    """

    def __init__(
        self, lidar: Mapping[str, Any], weather: Mapping[str, Any] | None = None
    ) -> None:
        effects = lidar["effects"]
        weather = CLEAR_AIR if weather is None else weather
        self._divergence = effects["beam_divergence_rad"]
        self._edge_jump = effects["edge_min_jump_m"]
        self._detection = effects["detection"]
        self._alpha = extinction_per_m(weather["visibility_m"])
        self._clutter_rate = weather["lidar_clutter_per_m"]
        self._clutter_range = weather["lidar_clutter_max_range_m"]
        self._clutter_rho = weather["lidar_clutter_reflectivity"]
        self._mode = effects["return_mode"]
        self._resolution = effects["range_resolution_m"]
        self._noise_std = effects["noise_std_m"]
        self._noise_per_m = effects["noise_per_m"]
        self._dropout = effects["dropout_probability"]
        self._exponent = effects["far_dropout_exponent"]
        self._scan_time = effects["scan_time_s"]
        self._max_range = lidar["max_range_m"]
        self._rng = np.random.default_rng(effects["seed"])
        self.enabled = bool(
            lidar["enabled"]
            and (
                self._noise_std
                or self._noise_per_m
                or self._dropout
                or self._exponent
                or self._scan_time
                or self._divergence
                or self._detection["enabled"]
                or self._clutter_rate
                or self._mode != "first"
            )
        )
        self.last_reflectivity = np.zeros(0)
        self.second_returns = 0
        self.points_in = 0
        self.points_out = 0
        self.beams = 0
        self.mixed = 0
        self.clutter = 0
        self.undetected = 0

    def process_scan(
        self,
        message: Any,
        velocity_sensor_mps: ArrayLike,
        rates_sensor_rad_s: ArrayLike,
    ) -> NDArray[np.float64]:
        """Full pipeline for a Gazebo LaserScan; returns sensor-frame points.

        The reported reflectivity of each returned point is left in
        ``last_reflectivity`` (same order).
        """
        scan = decode_scan(message, self._detection["default_reflectivity"])
        returns = self._beam_returns(scan)
        points, fraction = beam_points(scan, returns.beam, returns.range_m)
        points, reflectivity = self._process(
            points,
            fraction,
            velocity_sensor_mps,
            rates_sensor_rad_s,
            returns.reflectivity,
        )
        self.last_reflectivity = reflectivity
        return points

    def _candidates(self, scan: LidarScan):
        """Per-beam candidate returns (N×3 range, strength before atmosphere)."""
        own_r, own_rho = scan.ranges_m, scan.reflectivity
        share = np.ones(own_r.shape)
        other_r = np.full(own_r.shape, np.inf)
        other_rho = np.zeros(own_r.shape)
        if self._divergence:
            # Along each channel's sweep, then across channels (later wins).
            h = edge_split(
                own_r,
                own_rho,
                scan.azimuth_step_rad,
                self._divergence,
                self._edge_jump,
                self._rng,
            )
            v = [
                a.T
                for a in edge_split(
                    own_r.T,
                    own_rho.T,
                    scan.elevation_step_rad,
                    self._divergence,
                    self._edge_jump,
                    self._rng,
                )
            ]
            mixed = h[3] | v[3]
            share = np.where(v[3], v[0], h[0])
            other_r = np.where(v[3], v[1], h[1])
            other_rho = np.where(v[3], v[2], h[2])
            self.mixed += int(np.count_nonzero(mixed))
        # Surfaces closer than the range resolution merge into one ghost return.
        with np.errstate(invalid="ignore"):
            merge = np.isfinite(other_r) & (
                True
                if self._resolution is None
                else np.abs(own_r - other_r) < self._resolution
            )
        merged_r, merged_rho = _mix(own_r, own_rho, other_r, other_rho, share)
        first_r = np.where(merge, merged_r, own_r)
        first_s = np.where(merge, merged_rho, share * own_rho)
        second_r = np.where(merge, np.inf, other_r)
        second_s = np.where(merge, 0.0, (1 - share) * other_rho)
        clutter_r = np.full(own_r.shape, np.inf)
        if self._clutter_rate:
            # Backscatter only in front of the nearest hard surface.
            hard = np.minimum(first_r, second_r)
            clutter_r, clutter = weather_clutter(
                hard,
                self._clutter_rate,
                self._clutter_range,
                scan.range_min_m,
                self._rng,
            )
            clutter_r = np.where(clutter, clutter_r, np.inf)
        ranges = np.stack([first_r, second_r, clutter_r], axis=-1).reshape(-1, 3)
        strength = np.stack(
            [first_s, second_s, np.full(own_r.shape, self._clutter_rho)], axis=-1
        ).reshape(-1, 3)
        return ranges, strength

    def _beam_returns(self, scan: LidarScan) -> BeamReturns:
        """Detected, selected returns per beam (see lidar_returns.py)."""
        self.beams += scan.ranges_m.size
        ranges, strength = self._candidates(scan)
        with np.errstate(invalid="ignore", over="ignore"):
            reported = strength * np.exp(-2 * self._alpha * ranges)
        valid = scan.in_range(ranges)
        if self._detection["enabled"]:
            seen = detected(ranges, strength, self._alpha, self._detection)
            undetected_beams = valid.any(axis=1) & ~(valid & seen).any(axis=1)
            self.undetected += int(np.count_nonzero(undetected_beams))
            valid &= seen
        parts = select_returns(ranges, reported, valid, self._mode)
        beam = np.concatenate([b for b, _ in parts])
        candidate = np.concatenate([c for _, c in parts])
        self.clutter += int(np.count_nonzero(candidate == 2))
        self.second_returns += len(parts[1][0]) if len(parts) > 1 else 0
        return BeamReturns(
            beam,
            ranges[beam, candidate],
            np.nan_to_num(reported[beam, candidate]),
            candidate,
        )

    def process(
        self,
        points_sensor: NDArray[np.float64],
        azimuth_fraction: NDArray[np.float64],
        velocity_sensor_mps: ArrayLike,
        rates_sensor_rad_s: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return the surviving, perturbed points (N×3, sensor axes).

        ``azimuth_fraction`` gives each point's position in the sweep, [0, 1).
        """
        points, _ = self._process(
            points_sensor,
            azimuth_fraction,
            velocity_sensor_mps,
            rates_sensor_rad_s,
            np.zeros(len(points_sensor)),
        )
        return points

    def _process(self, points_sensor, azimuth_fraction, velocity, rates, values):
        """Noise, dropouts, and distortion; ``values`` follow the kept points."""
        points = np.asarray(points_sensor, float)
        values = np.asarray(values, float)
        self.points_in += len(points)
        if len(points) == 0:
            return points, values
        ranges = np.linalg.norm(points, axis=1)
        # Draw all random numbers every scan so sequences depend only on scans.
        noise = self._rng.normal(size=len(points))
        keep_draw = self._rng.random(len(points))
        sigma = self._noise_std + self._noise_per_m * ranges
        points = points * ((ranges + sigma * noise) / ranges)[:, None]
        p_drop = self._dropout
        if self._exponent:
            p_drop = (
                p_drop
                + (1 - p_drop)
                * np.clip(ranges / self._max_range, 0, 1) ** self._exponent
            )
        keep = keep_draw >= p_drop
        points, fraction = points[keep], np.asarray(azimuth_fraction, float)[keep]
        values = values[keep]
        if self._scan_time and len(points):
            points = self._distort(points, fraction, velocity, rates)
        self.points_out += len(points)
        return points, values

    def _distort(self, points, fraction, velocity, rates):
        delta = (fraction - 0.5) * self._scan_time
        velocity, rates = np.asarray(velocity, float), np.asarray(rates, float)
        shifted = points - delta[:, None] * velocity
        return Rotation.from_rotvec(-delta[:, None] * rates).apply(shifted)

    def summary(self) -> dict[str, Any]:
        kept = self.points_out / self.points_in if self.points_in else None
        return dict(
            enabled=self.enabled,
            points_in=self.points_in,
            points_kept_fraction=kept,
            beams=self.beams,
            mixed_edge_returns=self.mixed,
            weather_clutter_returns=self.clutter,
            undetected_returns=self.undetected,
            second_returns=self.second_returns,
            return_mode=self._mode,
            extinction_per_m=self._alpha,
        )


def sensor_frame_scan(message: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Valid returns of a Gazebo LaserScan as sensor-frame points plus sweep fraction.

    Uses the same row-major channel layout as ``sensors.gazebo.lidar_points``.
    """
    scan = decode_scan(message)
    return scan_points(scan, scan.ranges_m, scan.in_range(scan.ranges_m))
