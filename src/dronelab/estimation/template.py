"""Template estimator plug-in: copy this file to start your own.

Select it with ``algorithms.estimator: dronelab.estimation.template:TemplateEstimator``
(or your copy's ``package.module:Class``), and test it with
``dronelab.testing.plugins.check_estimator`` and ``dronelab-replay``.

The algorithm is deliberately simple, a complementary filter, so the
structure stays visible:

- attitude: integrate the gyro (body rates), and pull the tilt slowly
  towards the accelerometer's gravity direction, which is only valid when
  the vehicle is not accelerating much (gain ``tilt_gain`` per second);
- position and velocity: blend predicted motion with each GPS fix
  (``gps_gain``) in the local ENU frame around ``context.gps_reference``;
  between fixes, coast at the last velocity;
- biases: not estimated (reported as zero);
- covariance: a fixed diagonal from ``position_std_m`` and friends, not a
  propagated uncertainty, so NEES from this filter is not meaningful.

What every plug-in must respect (see estimation/interface.py):

- inputs arrive as ``Sample``s in delivery order; use ``sample.sim_time_ns``,
  never wall time;
- frames: IMU readings are in sensor axes; rotate with the declared mount
  (``context.sensors["imu"].mount_quaternion_xyzw``);
- never use simulator truth: the context and the samples are all you get;
- ``latest()`` returns the newest ``Sample[EstimatedState]`` valid at its
  ``sim_time_ns``; ``summary()`` returns plain JSON data;
- ``context.log(name, sim_time_ns, values, labels)`` records debug series.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.algorithms import AlgorithmContext
from ..core.measurements import EstimatedState, Sample
from ..sensors.geodesy import gps_to_enu
from .interface import Innovation

DEFAULTS = dict(
    tilt_gain=0.5,  # 1/s: how fast the tilt follows the accelerometer
    gps_gain=0.3,  # fraction of each GPS position error corrected
    position_std_m=1.0,  # reported (not propagated) uncertainties
    velocity_std_mps=0.5,
    attitude_std_deg=5.0,
    initial_position_m=[0.0, 0.0, 0.0],
)


class TemplateEstimator:
    """Complementary IMU/GPS filter following the ``StateEstimator`` contract."""

    inputs = ("imu", "gps")

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"TemplateEstimator: unknown options {sorted(unknown)}")
        self.options = {**DEFAULTS, **options}
        for kind in self.inputs:
            if kind not in context.sensors:
                raise ValueError(f"TemplateEstimator needs the {kind} sensor")
        self._context = context
        self._mount = Rotation.from_quat(context.sensors["imu"].mount_quaternion_xyzw)
        self._attitude = Rotation.identity()  # body → world
        self._position = np.asarray(self.options["initial_position_m"], float)
        self._velocity = np.zeros(3)
        self._last_imu_ns: int | None = None
        self._last_gps: tuple[int, np.ndarray] | None = None
        self._latest: Sample[EstimatedState] | None = None
        self._sequence = 0
        self._gps_fixes = 0
        self._covariance = self._reported_covariance()

    def on_sample(self, kind: str, sample: Sample[Any]) -> None:
        if kind == "imu":
            self._on_imu(sample)
        elif kind == "gps":
            self._on_gps(sample)

    def _on_imu(self, sample: Sample[Any]) -> None:
        reading = sample.payload
        # Calibrated values when an IMU calibration is active, else raw.
        rate = self._mount.apply(reading.best_angular_rate_rad_s)
        force = self._mount.apply(reading.best_specific_force_mps2)
        if self._last_imu_ns is not None:
            dt = (sample.sim_time_ns - self._last_imu_ns) * 1e-9
            self._attitude = self._attitude * Rotation.from_rotvec(rate * dt)
            self._correct_tilt(force, dt)
            self._position = self._position + self._velocity * dt
        self._last_imu_ns = sample.sim_time_ns
        self._publish(sample.sim_time_ns)

    def _correct_tilt(self, specific_force_body: np.ndarray, dt: float) -> None:
        """Rotate the estimate slightly so body 'up' matches measured gravity."""
        measured_up = self._attitude.apply(specific_force_body)
        norm = np.linalg.norm(measured_up)
        if norm < 1e-6:
            return
        axis = np.cross(measured_up / norm, [0.0, 0.0, 1.0])
        correction = Rotation.from_rotvec(self.options["tilt_gain"] * dt * axis)
        self._attitude = correction * self._attitude

    def _on_gps(self, sample: Sample[Any]) -> None:
        position = gps_to_enu(sample.payload, self._context.gps_reference)
        if self._last_gps is not None:
            then_ns, then_position = self._last_gps
            dt = (sample.sim_time_ns - then_ns) * 1e-9
            if dt > 0:
                self._velocity = (position - then_position) / dt
        self._last_gps = (sample.sim_time_ns, position)
        error = position - self._position
        self._position = self._position + self.options["gps_gain"] * error
        self._gps_fixes += 1
        self._context.log(
            "template/gps_error",
            sample.sim_time_ns,
            list(error),
            ["east", "north", "up"],
        )

    def _reported_covariance(self) -> np.ndarray:
        o = self.options
        std = np.repeat(
            [
                o["position_std_m"],
                o["velocity_std_mps"],
                np.radians(o["attitude_std_deg"]),
                0.1,  # biases are not estimated: generous fixed values
                0.01,
            ],
            3,
        )
        return np.diag(std**2)

    def _publish(self, sim_time_ns: int) -> None:
        self._sequence += 1
        state = EstimatedState(
            self._position,
            self._velocity,
            self._attitude.as_quat(),
            np.zeros(3),
            np.zeros(3),
            self._covariance,
        )
        self._latest = Sample(
            "estimates/template", self._sequence, 0, sim_time_ns, 0, "world", state
        )

    def latest(self) -> Sample[EstimatedState] | None:
        return self._latest

    def drain_innovations(self) -> list[Innovation]:
        return []  # a filter with a measurement model would report NIS here

    def summary(self) -> dict[str, Any]:
        return dict(gps_fixes=self._gps_fixes, options=self.options)
