"""Reusable plumbing for recursive (predict/update) estimators.

Write only the filter: a class with one ``predict_<kind>`` for the stream
that drives it (e.g. the IMU, or wheel encoders), one ``update_<kind>`` per
measurement it fuses, and ``state`` (``RecursiveFilter``, protocol.py).
``FilterNode`` (node.py) handles everything else for any such filter:
sample checks, held driving readings, prediction up to each measurement's
time, measurement conversion (measurements.py: mounts, ENU, wheel
geometry), and estimate publication. ``FilterEstimator`` (estimator.py) is
the ``StateEstimator`` plug-in around it, selected as
``algorithms.estimator: filter`` with ``estimator_options.filter:
"package.module:Class"``. The built-in GPS/IMU EKF
(estimation/gps_imu_ekf/) is just such a filter; ``algorithms.estimator:
gps_imu_ekf`` is this estimator running it. Examples:
estimation/examples/imu_gps_filter.py (IMU-driven) and
wheel_odometry_filter.py (wheel-encoder-driven).

Shared filter mathematics: ``StateLayout`` (state_layout.py, named
error-state blocks), measurement models (models.py: GPS position and
velocity, barometer, magnetometer, Jacobians placed by block name), and
``kalman_update`` (kalman.py, gated Joseph-form correction).
"""

from .kalman import KalmanCorrection, checked_covariance, kalman_update
from .measurements import BaroAltitude, GpsFix, ImuInput, MagneticField, WheelMotion
from .models import (
    LinearizedMeasurement,
    baro_altitude,
    gps_position,
    gps_velocity,
    magnetometer_field,
)
from .node import FilterNode
from .protocol import RecursiveFilter
from .state_layout import StateLayout

__all__ = [
    "BaroAltitude",
    "FilterNode",
    "GpsFix",
    "ImuInput",
    "KalmanCorrection",
    "LinearizedMeasurement",
    "MagneticField",
    "RecursiveFilter",
    "StateLayout",
    "WheelMotion",
    "baro_altitude",
    "checked_covariance",
    "gps_position",
    "gps_velocity",
    "kalman_update",
    "magnetometer_field",
]
