"""Reusable plumbing for inertial (IMU-driven) estimators.

Write only the filter: a class with ``predict``, ``state``, and one
``update_<kind>`` method per measurement it fuses (``InertialFilter``,
filter.py). ``InertialNode`` (node.py) handles everything else for any such
filter: sample checks, IMU mounts and held readings, prediction up to each
measurement's time, measurement conversion (measurements.py), and estimate
publication. ``InertialEstimator`` (estimator.py) is the ``StateEstimator``
plug-in around it, selected as ``algorithms.estimator: inertial`` with
``estimator_options.filter: "package.module:Class"``. The built-in
GPS/IMU EKF (estimation/gps_imu_ekf/) uses the same pieces.
"""

from .filter import InertialFilter
from .measurements import BaroAltitude, GpsFix, MagneticField
from .node import InertialNode

__all__ = ["BaroAltitude", "GpsFix", "InertialFilter", "InertialNode", "MagneticField"]
