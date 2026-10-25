"""IMU models: errors, imperfections (saturation, timing), temperature,
faults, accelerometer vibration, and applying a calibration.

Every model that alters readings is an ``ImuPerturbation``: it maps one
(specific force m/s², angular rate rad/s) pair in sensor axes at an integer
simulation time to the perturbed pair, and reports ``enabled`` (a disabled
model leaves readings untouched). The sensor pipeline applies them as one
ordered chain (experiments/common/sensor_pipeline.py).
"""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ImuPerturbation(Protocol):
    """See module docs: ``ImuErrorModel``, ``ImuImperfections``,
    ``ImuTemperatureModel``, ``AccelerometerVibration``, ``ImuSaturation``,
    ``ImuFaults``."""

    @property
    def enabled(self) -> bool: ...

    def apply(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
