"""What a plug-in algorithm may know about the vehicle, without truth.

``AlgorithmContext`` is the only information besides sensor samples that an
estimator or controller receives: sensor descriptions (frames, nominal
rates, mounts, camera intrinsics, as from a calibration file), the nominal
vehicle properties
(as designed, not randomized truth), and the local navigation origin. It
deliberately excludes simulator truth and the configured simulator error
models: configured noise describes the simulator, never knowledge an
algorithm should have.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .measurements import GpsReading
from .vehicle import DroneProperties
from .track import Track
from .camera import SENSOR_FROM_OPTICAL, CameraCalibration
from .wheeled_vehicle import WheeledProperties


@dataclass(frozen=True)
class SensorDescription:
    """One sensor stream: frame, nominal rate, and mount relative to the body.

    The mount rotation (xyzw) rotates sensor-axis vectors into body axes.
    """

    kind: str
    frame_id: str
    rate_hz: float | None
    mount_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Sensor calibration as from a calibration file (e.g. the magnetometer's
    # offsets, correction, and local field), or None.
    calibration: Mapping[str, Any] | None = None
    # Camera intrinsics and optical frame (camera streams only).
    camera: CameraCalibration | None = None

    def to_body(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rotation (3×3, sensor axes → body) and mount position (body, m)."""
        rotation = Rotation.from_quat(self.mount_quaternion_xyzw).as_matrix()
        return rotation, np.array(self.mount_position_m, float)


def extrinsics(
    source: SensorDescription,
    target: SensorDescription,
    *,
    source_optical: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(R, t)`` with ``p_target = R p_source + t`` between two sensors' axes.

    ``source_optical`` takes the source in its camera optical frame (z
    forward, x right, y down; ``core/camera.py``) instead of its Gazebo
    sensor axes: e.g. ``extrinsics(camera, imu, source_optical=True)`` is
    the camera-to-IMU transform of visual-inertial odometry. Raises
    ``ValueError`` if ``source_optical`` is set for a non-camera.
    """
    rotation_s, position_s = source.to_body()
    if source_optical:
        if source.camera is None:
            raise ValueError(f"{source.kind} has no camera optical frame")
        rotation_s = rotation_s @ SENSOR_FROM_OPTICAL
    rotation_t, position_t = target.to_body()
    return rotation_t.T @ rotation_s, rotation_t.T @ (position_s - position_t)


DebugLog = Callable[[str, int, Sequence[float], Sequence[str] | None], None]


def ignore_debug(
    name: str,
    sim_time_ns: int,
    values: Sequence[float],
    labels: Sequence[str] | None = None,
) -> None:
    """Default debug channel: discard (algorithms may always call ``log``)."""


@dataclass(frozen=True)
class AlgorithmContext:
    """Static information for algorithm construction (see module docs).

    ``log(name, sim_time_ns, values, labels=None)`` records a plug-in's own
    time series (innovations, internal states, gains) for inspection: to
    ``algorithm-debug.jsonl`` and Rerun (``algorithms/<name>``) in Gazebo
    runs, and to the replay output offline. Stamp entries with the sample
    time they describe. ``values`` must be finite numbers; ``labels``
    (default ``0, 1, ...``) must match them in length.
    """

    gps_reference: GpsReading
    # Nominal multirotor (for wheeled robots the unused multirotor defaults;
    # use ``wheeled`` there).
    vehicle: DroneProperties
    sensors: Mapping[str, SensorDescription] = field(default_factory=dict)
    gravity_mps2: float = 9.81
    log: DebugLog = field(
        default=ignore_debug, compare=False, repr=False, metadata={"serialize": False}
    )
    # Nominal wheeled vehicle (rover, car, tractor runs), else None.
    wheeled: WheeledProperties | None = field(
        default=None, metadata={"serialize": False}
    )
    # The scene's race track (map knowledge, core/track.py), else None.
    track: Track | None = field(default=None, metadata={"serialize": False})

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensors", MappingProxyType(dict(self.sensors)))
