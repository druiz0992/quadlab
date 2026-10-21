"""What a plug-in algorithm may know about the vehicle, without truth.

``AlgorithmContext`` is the only information besides sensor samples that an
estimator or controller receives: sensor descriptions (frames, nominal
rates, mounts, as from a calibration file), the nominal vehicle properties
(as designed, not randomized truth), and the local navigation origin. It
deliberately excludes simulator truth and the configured simulator error
models: configured noise describes the simulator, never knowledge an
algorithm should have.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from types import MappingProxyType

from .measurements import GpsReading
from .vehicle import DroneProperties


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
    vehicle: DroneProperties
    sensors: Mapping[str, SensorDescription] = field(default_factory=dict)
    gravity_mps2: float = 9.81
    log: DebugLog = field(
        default=ignore_debug, compare=False, repr=False, metadata={"serialize": False}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensors", MappingProxyType(dict(self.sensors)))
