"""Timestamped sample handling for any ``InertialFilter``.

``InertialNode`` turns bus samples into filter calls and publishes the
estimate; the filter never sees streams, frames, or timestamps:

- every sample is checked: payload type, frame, epoch, stream identity,
  consecutive sequence, strictly increasing time per stream (ValueError);
- IMU readings (driver-calibrated values when present) are rotated by the
  IMU mount into body axes and **held from their interval start**: a new
  IMU sample first predicts with the previous reading up to its own time,
  then becomes the held reading. IMU intervals over 0.1 s are rejected;
  the IMU must be at the body origin (no lever-arm compensation);
- a measurement newer than the filter time first predicts with the held
  reading up to its time (``allow_intermediate``, segmented propagation);
  one at another time is counted in ``rejected_timing[kind]`` and dropped,
  since there is no rewind and re-propagation. It is then converted
  (measurements.py) and passed to ``update_<kind>``;
- an estimate is published after every IMU sample and after every accepted
  correction or initialization, on ``estimates``, with the samples that
  produced it as parents. The node starts no threads; a lock serializes
  handlers.
"""

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ...core.algorithms import SensorDescription
from ...core.measurements import (
    EstimatedState,
    GpsReading,
    ImuReading,
    Sample,
    readonly_array,
)
from ...sensors.bus import STREAMS
from ...sensors.streams import Channel, Publisher
from ..interface import Innovation
from .filter import UPDATE_PREFIX, update_kinds
from .measurements import MeasurementConverter

StreamReference: TypeAlias = tuple[str, int]
StreamIdentity: TypeAlias = tuple[str, int, int]
ImuValues: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]
MAX_IMU_INTERVAL_NS = 100_000_000


@dataclass(frozen=True)
class MeasurementOutcome:
    """What one measurement did: ``used`` (time-aligned and passed to the
    filter), the filter's ``innovation`` (None: initialization or not
    used), and the estimate published after it (None when rejected)."""

    used: bool
    innovation: Innovation | None
    estimate: Sample[EstimatedState] | None

    @property
    def accepted(self) -> bool:
        """The state changed: an accepted correction or an initialization."""
        return self.estimate is not None


class InertialNode:
    """Drive ``filter_`` from timestamped samples (see module docs).

    ``name`` names the output stream (``algorithms/<name>/state``);
    ``gps_reference`` is the local ENU origin; ``sensors`` the context's
    sensor descriptions (mounts); ``converter_options`` the
    ``MeasurementConverter`` options.
    """

    def __init__(
        self,
        filter_: Any,
        *,
        name: str,
        gps_reference: GpsReading,
        sensors: Mapping[str, SensorDescription] | None = None,
        imu_quaternion_body_xyzw: ArrayLike | None = None,
        converter_options: Mapping[str, Any] | None = None,
        epoch: int = 0,
        allow_intermediate: bool = True,
    ) -> None:
        sensors = dict(sensors or {})
        if not isinstance(gps_reference, GpsReading):
            raise TypeError("Explicit WGS84 GPS reference required")
        self.filter = filter_
        self.kinds = update_kinds(filter_)
        self.converter = MeasurementConverter(
            sensors, gps_reference, converter_options or {}
        )
        self.imu_rotation = _imu_rotation(sensors.get("imu"), imu_quaternion_body_xyzw)
        self._publisher: Publisher[EstimatedState] = Publisher(
            f"algorithms/{name}/state",
            EstimatedState,
            frame_id="local_enu",
            epoch=epoch,
        )
        self.estimates: Channel[EstimatedState] = self._publisher.channel
        self.outputs: Mapping[str, Channel[EstimatedState]] = MappingProxyType(
            {"state": self.estimates}
        )
        self.allow_intermediate = allow_intermediate
        self.rejected_timing = {kind: 0 for kind in self.kinds}
        self._epoch = epoch
        self._lock = threading.RLock()
        # A non-None filter time implies a held IMU reading exists.
        self._time: int | None = None
        self._held: ImuValues | None = None
        self._held_ref: StreamReference | None = None
        self._output_ref: StreamReference | None = None
        self._last: dict[str, StreamIdentity] = {}
        self._closed = False

    @property
    def time_ns(self) -> int | None:
        """The filter's current time (the newest IMU or measurement time)."""
        return self._time

    def _check(
        self, kind: str, sample: Sample[Any], payload_type: type
    ) -> StreamIdentity:
        if self._closed:
            raise RuntimeError("Inertial node closed")
        if (
            sample.epoch != self._epoch
            or sample.frame_id != STREAMS[kind][1]
            or not isinstance(sample.payload, payload_type)
        ):
            raise ValueError("Input payload, epoch, or frame mismatch")
        previous = self._last.get(kind)
        if previous is not None:
            stream, sequence, timestamp = previous
            if (
                sample.stream_id != stream
                or sample.sequence != sequence + 1
                or sample.sim_time_ns <= timestamp
            ):
                raise ValueError(
                    "Input stream changed, lost samples, or reversed/repeated time"
                )
        return (sample.stream_id, sample.sequence, sample.sim_time_ns)

    def on_imu(self, sample: Sample[ImuReading]) -> Sample[EstimatedState]:
        """Predict to ``sample``'s time with the held reading, hold the new
        one, and publish (module docs)."""
        with self._lock:
            identity = self._check("imu", sample, ImuReading)
            previous = self._last.get("imu")
            if (
                previous is not None
                and sample.sim_time_ns - previous[2] > MAX_IMU_INTERVAL_NS
            ):
                raise ValueError("IMU interval exceeds 0.1 seconds")
            current = (
                self.imu_rotation.apply(sample.payload.best_specific_force_mps2),
                self.imu_rotation.apply(sample.payload.best_angular_rate_rad_s),
            )
            if self._time is not None:
                dt_ns = sample.sim_time_ns - self._time
                if dt_ns > 0:
                    assert self._held is not None
                    self.filter.predict(*self._held, dt_ns * 1e-9)
                elif dt_ns < 0:
                    raise ValueError("IMU predates current estimator time")
            previous_ref = self._held_ref
            self._time = sample.sim_time_ns
            self._held = current
            self._held_ref = (sample.stream_id, sample.sequence)
            self._last["imu"] = identity
            return self._publish(
                sample, () if previous_ref is None else (previous_ref,)
            )

    def on_measurement(self, kind: str, sample: Sample[Any]) -> MeasurementOutcome:
        """Align, convert, and fuse one ``kind`` sample (module docs); raises
        ValueError for a kind the filter has no ``update_<kind>`` for."""
        if kind not in self.kinds:
            raise ValueError(
                f"{type(self.filter).__name__} has no {UPDATE_PREFIX}{kind}"
            )
        with self._lock:
            identity = self._check(kind, sample, STREAMS[kind][0])
            self._last[kind] = identity
            t = sample.sim_time_ns
            if self.allow_intermediate and self._time is not None and t > self._time:
                assert self._held is not None
                self.filter.predict(*self._held, (t - self._time) * 1e-9)
                self._time = t
            if self._time is None or t != self._time:
                self.rejected_timing[kind] += 1
                return MeasurementOutcome(False, None, None)
            measurement = self.converter.convert(kind, sample.payload)
            innovation = getattr(self.filter, UPDATE_PREFIX + kind)(measurement)
            if innovation is not None and not innovation.accepted:
                return MeasurementOutcome(True, innovation, None)
            parents = () if self._held_ref is None else (self._held_ref,)
            return MeasurementOutcome(True, innovation, self._publish(sample, parents))

    def _publish(
        self, parent: Sample[Any], extra: tuple[StreamReference, ...]
    ) -> Sample[EstimatedState]:
        parents = ((parent.stream_id, parent.sequence),) + extra
        if self._output_ref is not None:
            parents += (self._output_ref,)
        assert self._time is not None
        result = self._publisher.publish(
            self.filter.state(),
            sim_time_ns=self._time,
            parents=tuple(dict.fromkeys(parents)),
        )
        self._output_ref = (result.stream_id, result.sequence)
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._publisher.close()


def _imu_rotation(
    sensor: SensorDescription | None, quaternion: ArrayLike | None
) -> Rotation:
    """Sensor-to-body rotation of the IMU: the explicit ``quaternion``, else
    the sensor description's mount; raises ValueError for a mount position
    off the body origin or an unnormalized quaternion."""
    if sensor is not None and any(sensor.mount_position_m):
        raise ValueError(
            "Inertial filters require a body-origin IMU; lever-arm acceleration "
            "correction is not implemented"
        )
    if quaternion is None:
        quaternion = (0, 0, 0, 1) if sensor is None else sensor.mount_quaternion_xyzw
    q = readonly_array(quaternion, (4,))
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("IMU mount quaternion must be normalized")
    return Rotation.from_quat(q)
