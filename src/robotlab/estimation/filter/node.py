"""Timestamped sample handling for any ``RecursiveFilter``.

``FilterNode`` turns bus samples into filter calls and publishes the
estimate; the filter never sees streams, frames, or timestamps:

- every sample is checked: payload type, frame, epoch, stream identity,
  consecutive sequence, strictly increasing time per stream (ValueError);
- the **driving** stream (the filter's ``predict_<kind>``, e.g. ``imu``) is
  converted (measurements.py) and each reading is **held from its sample
  time**: a new reading first predicts with the previous one up to its own
  time, then becomes the held reading. Intervals over
  ``max_driver_interval_s`` (default 0.1 s) are rejected (ValueError);
- a measurement newer than the filter time first predicts with the held
  reading up to its time (``allow_intermediate``, segmented propagation);
  one at another time is counted in ``rejected_timing[kind]`` and dropped,
  since there is no rewind and re-propagation. It is then converted and
  passed to ``update_<kind>``;
- an estimate is published after every driving sample and after every
  accepted correction or initialization, on ``estimates``, with the samples
  that produced it as parents. The node starts no threads; a lock
  serializes handlers.
"""

import threading
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

from numpy.typing import ArrayLike

from ...core.algorithms import SensorDescription
from ...core.measurements import EstimatedState, GpsReading, Sample
from ...core.wheeled_vehicle import WheeledProperties
from ...sensors.bus import STREAMS
from ...sensors.bus.streams import Channel, Publisher
from ..interface import Innovation
from .measurements import MeasurementConverter
from .protocol import PREDICT_PREFIX, UPDATE_PREFIX, predict_kind, update_kinds

StreamReference: TypeAlias = tuple[str, int]
StreamIdentity: TypeAlias = tuple[str, int, int]
DEFAULT_MAX_DRIVER_INTERVAL_S = 0.1


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


class FilterNode:
    """Drive ``filter_`` from timestamped samples (see module docs).

    ``name`` names the output stream (``algorithms/<name>/state``);
    ``gps_reference`` is the local ENU origin; ``sensors`` the context's
    sensor descriptions (mounts); ``wheeled`` the context's wheeled vehicle
    (wheel geometry); ``converter_options`` the ``MeasurementConverter``
    options; ``updates`` limits the fused streams to these of the filter's
    ``update_<kind>`` kinds (default: all). Raises ValueError for a filter without exactly one driving
    stream (protocol.py) or a stream its converter cannot handle.
    """

    def __init__(
        self,
        filter_: Any,
        *,
        name: str,
        gps_reference: GpsReading,
        sensors: Mapping[str, SensorDescription] | None = None,
        wheeled: WheeledProperties | None = None,
        imu_quaternion_body_xyzw: ArrayLike | None = None,
        converter_options: Mapping[str, Any] | None = None,
        epoch: int = 0,
        allow_intermediate: bool = True,
        max_driver_interval_s: float = DEFAULT_MAX_DRIVER_INTERVAL_S,
        updates: Collection[str] | None = None,
    ) -> None:
        if not isinstance(gps_reference, GpsReading):
            raise TypeError("Explicit WGS84 GPS reference required")
        if not max_driver_interval_s > 0:
            raise ValueError("max_driver_interval_s must be positive")
        self.filter = filter_
        self.driver = predict_kind(filter_)
        self.kinds = tuple(
            kind for kind in update_kinds(filter_) if updates is None or kind in updates
        )
        self.converter = MeasurementConverter(
            dict(sensors or {}),
            gps_reference,
            converter_options or {},
            kinds=(self.driver, *self.kinds),
            wheeled=wheeled,
            imu_quaternion_body_xyzw=imu_quaternion_body_xyzw,
        )
        self._predict = getattr(filter_, PREDICT_PREFIX + self.driver)
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
        self.max_driver_interval_ns = round(max_driver_interval_s * 1e9)
        self.rejected_timing = {kind: 0 for kind in self.kinds}
        self._epoch = epoch
        self._lock = threading.RLock()
        # A non-None filter time implies a held driving reading exists.
        self._time: int | None = None
        self._held: Any = None
        self._held_ref: StreamReference | None = None
        self._output_ref: StreamReference | None = None
        self._last: dict[str, StreamIdentity] = {}
        self._closed = False

    @property
    def time_ns(self) -> int | None:
        """The filter's current time (the newest driving or measurement time)."""
        return self._time

    def _check(self, kind: str, sample: Sample[Any]) -> StreamIdentity:
        if self._closed:
            raise RuntimeError("Filter node closed")
        payload_type, frame_id = STREAMS[kind]
        if (
            sample.epoch != self._epoch
            or sample.frame_id != frame_id
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

    def on_driver(self, sample: Sample[Any]) -> Sample[EstimatedState]:
        """Predict to ``sample``'s time with the held reading, hold the new
        one, and publish (module docs)."""
        with self._lock:
            identity = self._check(self.driver, sample)
            previous = self._last.get(self.driver)
            if (
                previous is not None
                and sample.sim_time_ns - previous[2] > self.max_driver_interval_ns
            ):
                raise ValueError(
                    f"{self.driver} interval exceeds "
                    f"{self.max_driver_interval_ns * 1e-9} seconds"
                )
            current = self.converter.convert(self.driver, sample.payload)
            if self._time is not None:
                dt_ns = sample.sim_time_ns - self._time
                if dt_ns > 0:
                    self._predict(self._held, dt_ns * 1e-9)
                elif dt_ns < 0:
                    raise ValueError(f"{self.driver} predates current filter time")
            previous_ref = self._held_ref
            self._time = sample.sim_time_ns
            self._held = current
            self._held_ref = (sample.stream_id, sample.sequence)
            self._last[self.driver] = identity
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
            identity = self._check(kind, sample)
            self._last[kind] = identity
            t = sample.sim_time_ns
            if self.allow_intermediate and self._time is not None and t > self._time:
                self._predict(self._held, (t - self._time) * 1e-9)
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
