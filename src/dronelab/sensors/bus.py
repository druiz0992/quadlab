"""Sensor bus: processed measurements as typed streams for algorithms.

Every measurement that algorithms may use is submitted here after the
sensor-error and fault models, with its measurement time. The bus stamps it,
holds it for a modelled delivery delay, and publishes it as an immutable
``Sample`` on ``sensors/<kind>``. Simulator truth never enters the bus.

Timing per stream (``sensor_timing.<kind>``):

- ``timestamp_offset_s``: the sensor's clock error. The sample's
  ``sim_time_ns`` is measurement time + offset (never below zero), as when a
  sensor's clock is not synchronized with the autopilot's.
- ``latency_s`` and ``jitter_s``: delivery happens at measurement time +
  latency + |N(0, jitter²)| (half-normal), never before an earlier sample of
  the same stream (streams stay first-in first-out).

Streams with different latencies arrive out of timestamp order relative to
each other (e.g. a GPS fix after newer IMU samples), as on real vehicles;
``summary()["late_samples"]`` counts samples older than one already
delivered on another stream. Gazebo publishes topics on separate transport
threads, so arrival can be slightly unordered even without modelled delay:
``reorder_window_s`` holds samples until the bus clock is that far past
their delivery time, then releases them by delivery time, timestamp, and a
fixed stream priority (IMU first). The window adds delivery delay only.
"""

from collections.abc import Iterable, Mapping
import heapq
from typing import Any

import numpy as np

from ..core.measurements import (
    BaroReading,
    CameraFrame,
    EscTelemetry,
    GpsHeadingReading,
    GpsReading,
    ImuReading,
    LidarPoints,
    MagnetometerReading,
    RangeReading,
    Sample,
)
from .airspeed import AirspeedReading
from .battery_monitor import BatteryReading
from .optical_flow import OpticalFlowReading
from .streams import Channel, Publisher

# kind: (payload type, frame ID). Order sets priority at equal times.
STREAMS: dict[str, tuple[type, str]] = {
    "imu": (ImuReading, "drone/imu"),
    "gps": (GpsReading, "wgs84"),
    "gps_heading": (GpsHeadingReading, "drone/gps_heading"),
    "baro": (BaroReading, "drone/baro"),
    "magnetometer": (MagnetometerReading, "drone/magnetometer"),
    "rangefinder": (RangeReading, "drone/rangefinder"),
    "optical_flow": (OpticalFlowReading, "drone/optical_flow"),
    "battery_monitor": (BatteryReading, "drone/battery"),
    "airspeed": (AirspeedReading, "drone/airspeed"),
    "esc": (EscTelemetry, "drone/esc"),
    "lidar": (LidarPoints, "drone/body"),
    "camera": (CameraFrame, "drone/camera"),
}
PRIORITY = {kind: index for index, kind in enumerate(STREAMS)}


class SensorBus:
    """Delay and publish processed measurements (see module docs).

    ``timing`` is the validated ``sensor_timing`` section; ``kinds`` the
    enabled streams. Call ``advance(now_ns)`` with the newest simulation
    time seen to release due samples.
    """

    def __init__(self, timing: Mapping[str, Any], kinds: Iterable[str]) -> None:
        self._timing = timing
        self._rng = np.random.default_rng(timing["seed"])
        self._window_ns = round(timing["reorder_window_s"] * 1e9)
        self._publishers: dict[str, Publisher] = {}
        for kind in kinds:
            payload_type, frame_id = STREAMS[kind]
            self._publishers[kind] = Publisher(
                f"sensors/{kind}", payload_type, frame_id=frame_id
            )
        self._heap: list[tuple[int, int, int, int, str, Any]] = []
        self._order = 0
        self._last_delivery: dict[str, int] = {}
        self._newest_delivered_ns = -1
        self.stats: dict[str, Any] = dict(
            submitted={k: 0 for k in self._publishers},
            delivered={k: 0 for k in self._publishers},
            max_delay_ms={k: 0.0 for k in self._publishers},
            late_samples=0,
        )

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self._publishers)

    def channel(self, kind: str) -> Channel:
        """Read-only channel of one stream; raises KeyError if not enabled."""
        if kind not in self._publishers:
            raise KeyError(f"Sensor stream {kind!r} is not enabled")
        return self._publishers[kind].channel

    def submit(self, kind: str, payload: Any, measured_ns: int) -> None:
        """Queue one measurement taken at ``measured_ns`` (simulation time)."""
        if kind not in self._publishers:
            return
        timing = self._timing[kind]
        stamp = max(0, measured_ns + round(timing["timestamp_offset_s"] * 1e9))
        delay_s = timing["latency_s"] + abs(self._rng.normal()) * timing["jitter_s"]
        deliver = max(
            measured_ns + round(delay_s * 1e9), self._last_delivery.get(kind, 0)
        )
        self._last_delivery[kind] = deliver
        self._order += 1
        heapq.heappush(
            self._heap,
            (deliver, stamp, PRIORITY[kind], self._order, kind, (payload, measured_ns)),
        )
        self.stats["submitted"][kind] += 1

    def advance(self, now_ns: int) -> list[tuple[str, Sample]]:
        """Publish every sample due by ``now_ns − reorder_window``."""
        released = []
        while self._heap and self._heap[0][0] <= now_ns - self._window_ns:
            deliver, stamp, _, _, kind, (payload, measured) = heapq.heappop(self._heap)
            sample = self._publishers[kind].publish(payload, sim_time_ns=stamp)
            released.append((kind, sample))
            self._record(kind, stamp, max(deliver, now_ns - self._window_ns), measured)
        return released

    def _record(self, kind: str, stamp: int, delivered_ns: int, measured: int) -> None:
        stats = self.stats
        stats["delivered"][kind] += 1
        delay = (delivered_ns - measured) * 1e-6
        stats["max_delay_ms"][kind] = max(stats["max_delay_ms"][kind], delay)
        if stamp < self._newest_delivered_ns:
            stats["late_samples"] += 1
        self._newest_delivered_ns = max(self._newest_delivered_ns, stamp)

    def close(self) -> None:
        for publisher in self._publishers.values():
            publisher.close()

    def summary(self) -> dict[str, Any]:
        return dict(self.stats, pending_at_close=len(self._heap))
