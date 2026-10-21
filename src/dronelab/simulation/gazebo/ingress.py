"""Gazebo transport ingress: subscriptions into one bounded, counted queue.

Gazebo delivers each topic on its own transport thread. Every callback
puts ``(kind, message, wall_receipt_s)`` into a single bounded queue that
the mission thread drains, so all processing happens serially. When the
queue is full the message is dropped and counted per kind (never silently):
``dropped`` ends up in ``summary.json`` and the health verdict. The wall
receipt time is used only for real-time staleness checks, never as a
measurement timestamp.

In lockstep, ``LockstepOrderer`` turns the arrival order into a
deterministic processing order (see its docs).

Also builds the ``gz sim`` command line. Gazebo bindings are imported
lazily so world generation works without them.
"""

from collections import Counter, deque
from collections.abc import Callable, Mapping
from pathlib import Path
from queue import Empty, Full, Queue
import time
from typing import Any

from .vehicle import ROTOR_STATE_TOPIC

QUEUE_CAPACITY = 4000

# Transported sensors: config section, message type name, topic.
SENSOR_TOPICS = (
    ("imu", "imu_pb2.IMU", "/dronelab/imu"),
    ("gps", "navsat_pb2.NavSat", "/dronelab/gps"),
    ("baro", "fluid_pressure_pb2.FluidPressure", "/dronelab/baro"),
    ("magnetometer", "magnetometer_pb2.Magnetometer", "/dronelab/magnetometer"),
    ("rangefinder", "laserscan_pb2.LaserScan", "/dronelab/rangefinder"),
    ("contacts", "contacts_pb2.Contacts", "/dronelab/contacts"),
    ("camera", "image_pb2.Image", "/dronelab/camera"),
    ("lidar", "laserscan_pb2.LaserScan", "/dronelab/lidar"),
)


def _message_class(name: str) -> type:
    """``"imu_pb2.IMU"`` → ``gz.msgs10.imu_pb2.IMU``."""
    import importlib

    module, cls = name.split(".")
    return getattr(importlib.import_module(f"gz.msgs10.{module}"), cls)


def subscriptions(
    sensors: Mapping[str, Any], esc_telemetry: bool
) -> list[tuple[str, str, str]]:
    """``(message type, topic, kind)`` for truth, enabled sensors, and ESCs."""
    topics = [
        ("odometry_pb2.Odometry", "/dronelab/odometry", "odometry"),
        ("pose_v_pb2.Pose_V", "/world/dronelab/pose/info", "poses"),
    ]
    if esc_telemetry:
        topics.append(("model_pb2.Model", ROTOR_STATE_TOPIC, "esc"))
    for kind, message_type, topic in SENSOR_TOPICS:
        if sensors[kind]["enabled"]:
            topics.append((message_type, topic, kind))
    return topics


class MessageIngress:
    """Bounded queue fed by transport callbacks (see module docs).

    ``clock()`` gives wall seconds for receipt stamps (injectable for tests).
    """

    def __init__(
        self,
        capacity: int = QUEUE_CAPACITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queue: Queue = Queue(maxsize=capacity)
        self.dropped: Counter = Counter()
        self._clock = clock

    def callback(self, kind: str) -> Callable[[Any], None]:
        """Transport callback queuing messages of one kind."""

        def receive(message: Any) -> None:
            try:
                self.queue.put_nowait((kind, message, self._clock()))
            except Full:
                self.dropped[kind] += 1

        return receive

    def subscribe(self, node: Any, topics: list[tuple[str, str, str]]) -> None:
        """Subscribe ``node`` to every topic; ValueError if one fails."""
        for message_type, topic, kind in topics:
            if not node.subscribe(
                _message_class(message_type), topic, self.callback(kind)
            ):
                raise ValueError(f"Unable to subscribe to {topic}")

    def get(self, timeout_s: float) -> tuple[str, Any, float]:
        """Next ``(kind, message, wall_receipt_s)``; raises queue.Empty."""
        return self.queue.get(timeout=timeout_s)


def gz_sim_command(
    world: Path, *, lockstep: bool, headless: bool, render_sensors: bool
) -> list[str]:
    """``gz sim`` command line; lockstep starts paused (Python steps it)."""
    command = ["gz", "sim", "-v", "3"] if lockstep else ["gz", "sim", "-r", "-v", "3"]
    if headless:
        command.append("-s")
        if render_sensors:
            command.append("--headless-rendering")
    command.append(str(world))
    return command


# Processing order of messages with equal timestamps: measurements first,
# truth odometry last, so data stamped T is available to control at T.
KIND_PRIORITY = {
    kind: index
    for index, kind in enumerate(
        (
            "imu",
            "gps",
            "baro",
            "magnetometer",
            "rangefinder",
            "esc",
            "camera",
            "lidar",
            "poses",
            "contacts",
            "odometry",
        )
    )
}


def scheduled_stamp_ns(period_ns: float, step_ns: int, barrier_ns: int) -> int:
    """Stamp of a sensor's last sample at or before ``barrier_ns``.

    Gazebo samples a sensor of period P at the first physics step at or
    after k·P (k = 0, 1, ...), and the first step is at ``step_ns`` (measured
    with gz-sim 8: a 5 ms IMU on a 2 ms step gives stamps 2, 6, 10, 16 ms).
    """
    k = int(barrier_ns / period_ns + 1e-9)
    steps = -(-round(k * period_ns) // step_ns)  # ceil to the step grid
    return max(step_ns, steps * step_ns)


class LockstepOrderer:
    """Deterministic message order for lockstep runs.

    Gazebo delivers topics on separate transport threads, so their arrival
    order varies between otherwise identical runs; processing must not
    depend on it. Each odometry message ends a lockstep period at T. Before
    it is processed, the orderer waits until every scheduled sensor stream
    (``periods_ns``) has delivered its last sample stamped ≤ T, then
    releases all buffered messages stamped ≤ T in (stamp, kind priority,
    arrival) order, odometry last.

    Each topic arrives in stamp order (measured; only the interleaving of
    topics varies). Gazebo republishes some topics while paused (joint
    states, poses, contacts); a message whose stamp is not newer than the
    last one of its kind is a duplicate, dropped and counted. Unscheduled streams (poses,
    contacts) are ordered but not waited for. If a scheduled sample does
    not arrive within ``wait_timeout_s`` (wall), the period is released
    anyway and counted in ``barrier_timeouts`` (determinism is then not
    guaranteed for that period). ``get`` has the ``MessageIngress.get``
    contract.
    """

    def __init__(
        self,
        ingress: MessageIngress,
        periods_ns: Mapping[str, float],
        step_ns: int,
        *,
        wait_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ingress = ingress
        self._periods_ns = dict(periods_ns)
        self._step_ns = step_ns
        self._wait_timeout_s = wait_timeout_s
        self._clock = clock
        self._buffer: list[tuple[int, int, int, str, Any, float]] = []
        self._ready: deque = deque()
        self._barriers: deque[tuple[int, float]] = deque()
        self._last_ns: dict[str, int] = {}
        self._sequence = 0
        self.duplicates: Counter = Counter()
        self.barrier_timeouts = 0
        self.timeout_streams: Counter = Counter()

    @property
    def dropped(self) -> Counter:
        """Messages lost at the transport queue (see ``MessageIngress``)."""
        return self._ingress.dropped

    def get(self, timeout_s: float) -> tuple[str, Any, float]:
        """Next message in deterministic order; raises queue.Empty on timeout."""
        deadline = self._clock() + timeout_s
        while not self._ready:
            self._release_if_complete()
            if self._ready:
                break
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise Empty
            try:
                kind, message, received_at = self._ingress.get(min(remaining, 0.01))
            except Empty:
                continue
            self._accept(kind, message, received_at)
        return self._ready.popleft()

    def _accept(self, kind: str, message: Any, received_at: float) -> None:
        from .messages import timestamp_ns

        stamp = timestamp_ns(message)
        if stamp <= self._last_ns.get(kind, -1):
            self.duplicates[kind] += 1
            return
        self._last_ns[kind] = stamp
        self._sequence += 1
        priority = KIND_PRIORITY.get(kind, len(KIND_PRIORITY) - 1)
        self._buffer.append(
            (stamp, priority, self._sequence, kind, message, received_at)
        )
        if kind == "odometry":
            self._barriers.append((stamp, self._clock()))

    def _missing(self, barrier_ns: int) -> list[str]:
        """Scheduled streams whose last sample at or before the barrier is due."""
        return [
            kind
            for kind, period in self._periods_ns.items()
            if self._last_ns.get(kind, -1)
            < scheduled_stamp_ns(period, self._step_ns, barrier_ns)
        ]

    def _release_if_complete(self) -> None:
        if not self._barriers:
            return
        barrier_ns, arrived_at = self._barriers[0]
        missing = self._missing(barrier_ns)
        if missing:
            if self._clock() - arrived_at < self._wait_timeout_s:
                return
            self.barrier_timeouts += 1
            self.timeout_streams.update(missing)
        self._barriers.popleft()
        due = sorted(item for item in self._buffer if item[0] <= barrier_ns)
        self._buffer = [item for item in self._buffer if item[0] > barrier_ns]
        self._ready.extend(
            (kind, message, received) for *_, kind, message, received in due
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            duplicates=dict(self.duplicates),
            barrier_timeouts=self.barrier_timeouts,
            timeout_streams=dict(self.timeout_streams),
            pending_at_close=len(self._buffer),
        )
