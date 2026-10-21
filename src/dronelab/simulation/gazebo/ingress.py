"""Gazebo transport ingress: subscriptions into one bounded, counted queue.

Gazebo delivers each topic on its own transport thread. Every callback
puts ``(kind, message, wall_receipt_s)`` into a single bounded queue that
the mission thread drains, so all processing happens serially. When the
queue is full the message is dropped and counted per kind (never silently):
``dropped`` ends up in ``summary.json`` and the health verdict. The wall
receipt time is used only for real-time staleness checks, never as a
measurement timestamp.

Also builds the ``gz sim`` command line. Gazebo bindings are imported
lazily so world generation works without them.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from queue import Full, Queue
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
