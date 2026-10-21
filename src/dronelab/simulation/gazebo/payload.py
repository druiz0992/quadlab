"""Payload carried under the drone: scheduled or keyboard drop and pickup.

The payload is its own Gazebo model attached to ``drone::base_link`` by the
DetachableJoint system, which starts attached and exposes attach/detach
topics (std Empty messages). Detaching lets it fall under physics. Gazebo
re-attaches a child wherever it currently is, even metres away, so pickup is
only requested when the payload is within ``pickup_radius_m`` of the hook
point (the attachment offset below the body), using poses from the world
pose stream. The flight controller is not told about mass changes.

With ``payload.tether.enabled`` the payload hangs from a hook at the end of a
winch cable (see ``add_tether`` in vehicle.py); the hook's position comes
from the world pose stream (link poses are relative to the drone model), and
``Winch`` drives the cable length from held keys or a time schedule.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

DETACH_TOPIC = "/dronelab/payload/detach"
ATTACH_TOPIC = "/dronelab/payload/attach"


class PayloadControl:
    """Track attachment and publish drop/pickup requests."""

    def __init__(self, node: Any, settings: Mapping[str, Any]) -> None:
        from gz.msgs10.empty_pb2 import Empty

        self._empty = Empty
        self._detach = node.advertise(DETACH_TOPIC, Empty)
        self._attach = node.advertise(ATTACH_TOPIC, Empty)
        self._offset = np.asarray(settings["offset_m"], float)
        self._tethered = settings["tether"]["enabled"]
        self._half_height = settings["size_m"][2] / 2
        self._hook_relative: np.ndarray | None = None  # tethered: from pose stream
        self._radius = settings["pickup_radius_m"]
        drop_s = settings["drop_s"]
        self._drop_ns = None if drop_s is None else round(drop_s * 1e9)
        self.attached = True
        self.position: np.ndarray | None = None  # latest payload position (world)
        self.events: list[dict[str, Any]] = []

    def on_pose(self, name: str, position: ArrayLike) -> None:
        """Payload model pose (world) and, when tethered, hook link pose (drone frame)."""
        if name == "payload":
            self.position = np.asarray(position, float)
        elif name == "hook" and self._tethered:
            self._hook_relative = np.asarray(position, float)

    def hook_position(
        self, drone_position: ArrayLike, drone_quaternion: ArrayLike
    ) -> np.ndarray:
        """World position of the hook point."""
        relative = (
            self._hook_relative
            if self._tethered and self._hook_relative is not None
            else self._offset
        )
        return np.asarray(drone_position, float) + Rotation.from_quat(
            drone_quaternion
        ).apply(relative)

    def update(self, sim_time_ns: int) -> dict[str, Any] | None:
        """Apply a scheduled drop when its time arrives."""
        if self._drop_ns is not None and self.attached and sim_time_ns >= self._drop_ns:
            self._drop_ns = None
            return self.drop(sim_time_ns)
        return None

    def drop(self, sim_time_ns: int) -> dict[str, Any] | None:
        if not self.attached:
            return None
        self._detach.publish(self._empty())
        self.attached = False
        return self._event(sim_time_ns, "drop", True)

    def pickup(
        self, sim_time_ns: int, drone_position: ArrayLike, drone_quaternion: ArrayLike
    ) -> dict[str, Any]:
        """Attach if the payload is within reach of the hook point."""
        if self.attached:
            return self._event(sim_time_ns, "pickup", False, reason="already attached")
        hook = self.hook_position(drone_position, drone_quaternion)
        distance = (
            None
            if self.position is None
            else float(np.linalg.norm(self.position - hook))
        )
        # A tethered hook reaches the payload's top, half a height above its centre.
        reach = self._radius + (self._half_height if self._tethered else 0.0)
        if distance is None or distance > reach:
            return self._event(
                sim_time_ns, "pickup", False, reason="out of reach", distance_m=distance
            )
        self._attach.publish(self._empty())
        self.attached = True
        return self._event(sim_time_ns, "pickup", True, distance_m=distance)

    def _event(
        self, sim_time_ns: int, action: str, done: bool, **details: Any
    ) -> dict[str, Any]:
        event = dict(sim_time_ns=sim_time_ns, action=action, done=done, **details)
        self.events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        return dict(attached_at_end=self.attached, events=self.events)


class Winch:
    """Cable length target from held keys or a schedule, sent to Gazebo.

    The target moves at ``winch_speed_mps`` towards a scheduled length, or in
    the direction of the held key (+1 lower, −1 raise); Gazebo's joint
    controller then tracks it. Joint position = length − initial length.
    """

    def __init__(self, node: Any, tether: Mapping[str, Any]) -> None:
        from gz.msgs10.double_pb2 import Double

        self._double = Double
        self._publisher = node.advertise("/model/drone/joint/winch/0/cmd_pos", Double)
        self._initial = tether["initial_length_m"]
        self._max = tether["max_length_m"]
        self._speed = tether["winch_speed_mps"]
        self._schedule = [
            (round(e["time_s"] * 1e9), e["length_m"]) for e in tether["length_schedule"]
        ]
        self.length_m = self._initial
        self._goal = self._initial
        self._previous_ns: int | None = None

    def update(self, sim_time_ns: int, key_rate: int = 0) -> float:
        """Advance the target to sim_time_ns, publish it, and return the length."""
        dt = (
            0.0
            if self._previous_ns is None
            else (sim_time_ns - self._previous_ns) * 1e-9
        )
        self._previous_ns = sim_time_ns
        for time_ns, length in self._schedule:
            if sim_time_ns >= time_ns:
                self._goal = length
        if key_rate:
            self._goal = self.length_m + key_rate * self._speed * dt
        step = float(
            np.clip(self._goal - self.length_m, -self._speed * dt, self._speed * dt)
        )
        self.length_m = float(np.clip(self.length_m + step, 0.1, self._max))
        message = self._double()
        message.data = self.length_m - self._initial
        self._publisher.publish(message)
        return self.length_m
