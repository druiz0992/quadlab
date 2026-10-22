"""Gazebo session for calibration procedures: pose the drone, read a sensor.

Starts ``gz sim`` on a prepared world in a private transport partition,
subscribes to one sensor topic and to odometry, and moves the drone with the
UserCommands ``set_pose`` service. Replies from Python are unreliable, so
odometry confirming the attitude is the evidence that a pose took effect.
Every wait fails at once, naming the log, if Gazebo has exited; ``close``
stops Gazebo completely (see ``simulation/gazebo/process.py``) and cannot be
interrupted half-way by a second Ctrl-C or SIGTERM.
"""

import os
from pathlib import Path
from queue import Empty, Queue
import time
from typing import Any
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from ..simulation.gazebo.environment import gazebo_environment
from ..simulation.gazebo.process import GazeboProcess
from .interrupts import shielded

HOLD_POSITION_M = (0.0, 0.0, 10.0)
SET_POSE_SERVICE = "/world/robotlab/set_pose"


class CalibrationSession:
    """Gazebo process, one sensor subscription, and pose control."""

    def __init__(self, world: Path, log: Path, topic: str, message_type: Any) -> None:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.pose_pb2 import Pose
        from gz.transport13 import Node

        os.environ["GZ_PARTITION"] = "robotlab_calibration_" + uuid.uuid4().hex
        os.environ.setdefault("GZ_IP", "127.0.0.1")
        self._pose_type, self._boolean = Pose, Boolean
        self.node = Node()
        self.messages: Queue = Queue()
        self.attitude: np.ndarray | None = None
        self.node.subscribe(message_type, topic, lambda m: self.messages.put(m))
        self.node.subscribe(Odometry, "/robotlab/odometry", self._on_odometry)
        self._log_path = log
        self._log = log.open("w")
        self.process = GazeboProcess(
            ["gz", "sim", "-s", "-r", "-v", "2", str(world)],
            self._log,
            gazebo_environment(),
        )

    def _check_alive(self) -> None:
        """Raise if Gazebo has exited (instead of waiting out a timeout)."""
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Gazebo exited ({self.process.returncode}); see {self._log_path}"
            )

    def _on_odometry(self, message: Any) -> None:
        q = message.pose.orientation
        self.attitude = np.array([q.x, q.y, q.z, q.w])

    def set_attitude(self, rotation: Rotation, timeout_s: float = 10.0) -> None:
        """Move the drone to ``rotation`` and wait until odometry reports it."""
        request = self._pose_type()
        request.name = "drone"
        request.position.x, request.position.y, request.position.z = HOLD_POSITION_M
        x, y, z, w = rotation.as_quat()
        request.orientation.x, request.orientation.y = x, y
        request.orientation.z, request.orientation.w = z, w
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._check_alive()
            self.node.request(
                SET_POSE_SERVICE, request, self._pose_type, self._boolean, 100
            )
            time.sleep(0.05)
            if (
                self.attitude is not None
                and abs(np.dot(self.attitude, rotation.as_quat())) > 0.99999
            ):
                return
        raise RuntimeError("Gazebo did not apply the calibration pose; see gazebo log")

    def fresh(self, count: int, timeout_s: float = 10.0) -> list:
        """``count`` sensor messages published after this call."""
        while not self.messages.empty():
            self.messages.get_nowait()
        self._next(timeout_s)  # may predate the pose: discard
        return [self._next(timeout_s) for _ in range(count)]

    def _next(self, timeout_s: float) -> Any:
        """Next sensor message; checks Gazebo is alive every 0.5 s."""
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_alive()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"No sensor data; see {self._log_path}")
            try:
                return self.messages.get(timeout=min(0.5, remaining))
            except Empty:
                pass

    def close(self) -> None:
        """Stop Gazebo (SIGTERM, then kill after 10 s) and close the log."""
        with shielded():
            self.process.stop(timeout_s=10)
            self._log.close()
