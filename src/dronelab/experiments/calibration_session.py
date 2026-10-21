"""Gazebo session for calibration procedures: pose the drone, read a sensor.

Starts ``gz sim`` on a prepared world in a private transport partition,
subscribes to one sensor topic and to odometry, and moves the drone with the
UserCommands ``set_pose`` service. Replies from Python are unreliable, so
odometry confirming the attitude is the evidence that a pose took effect.
"""

import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import time
from typing import Any
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from ..simulation.gazebo.environment import gazebo_environment

HOLD_POSITION_M = (0.0, 0.0, 10.0)
SET_POSE_SERVICE = "/world/dronelab/set_pose"


class CalibrationSession:
    """Gazebo process, one sensor subscription, and pose control."""

    def __init__(self, world: Path, log: Path, topic: str, message_type: Any) -> None:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.pose_pb2 import Pose
        from gz.transport13 import Node

        os.environ["GZ_PARTITION"] = "dronelab_calibration_" + uuid.uuid4().hex
        os.environ.setdefault("GZ_IP", "127.0.0.1")
        self._pose_type, self._boolean = Pose, Boolean
        self.node = Node()
        self.messages: Queue = Queue()
        self.attitude: np.ndarray | None = None
        self.node.subscribe(message_type, topic, lambda m: self.messages.put(m))
        self.node.subscribe(Odometry, "/dronelab/odometry", self._on_odometry)
        self._log = log.open("w")
        self.process = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "-v", "2", str(world)],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=gazebo_environment(),
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
        self.messages.get(timeout=timeout_s)  # may predate the pose: discard
        received = []
        while len(received) < count:
            try:
                received.append(self.messages.get(timeout=timeout_s))
            except Empty:
                raise RuntimeError("No sensor data; see gazebo log") from None
        return received

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._log.close()
