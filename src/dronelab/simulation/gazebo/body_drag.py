"""Apply a body force and torque to the drone through ApplyLinkWrench.

Measured with gz-sim 8: wrenches on ``/world/dronelab/wrench/persistent`` are
in world axes, act every step at the link origin (not the centre of mass;
checked with an offset inertial pose), and **accumulate** (two 1 N messages
give 2 N).
So each update publishes only the change since the last one, keeping the
running total equal to the requested force on a single, ordered topic; a
+1 N then −1 N pair was verified to cancel exactly.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

PERSISTENT_WRENCH_TOPIC = "/world/dronelab/wrench/persistent"


class BodyForce:
    """Keep a persistent world-axis force on ``drone::base_link``."""

    def __init__(self, node: Any) -> None:
        from gz.msgs10.entity_pb2 import Entity
        from gz.msgs10.entity_wrench_pb2 import EntityWrench

        self._message_type, self._link = EntityWrench, Entity.LINK
        self._publisher = node.advertise(PERSISTENT_WRENCH_TOPIC, EntityWrench)
        self._applied = np.zeros(6)

    def set(
        self, force_world_n: ArrayLike, torque_world_nm: ArrayLike | None = None
    ) -> None:
        """Make the total applied wrench equal the given force (N) and torque
        (N·m, about the link origin), both in world axes."""
        torque = np.zeros(3) if torque_world_nm is None else torque_world_nm
        wrench = np.concatenate(
            (np.asarray(force_world_n, dtype=float), np.asarray(torque, dtype=float))
        )
        delta = wrench - self._applied
        if not np.any(delta):
            return
        message = self._message_type()
        message.entity.name = "drone::base_link"
        message.entity.type = self._link
        w = message.wrench
        w.force.x, w.force.y, w.force.z = map(float, delta[:3])
        w.torque.x, w.torque.y, w.torque.z = map(float, delta[3:])
        if not self._publisher.publish(message):
            raise RuntimeError("Unable to publish body wrench")
        self._applied = wrench

    def clear(self) -> None:
        self.set(np.zeros(3))
