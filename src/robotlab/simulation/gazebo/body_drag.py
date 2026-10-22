"""Apply a body force and torque to the drone through ApplyLinkWrench.

Measured with gz-sim 8: wrenches on ``/world/robotlab/wrench/persistent`` are
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

PERSISTENT_WRENCH_TOPIC = "/world/robotlab/wrench/persistent"


class BodyForce:
    """Keep a persistent world-axis wrench on one link (default
    ``drone::base_link``; wheeled robots: ``<model>::base_link``)."""

    def __init__(self, node: Any, link: str = "drone::base_link") -> None:
        from gz.msgs10.entity_pb2 import Entity
        from gz.msgs10.entity_wrench_pb2 import EntityWrench

        self._message_type, self._link = EntityWrench, Entity.LINK
        self._publisher = node.advertise(PERSISTENT_WRENCH_TOPIC, EntityWrench)
        self._applied = np.zeros(6)
        self._entity_name = link
        self.messages = 0

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
        message.entity.name = self._entity_name
        message.entity.type = self._link
        w = message.wrench
        w.force.x, w.force.y, w.force.z = map(float, delta[:3])
        w.torque.x, w.torque.y, w.torque.z = map(float, delta[3:])
        if not self._publisher.publish(message):
            raise RuntimeError("Unable to publish body wrench")
        self._applied = wrench
        self.messages += 1

    def clear(self) -> None:
        self.set(np.zeros(3))


INSTANT_WRENCH_TOPIC = "/world/robotlab/wrench"


class BurstImpulseWrench:
    """Apply a wrench held over one lockstep burst as a single-step impulse.

    Why not ``BodyForce``: persistent wrenches accumulate as separate
    entries that ApplyLinkWrench resolves every physics step, so publishing
    a delta every period makes each step slower as the run goes on
    (measured with gz-sim 8, 500 Hz wheeled-robot updates: a 6 s drive took
    30 s instead of 13 s, growing quadratically), and clearing then
    re-adding is not applied in a reliable order (the wrench was missing
    for whole steps). Instead, each period publishes one instantaneous
    wrench (``/world/robotlab/wrench``; Gazebo applies it on the next
    physics step only) scaled by the burst's step count: the first step
    receives ``steps`` times the force and the other steps none, which
    gives the same impulse as the force held for the burst and differs in
    position only at second order in the step (a 6 s, 48 m cruise agreed
    with the persistent wrench to 0.2 mm). Lockstep only: the plugin barrier
    guarantees the message is read before the burst's first step.
    """

    def __init__(self, node: Any, link: str, steps_per_burst: int) -> None:
        from gz.msgs10.entity_pb2 import Entity
        from gz.msgs10.entity_wrench_pb2 import EntityWrench

        if steps_per_burst < 1:
            raise ValueError("BurstImpulseWrench needs at least one step per burst")
        self._message_type, self._link = EntityWrench, Entity.LINK
        self._publisher = node.advertise(INSTANT_WRENCH_TOPIC, EntityWrench)
        self._entity_name = link
        self._steps = steps_per_burst
        self.messages = 0

    def set(
        self, force_world_n: ArrayLike, torque_world_nm: ArrayLike | None = None
    ) -> None:
        """Hold force (N) and torque (N·m, about the link origin), world axes,
        for the coming burst."""
        force = self._steps * np.asarray(force_world_n, dtype=float)
        torque = self._steps * (
            np.zeros(3)
            if torque_world_nm is None
            else np.asarray(torque_world_nm, dtype=float)
        )
        message = self._message_type()
        message.entity.name = self._entity_name
        message.entity.type = self._link
        w = message.wrench
        w.force.x, w.force.y, w.force.z = map(float, force)
        w.torque.x, w.torque.y, w.torque.z = map(float, torque)
        if not self._publisher.publish(message):
            raise RuntimeError("Unable to publish wheel wrench")
        self.messages += 1

    def clear(self) -> None:
        """Nothing persists between bursts."""
