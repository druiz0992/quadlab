"""Lockstep stepping of a paused Gazebo world through its control service.

In lockstep the simulator never runs ahead of Python: Gazebo starts paused,
and after Python has processed one odometry period (controller tick, motor
commands, wind updates) it asks Gazebo to advance exactly that many physics
steps. Python's processing time therefore costs no simulation time, and runs
are repeatable.

Measured with gz-sim 8 on this project: the control service returns
immediately and the steps run afterwards, so the odometry message at the end
of each burst is the barrier. On an unloaded machine a motor command
published just before a step request is applied in that step (identical
trajectories with and without an added 5 ms delivery delay). Under heavy
load it can arrive after the steps started: topics and the service use
separate connections, so their order is not guaranteed (see
docs/2_simulation.md, lockstep limitation). Gazebo still paces bursts by the world's
real_time_factor; 0 runs them as fast as possible.

The service reply is unreliable from Python: in a probe of 300 bursts, 22 %
of requests timed out after 1 s (43 % after 10 ms), yet every burst ran. So a
step request uses a short timeout and its reply is ignored; the odometry
barrier is the proof that the burst ran. Retrying on a missing reply would
double-step, so a burst is only re-sent after ``resend_after_s`` without
the barrier (counted in ``resends``).
"""

from collections.abc import Callable
import time
from typing import Any

WORLD_CONTROL_SERVICE = "/world/dronelab/control"


def steps_per_period(period_s: float, step_s: float) -> int:
    """Physics steps in one odometry period; the period must be a multiple."""
    steps = round(period_s / step_s)
    if steps < 1 or abs(steps * step_s - period_s) > 1e-9:
        raise ValueError(
            "Lockstep requires the odometry period to be a multiple of the physics step"
        )
    return steps


class LockstepStepper:
    """Advance a paused Gazebo world in fixed bursts of physics steps.

    ``request`` is ``Node.request``-compatible and ``messages`` provides the
    WorldControl/Boolean types; both are injected so tests can use fakes.
    """

    def __init__(
        self,
        request: Callable[..., tuple[bool, Any]],
        messages: Any,
        *,
        steps: int,
        period_ns: int,
        reply_timeout_ms: int = 10,
        resend_after_s: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._request = request
        self._messages = messages
        self.steps = steps
        self._period_ns = period_ns
        self._reply_timeout_ms = reply_timeout_ms
        self._resend_after_s = resend_after_s
        self._now = monotonic
        self._target_ns: int | None = None
        self._sent_at = 0.0
        self.bursts = 0
        self.lost_replies = 0
        self.resends = 0

    @property
    def target_ns(self) -> int | None:
        """Simulation time the last requested burst ends at (None before start)."""
        return self._target_ns

    def start(self, timeout_s: float = 60.0, poll_s: float = 0.2) -> None:
        """Wait for the world control service, then request the first burst.

        Pause-only requests are idempotent, so this waits for a confirmed reply.
        """
        deadline = self._now() + timeout_s
        while not self._call(multi_step=0, timeout_ms=1000):
            if self._now() > deadline:
                raise RuntimeError(
                    "Gazebo world control service did not become available"
                )
            time.sleep(poll_s)
        self._target_ns = 0
        self._burst()

    def on_odometry(self, sim_time_ns: int) -> bool:
        """Call after fully processing an odometry sample; step if it ends the burst.

        Returns True when a new burst was requested.
        """
        if self._target_ns is None or sim_time_ns < self._target_ns:
            return False
        self._burst()
        return True

    def check_stalled(self) -> bool:
        """Re-send the current burst if its barrier has not arrived in time.

        Call while waiting for messages. Returns True when a burst was re-sent.
        """
        if (
            self._target_ns is None
            or self._now() - self._sent_at < self._resend_after_s
        ):
            return False
        self.resends += 1
        self._send()
        return True

    def _burst(self) -> None:
        self._target_ns += self._period_ns
        self.bursts += 1
        self._send()

    def _send(self) -> None:
        # A missing reply does not mean the burst was lost; see module notes.
        if not self._call(multi_step=self.steps, timeout_ms=self._reply_timeout_ms):
            self.lost_replies += 1
        self._sent_at = self._now()

    def _call(self, *, multi_step: int, timeout_ms: int) -> bool:
        WorldControl, Boolean = self._messages.WorldControl, self._messages.Boolean
        message = WorldControl()
        message.pause = True
        if multi_step:
            message.multi_step = multi_step
        ok, reply = self._request(
            WORLD_CONTROL_SERVICE, message, WorldControl, Boolean, timeout_ms
        )
        return bool(ok and reply.data)

    def summary(self) -> dict[str, int]:
        return dict(
            bursts=self.bursts,
            steps_per_burst=self.steps,
            lost_replies=self.lost_replies,
            resends=self.resends,
        )


class TrackedNode:
    """A transport node that remembers the topics it advertises.

    Gazebo silently drops messages published before its subscriber has
    connected (transport discovery). In lockstep the first steps can run
    within milliseconds of start-up, so commands could be lost; the stepper
    therefore waits until every tracked publisher is connected. All other
    attributes are delegated to the wrapped node.
    """

    def __init__(self, node: Any) -> None:
        self._node = node
        self.publishers: list[tuple[str, Any]] = []

    def advertise(self, topic: str, message_type: Any) -> Any:
        publisher = self._node.advertise(topic, message_type)
        self.publishers.append((topic, publisher))
        return publisher

    def __getattr__(self, name: str) -> Any:
        return getattr(self._node, name)


def wait_for_subscribers(
    publishers: list[tuple[str, Any]],
    *,
    timeout_s: float = 30.0,
    poll_s: float = 0.05,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until every publisher has a connected subscriber.

    Raises RuntimeError naming the unconnected topics after ``timeout_s``
    (wall seconds): a command topic nobody subscribes to is a configuration
    or SDF error, never something to ignore.
    """
    deadline = monotonic() + timeout_s
    while True:
        missing = sorted(t for t, p in publishers if not p.has_connections())
        if not missing:
            return
        if monotonic() > deadline:
            raise RuntimeError(
                f"Gazebo did not subscribe to command topics within {timeout_s:g} s: "
                f"{missing}"
            )
        sleep(poll_s)
