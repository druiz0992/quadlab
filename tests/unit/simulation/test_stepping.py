"""Lockstep stepper logic with a fake Gazebo control service."""

from types import SimpleNamespace as NS

import pytest

from dronelab.simulation.gazebo.stepping import (
    WORLD_CONTROL_SERVICE,
    LockstepStepper,
    steps_per_period,
)

MS = 1_000_000


class FakeMessage:
    def __init__(self):
        self.pause = False
        self.multi_step = 0


class FakeService:
    """Records step requests; replies can be scripted to go missing."""

    def __init__(self, replies=None):
        self.requests = []
        self.replies = list(replies or [])

    def __call__(self, service, message, request_type, reply_type, timeout_ms):
        assert service == WORLD_CONTROL_SERVICE and message.pause
        self.requests.append((message.multi_step, timeout_ms))
        ok = self.replies.pop(0) if self.replies else True
        return ok, NS(data=ok)


def stepper(service, clock=None, **kwargs):
    return LockstepStepper(
        service,
        NS(WorldControl=FakeMessage, Boolean=object),
        steps=10,
        period_ns=20 * MS,
        monotonic=clock or (lambda: 0.0),
        **kwargs,
    )


def test_steps_per_period_requires_whole_steps():
    assert steps_per_period(0.02, 0.002) == 10
    assert steps_per_period(0.004, 0.002) == 2
    with pytest.raises(ValueError):
        steps_per_period(0.005, 0.002)


def test_start_waits_for_service_then_requests_first_burst():
    service = FakeService(replies=[False, False, True])
    s = stepper(service)
    s.start(poll_s=0)
    # Three pause-only probes (multi_step 0), then one burst of 10 steps.
    assert [r[0] for r in service.requests] == [0, 0, 0, 10]
    assert s.target_ns == 20 * MS and s.bursts == 1


def test_odometry_barrier_steps_once_per_period():
    service = FakeService()
    s = stepper(service)
    s.start(poll_s=0)
    assert not s.on_odometry(10 * MS)  # mid-burst sample: wait
    assert s.on_odometry(20 * MS)  # burst complete: next burst
    assert s.target_ns == 40 * MS
    assert [r[0] for r in service.requests].count(10) == 2


def test_lost_reply_is_not_retried():
    service = FakeService(replies=[True, False])  # start probe ok, burst reply lost
    s = stepper(service)
    s.start(poll_s=0)
    assert s.lost_replies == 1 and s.bursts == 1
    assert [r[0] for r in service.requests] == [0, 10]  # no duplicate burst


def test_resend_only_after_barrier_timeout():
    now = [0.0]
    service = FakeService()
    s = stepper(service, clock=lambda: now[0], resend_after_s=2.0)
    s.start(poll_s=0)
    now[0] = 1.9
    assert not s.check_stalled()
    now[0] = 2.1
    assert s.check_stalled() and s.resends == 1
    assert s.target_ns == 20 * MS  # same burst, target unchanged
    assert s.summary()["resends"] == 1
