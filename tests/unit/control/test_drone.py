import math

import pytest

from dronelab.control.drone import (
    ControlBusy,
    ControlModeError,
    RotorThrustControl,
    SimulationPaused,
)
from dronelab.core.commands import RotorThrusts


def setup_control(**kwargs):
    outputs = []
    wall = [0.0]
    control = RotorThrustControl(
        outputs.append, monotonic=lambda: wall[0], start_worker=False, **kwargs
    )
    control.update_clock(0)
    return control, control.acquire(), outputs, wall


def test_receipt_ownership_units_and_order():
    c, h, outputs, _ = setup_control()
    source = [1, 2, 3, 4]
    command = RotorThrusts(source)
    source[0] = 99
    receipt = h.send(command, lifetime_sim_s=0.05)
    assert receipt.command_id == 1
    assert receipt.expires_sim_time_ns == 50_000_000
    assert receipt.accepted.thrust_n == (1, 2, 3, 4)
    assert [r.id for r in h.rotors] == [f"rotor_{i}" for i in range(4)]
    for speed, thrust in zip(receipt.motor_speed_targets_rad_s, (1, 2, 3, 4)):
        assert 8.55e-6 * speed**2 == pytest.approx(thrust)
    with pytest.raises(ControlBusy):
        c.acquire()
    c.tick()
    assert outputs == [(1, 2, 3, 4)]
    h.close()
    assert outputs[-1] == (0,) * 4
    with pytest.raises(RuntimeError):
        h.send(command, lifetime_sim_s=0.05)
    c.acquire().close()
    c.close()


@pytest.mark.parametrize(
    "values",
    [
        (1, 2, 3),
        (True, 1, 1, 1),
        (math.nan,) * 4,
        (math.inf,) * 4,
        (-1,) * 4,
        (9,) * 4,
        ("1",) * 4,
    ],
)
def test_invalid_does_not_replace_pending(values):
    c, h, outputs, _ = setup_control()
    h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.05)
    with pytest.raises(ValueError):
        h.send(RotorThrusts(values), lifetime_sim_s=0.05)
    c.tick()
    assert outputs[-1] == (1,) * 4
    c.close()


def test_clip_and_latest_only_and_expiry():
    c, h, outputs, _ = setup_control()
    first = h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.05)
    receipt = h.send(RotorThrusts((-1, 20, 0, 2)), lifetime_sim_s=0.05, bounds="clip")
    assert receipt.requested.thrust_n == (-1, 20, 0, 2)
    assert receipt.accepted.thrust_n == pytest.approx((0, 8.55, 0, 2))
    assert receipt.limited_rotor_indices == (0, 1)
    c.tick()
    assert outputs == [receipt.accepted.thrust_n]
    c.update_clock(50_000_000)
    c.tick()
    c.tick()
    assert outputs == [receipt.accepted.thrust_n, (0,) * 4]
    events = c.drain_events()
    assert [(e["status"], e["command_id"]) for e in events] == [
        ("superseded", first.command_id),
        ("published", receipt.command_id),
        ("expired", receipt.command_id),
    ]
    c.close()


def test_heartbeat_does_not_extend_watchdog():
    c, h, outputs, wall = setup_control()
    h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)
    for t in (0.1, 0.5, 0.9):
        wall[0] = t
        c.tick()
    wall[0] = 1.0
    c.tick()
    assert outputs[-1] == (0,) * 4
    assert c.drain_events()[-1]["status"] == "expired"
    c.close()


def test_pause_stop_and_time_reversal():
    c, h, outputs, _ = setup_control()
    h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)
    c.update_clock(10, paused=True)
    with pytest.raises(SimulationPaused):
        h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)
    c.update_clock(10)
    before = len(outputs)
    c.tick()
    assert len(outputs) == before
    h.send(RotorThrusts((2,) * 4), lifetime_sim_s=0.1)
    h.stop()
    c.tick()
    assert outputs[-1] == (0,) * 4
    with pytest.raises(RuntimeError, match="reversed"):
        c.update_clock(0)
    with pytest.raises(RuntimeError, match="faulted"):
        c.check_health()
    c.close()


def test_failure_faults_and_close_still_releases_worker():
    c, h, _, _ = setup_control()
    h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)

    def fail(values):
        raise OSError("transport failed")

    c._publish = fail
    with pytest.raises(OSError):
        c.tick()
    with pytest.raises(RuntimeError, match="faulted"):
        h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)
    assert c.drain_events()[-1]["status"] == "failed"
    with pytest.raises(OSError):
        c.close()
    c.close()


@pytest.mark.parametrize("lifetime", [0, -1, math.nan, math.inf, True, 0.2, 1e-12])
def test_invalid_lifetime(lifetime):
    c, h, _, _ = setup_control()
    with pytest.raises(ValueError):
        h.send(RotorThrusts((1,) * 4), lifetime_sim_s=lifetime)
    c.close()


def test_wrong_mode_and_clock_not_ready():
    c = RotorThrustControl(lambda _: None, control_mode="velocity", start_worker=False)
    with pytest.raises(ControlModeError):
        c.acquire()
    c.close()
    c = RotorThrustControl(lambda _: None, start_worker=False)
    with c.acquire() as h:
        with pytest.raises(RuntimeError, match="clock"):
            h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)
    c.close()


def test_stop_serializes_with_worker_publication():
    import threading

    entered, release = threading.Event(), threading.Event()
    outputs = []

    def publish(values):
        if any(values):
            entered.set()
            assert release.wait(2)
        outputs.append(values)

    c = RotorThrustControl(publish, publish_rate_hz=200)
    c.update_clock(0)
    h = c.acquire()
    try:
        h.send(RotorThrusts((1,) * 4), lifetime_sim_s=0.1)
        assert entered.wait(2)
        stopped = threading.Event()

        def stop():
            h.stop()
            stopped.set()

        thread = threading.Thread(target=stop)
        thread.start()
        release.set()
        assert stopped.wait(2)
        thread.join()
        c.tick()
        assert outputs[0] == (1,) * 4
        assert outputs[-1] == (0,) * 4
    finally:
        release.set()
        c.close()
    assert not c._worker.is_alive()


def test_publish_on_send_publishes_immediately_and_worker_still_expires():
    c, h, outputs, _ = setup_control(publish_on_send=True)
    h.send(RotorThrusts((1, 2, 3, 4)), lifetime_sim_s=0.05)
    assert outputs == [(1, 2, 3, 4)]
    assert [e["status"] for e in c.drain_events()] == ["published"]
    c.update_clock(60_000_000)
    c.tick()
    assert outputs[-1] == (0.0, 0.0, 0.0, 0.0)
    h.close()


def test_wall_watchdog_can_be_disabled_for_lockstep():
    c, h, outputs, wall = setup_control(watchdog_wall_s=None, publish_on_send=False)
    h.send(RotorThrusts((1, 2, 3, 4)), lifetime_sim_s=0.05)
    wall[0] = 1000.0  # a long wall-clock stall does not expire the command
    c.tick()
    assert outputs == [(1, 2, 3, 4)]
    h.close()
