"""Lockstep stepper logic with a fake Gazebo control service."""

from types import SimpleNamespace as NS

import pytest

from robotlab.simulation.gazebo.stepping import (
    WORLD_CONTROL_SERVICE,
    LockstepStepper,
    steps_per_period,
    RELEASE_NS,
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


# Plugin barrier ------------------------------------------------------------------


class Recorder:
    """Records grants and step requests in one timeline (their order matters)."""

    def __init__(self):
        self.events = []
        self.service = FakeService()

    def grant(self, ns):
        self.events.append(("grant", ns))

    def request(self, *args):
        result = self.service(*args)
        if args[1].multi_step:
            self.events.append(("step", args[1].multi_step))
        return result


def test_plugin_grant_precedes_each_step_request():
    r = Recorder()
    s = stepper(r.request, grant=r.grant)
    s.start(poll_s=0)
    s.on_odometry(20 * MS)
    assert r.events == [
        ("grant", 20 * MS),
        ("step", 10),
        ("grant", 40 * MS),
        ("step", 10),
    ]
    assert s.summary()["barrier"] == "plugin"
    assert stepper(FakeService()).summary()["barrier"] == "service"


def test_plugin_regrants_but_never_resends_steps_then_fails():
    now = [0.0]
    r = Recorder()
    s = stepper(r.request, clock=lambda: now[0], grant=r.grant, regrant_after_s=0.25)
    s.start(poll_s=0)
    now[0] = 0.2
    assert not s.check_stalled()
    for t in (0.3, 0.6, 2.5, 5.0):
        now[0] = t
        assert s.check_stalled()
    assert s.regrants == 4 and s.resends == 0
    assert r.events.count(("step", 10)) == 1  # one step request only
    assert all(e == ("grant", 20 * MS) for e in r.events if e[0] == "grant")
    now[0] = 10.5
    with pytest.raises(RuntimeError, match="did not finish"):
        s.check_stalled()


def test_release_lifts_the_barrier_only_in_plugin_mode():
    r = Recorder()
    s = stepper(r.request, grant=r.grant)
    s.release()
    assert r.events == [("grant", RELEASE_NS)]
    stepper(FakeService()).release()  # service mode: nothing to do


# Barrier selection, world, environment ----------------------------------------------

from robotlab.simulation.gazebo import plugins
from robotlab.simulation.gazebo.environment import gazebo_environment


def lockstep_settings(barrier, lockstep=True):
    return {"simulation": {"lockstep": lockstep, "lockstep_barrier": barrier}}


def test_barrier_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOTLAB_PLUGIN_DIR", str(tmp_path))
    assert plugins.lockstep_barrier(lockstep_settings("auto", lockstep=False)) is None
    assert plugins.lockstep_barrier(lockstep_settings("auto")) == "service"
    with pytest.raises(ValueError, match="robotlab-build-plugins"):
        plugins.lockstep_barrier(lockstep_settings("plugin"))
    (tmp_path / plugins.LOCKSTEP_LIBRARY).write_bytes(b"")
    assert plugins.lockstep_barrier(lockstep_settings("auto")) == "plugin"
    assert plugins.lockstep_barrier(lockstep_settings("service")) == "service"
    with pytest.raises(ValueError, match="lockstep_barrier"):
        plugins.lockstep_barrier(lockstep_settings("sometimes"))


def test_environment_adds_plugin_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOTLAB_PLUGIN_DIR", str(tmp_path))
    env = gazebo_environment({"GZ_SIM_SYSTEM_PLUGIN_PATH": "/other"})
    assert env["GZ_SIM_SYSTEM_PLUGIN_PATH"] == f"/other:{tmp_path}"


def test_world_lists_plugin_first_only_with_plugin_barrier(monkeypatch, tmp_path):
    import xml.etree.ElementTree as ET

    from robotlab.simulation.gazebo.config import load_drone_config
    from robotlab.simulation.gazebo.world import build_world, load_scene

    monkeypatch.setenv("ROBOTLAB_PLUGIN_DIR", str(tmp_path))
    (tmp_path / plugins.LOCKSTEP_LIBRARY).write_bytes(b"")
    layer = tmp_path / "lockstep.json"
    layer.write_text('{"simulation": {"lockstep": true}}')
    scene = load_scene("worlds/training.json")

    def first_plugin(config):
        root = ET.parse(build_world(scene, tmp_path / "w.sdf", config=config)).getroot()
        return root.find("world").find("plugin").get("filename")

    assert first_plugin(load_drone_config(layer)) == plugins.LOCKSTEP_FILENAME
    assert first_plugin(load_drone_config()) != plugins.LOCKSTEP_FILENAME
