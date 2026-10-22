"""Mission stages split out of experiments/gazebo.py (no Gazebo needed).

End-to-end behaviour of the stages together is covered by the opt-in
Gazebo suites; these pin each stage's own contract with fakes.
"""

from dataclasses import replace
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robotlab.control.manual import FlightCommand
from robotlab.experiments.algorithms import StaleEstimate
from robotlab.experiments.gazebo import GazeboConfig
from robotlab.experiments.keyboard_flight import apply_keyboard_effects
from robotlab.experiments.rotor_mission import RotorMission, control_state
from robotlab.experiments.run_setup import validate_run
from robotlab.experiments.run_summary import final_exit_reason
from robotlab.experiments.vehicle_forces import ExternalForces
from robotlab.simulation.aerodynamics import quadratic_body_drag_world
from robotlab.simulation.gazebo.config import load_drone_config

ROTOR = GazeboConfig(control_mode="rotor-thrust", headless=True)


def settings(**sections):
    data = load_drone_config().data
    for name, values in sections.items():
        data[name] = dict(data[name], **values)
    return data


# run_setup ------------------------------------------------------------------


def test_validate_run_reports_mode():
    mode = validate_run(
        ROTOR,
        settings(simulation={"lockstep": True}, control={"publish_on_send": True}),
        False,
    )
    assert mode.rotor_mode and mode.lockstep
    velocity = validate_run(GazeboConfig(), settings(), False)
    assert not velocity.rotor_mode and not velocity.lockstep


@pytest.mark.parametrize(
    "args, sections, has_source, message",
    [
        (replace(ROTOR, control_mode="hover"), {}, False, "Unknown control mode"),
        (
            replace(ROTOR, keyboard=True),
            {"simulation": {"lockstep": True}},
            False,
            "needs real time",
        ),
        (
            ROTOR,
            {"simulation": {"lockstep": True}, "control": {"publish_on_send": False}},
            False,
            "publish_on_send",
        ),
        (
            replace(ROTOR, ekf=True),
            {"algorithms": {"estimator": "gps_imu_ekf"}},
            False,
            "not both",
        ),
        (GazeboConfig(), {"mission": {"type": "hold"}}, False, "mission reference"),
        (ROTOR, {"mission": {"type": "hold"}}, True, "mission reference"),
        (replace(ROTOR, world="x.sdf"), {}, False, "generated scene"),
        (replace(ROTOR, duration=0), {}, False, "Duration"),
        (replace(ROTOR, target=(0, float("nan"), 1)), {}, False, "target"),
    ],
)
def test_validate_run_rejects_conflicts(args, sections, has_source, message):
    with pytest.raises(ValueError, match=message):
        validate_run(args, settings(**sections), has_source)


# vehicle_forces -------------------------------------------------------------


class FakeBodyForce:
    def __init__(self):
        self.calls = []

    def set(self, force, torque=None):
        self.calls.append((np.asarray(force), np.asarray(torque)))


class FakeRecorder:
    def __init__(self):
        self.vectors = []

    def log_vector(self, name, values, labels=None):
        self.vectors.append(name)


def test_external_forces_apply_body_drag():
    data = settings(
        aerodynamics={"body_drag": {"enabled": True, "drag_area_m2": [0.1] * 3}}
    )
    assert ExternalForces.needed(data)
    body, recorder = FakeBodyForce(), FakeRecorder()
    forces = ExternalForces(data, body, recorder)
    level = np.array([0.0, 0.0, 0.0, 1.0])
    wind = np.array([1.0, 0.0, 0.0])
    forces.update(0, np.array([4.0, 0.0, 0.0]), lambda: wind, level, np.zeros(3))
    expected = quadratic_body_drag_world(
        [4.0, 0.0, 0.0],
        wind,
        level,
        [0.1] * 3,
        data["aerodynamics"]["air_density_kg_m3"],
    )
    ((force, torque),) = body.calls
    np.testing.assert_allclose(force, expected)
    assert force[0] < 0 and not np.any(torque)
    assert forces.summary() == dict(max_force_n=pytest.approx(np.linalg.norm(expected)))
    assert recorder.vectors == ["aero/body_drag_n"]


def test_external_forces_disabled_publish_nothing():
    data = settings()
    assert not ExternalForces.needed(data)
    forces = ExternalForces(data, None, FakeRecorder())
    forces.update(
        0, np.ones(3), lambda: np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3)
    )
    assert forces.summary() is None


# rotor_mission ----------------------------------------------------------------


def test_control_state_truth_estimate_and_stale():
    truth = object()
    assert control_state(None, None, 0, 5.0, 0.1, truth) is truth
    source = SimpleNamespace(state=lambda now, latest: None)
    runner = SimpleNamespace(latest_estimate=None)
    assert control_state(source, runner, 0, 0.05, 0.1, truth) is None  # start-up
    with pytest.raises(StaleEstimate):
        control_state(source, runner, 0, 0.2, 0.1, truth)


def rotor_with(**parts):
    base = dict(lockstep=True, settings={"control": {"odometry_rate_hz": 50.0}})
    return RotorMission(SimpleNamespace(**dict(base, **parts)))


def test_rotor_mission_rejects_irregular_odometry():
    rotor = rotor_with()
    rotor._check_interval(1.00, received_at=0.0)
    rotor._check_interval(1.02, received_at=0.0)
    assert rotor.previous_control_time == 1.02
    with pytest.raises(RuntimeError, match="interval"):
        rotor._check_interval(1.02, received_at=0.0)  # repeated time
    gap = rotor_with()
    gap._check_interval(1.0, received_at=0.0)
    with pytest.raises(RuntimeError, match="interval"):
        gap._check_interval(1.2, received_at=0.0)  # longer than 0.1 s
    realtime = rotor_with(lockstep=False)
    with pytest.raises(RuntimeError, match="Stale odometry"):
        realtime._check_interval(1.0, received_at=-10.0)


class FakeTakeoff:
    complete = False

    def command(self, altitude, elapsed, command):
        return ("takeoff", altitude, elapsed, command)


def test_pilot_priority_failsafe_mission_takeoff_keyboard():
    held = FlightCommand(forward_mps=1.0)
    state = SimpleNamespace(position_m=np.array([0.0, 0.0, 2.0]))
    mission = SimpleNamespace(enabled=False, command=lambda t, p: ("mission", t))
    parts = dict(
        controller=SimpleNamespace(setpoint_m=None),
        mission=mission,
        takeoff=None,
        command_source=None,
    )
    pilot = rotor_with(**parts)._pilot(state, 10.0, held, None)
    assert pilot(12.0) is held
    source = rotor_with(**dict(parts, command_source=lambda t: ("source", t)))
    assert source._pilot(state, 10.0, held, None)(12.0) == ("source", 2.0)
    takeoff = rotor_with(**dict(parts, takeoff=FakeTakeoff()))
    assert takeoff._pilot(state, 10.0, held, None)(12.0) == ("takeoff", 2.0, 2.0, held)
    flying = rotor_with(**dict(parts, mission=replace_mission(mission)))
    assert flying._pilot(state, 10.0, held, None)(12.0) == ("mission", 2.0)
    failsafe = FlightCommand(climb_mps=-0.5)
    assert flying._pilot(state, 10.0, held, failsafe)(12.0) is failsafe


def replace_mission(mission):
    return SimpleNamespace(enabled=True, command=mission.command)


# keyboard_flight and run_summary ------------------------------------------------


class FakeKeyboard:
    def drain_actions(self):
        return ["spawn"]

    def drain_toggles(self):
        return [("wind", False), ("gps", True)]


class FakeEffects:
    def perform(self, name, stamp):
        return dict(action=name, sim_time_ns=stamp)

    def set(self, name, on, stamp):
        return None if name == "gps" else dict(effect=name, on=on, sim_time_ns=stamp)


def test_keyboard_effects_logged_in_order_with_sim_time():
    log, events = io.StringIO(), []
    recorder = SimpleNamespace(log_effect=events.append)
    apply_keyboard_effects(FakeKeyboard(), FakeEffects(), 7, log, recorder)
    lines = [json.loads(line) for line in log.getvalue().splitlines()]
    assert (
        lines
        == events
        == [
            dict(action="spawn", sim_time_ns=7),
            dict(effect="wind", on=False, sim_time_ns=7),
        ]
    )


def test_final_exit_reason():
    assert final_exit_reason(True, "duration") == "duration"
    assert final_exit_reason(False, "duration") == "error"
    assert final_exit_reason(False, "interrupted") == "interrupted"
    assert final_exit_reason(False, "keyboard_closed") == "keyboard_closed"


# mission_loop -----------------------------------------------------------------


def test_keyboard_polled_at_most_every_period(monkeypatch):
    from robotlab.experiments import mission_loop
    from robotlab.experiments.mission_loop import (
        KEYBOARD_POLL_PERIOD_WALL_S,
        MissionLoop,
    )

    class Keyboard:
        closed = False
        polls = 0

        def poll(self):
            self.polls += 1
            return FlightCommand(forward_mps=self.polls)

    keyboard = Keyboard()
    loop = MissionLoop.__new__(MissionLoop)
    loop.parts = SimpleNamespace(keyboard=keyboard, effects=None)
    loop._next_keyboard_poll_wall_s = 0.0
    now = [10.0]
    monkeypatch.setattr(mission_loop.time, "monotonic", lambda: now[0])
    assert loop._poll_keyboard() and keyboard.polls == 1
    now[0] += KEYBOARD_POLL_PERIOD_WALL_S / 2
    assert loop._poll_keyboard() and keyboard.polls == 1  # command held
    assert loop._pilot_command.forward_mps == 1
    now[0] += KEYBOARD_POLL_PERIOD_WALL_S
    keyboard.closed = True
    assert not loop._poll_keyboard() and keyboard.polls == 2


# calibration_session -----------------------------------------------------------


def test_calibration_session_fails_fast_when_gazebo_exits(tmp_path):
    from queue import Queue

    from robotlab.experiments.calibration_session import CalibrationSession

    session = CalibrationSession.__new__(CalibrationSession)
    session.messages = Queue()
    session._log_path = tmp_path / "gazebo.log"
    session.process = SimpleNamespace(poll=lambda: 1, returncode=1)
    with pytest.raises(RuntimeError, match="Gazebo exited"):
        session.fresh(3, timeout_s=3600)


# run_evaluation ------------------------------------------------------------------


def test_controller_tracking_reference_replaces_the_mission():
    from robotlab.control.references import ReferencePoint
    from robotlab.experiments.mission import MissionRunner
    from robotlab.experiments.run_evaluation import RunEvaluation

    data = settings()
    calls = []

    def reference(position, velocity, quaternion):
        calls.append(position.copy())
        return ReferencePoint(position + [0, 0, 1.0], velocity, np.zeros(3), 0.0)

    q = np.array([0, 0, 0, 1.0])
    for source, expected in ((reference, "controller"), (None, "mission")):
        evaluation = RunEvaluation(
            data,
            15.0,
            MissionRunner(data["mission"]),
            lambda *a, **k: None,
            tracking_reference=source,
        )
        errors = evaluation.on_truth(
            0, np.array([1.0, 2, 3]), np.zeros(3), q, controlled=True
        )
        summary = evaluation.summary()["control"]
        assert summary["tracking_reference"] == expected
        if source is not None:
            assert errors["position_m"] == pytest.approx(1.0)
        else:
            assert errors is None  # mission not started yet
    assert len(calls) == 1
