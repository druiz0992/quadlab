"""Opt-in Gazebo checks for wheeled robots (rover, car, tractor).

Run with ``ROBOTLAB_TEST_GAZEBO=1`` (about 1.5 minutes). Wheeled robots
always run in lockstep; ``real_time_factor`` 0 runs them as fast as
possible. Expected values were verified by hand on 2026-10-22 (see
docs/8_ground_vehicles.md); tolerances leave room for floating-point
differences across machines, not for behaviour changes.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from robotlab.control.wheeled import WheeledControlState
from robotlab.control.wheeled_template import TemplateWheeledController
from robotlab.experiments.gazebo import GazeboConfig, run
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.gazebo.world import load_scene
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.simulation.wheeled.model import WheeledVehicleModel
from robotlab.simulation.wheeled.rigid_body import RigidBody
from robotlab.simulation.wheeled.spawn import spawn_pose
from robotlab.testing.plugins import synthetic_context

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo transport and physics (ROBOTLAB_TEST_GAZEBO=1)",
)

FAST = {"simulation": {"real_time_factor": 0}}
CRUISE = {
    **FAST,
    "algorithms": {
        "controller": "template",
        "controller_options": {"cruise_speed_mps": 10.0},
    },
}


def drive(directory: Path, robot: str, layer: dict, duration_s: float, scene: str):
    """One headless wheeled run; return (summary, telemetry rows)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "layer.json"
    path.write_text(json.dumps(layer))
    output = directory / "run"
    run(
        GazeboConfig(
            robot=robot,
            scene=scene,
            drone=(str(path),),
            output=str(output),
            headless=True,
            duration=duration_s,
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "telemetry.jsonl").open()]
    return summary, rows


def offline_cruise(layer_path: Path, periods: int) -> np.ndarray:
    """The same car, controller, and command timing on the Python reference
    integrator (the configured odometry period and physics steps)."""
    config = load_drone_config(layer_path, robot="car")
    vehicle, settings = config.wheeled, config.data
    period_s = 1 / settings["control"]["odometry_rate_hz"]
    step_s = settings["simulation"]["step_s"]
    control_every = round(
        settings["control"]["odometry_rate_hz"] / settings["control"]["control_rate_hz"]
    )
    ground = GroundModel.from_scene(load_scene("worlds/airfield.json"), None)
    pose = spawn_pose(vehicle, ground, 0.0, 0.0, 0.0)
    body = RigidBody.at_rest(vehicle.mass_kg, vehicle.inertia_kg_m2, pose.position_m)
    model = WheeledVehicleModel(vehicle, ground)
    controller = TemplateWheeledController(
        synthetic_context(wheeled=vehicle), {"cruise_speed_mps": 10.0}
    )
    for period in range(periods):
        state = body.state()
        if period % control_every == 0:
            control = WheeledControlState(
                round(period * period_s * 1e9),
                state.position_m,
                state.velocity_world_mps,
                state.quaternion_xyzw,
                state.body_rates_rad_s,
                model.wheel_speed_rad_s.copy(),
                model.actuators.steer_rad.copy(),
            )
            dt = control_every * period_s if period else 0.0
            model.actuators.accept(controller.update(control, None, dt), 0)
        wrench = model.step(state, period_s)
        for _ in range(round(period_s / step_s)):
            body.step(wrench, step_s)
    return body.position_m


def test_car_cruise_matches_python_reference(tmp_path):
    summary, rows = drive(tmp_path, "car", CRUISE, 6.0, "worlds/airfield.json")
    assert summary["health"]["verdict"] == "ok"
    assert summary["lockstep"]["steps_per_burst"] == 2
    final = np.array(rows[-1]["position"])
    # Verified 2026-10-22: 48.59 m at 6.004 s (template controller, 10 m/s;
    # 48.531 m at 6.002 s with the earlier 500 Hz / 1 ms preset).
    assert final[0] == pytest.approx(48.589, abs=0.05)
    assert abs(final[1]) < 1e-6 and final[2] == pytest.approx(0.358, abs=0.002)
    # Gazebo's first odometry is one period after the reference's start.
    reference = offline_cruise(tmp_path / "layer.json", len(rows))
    assert np.linalg.norm(final - reference) < 0.1


@pytest.mark.parametrize(
    "robot, height_m", [("rover", 0.219), ("car", 0.358), ("tractor", 1.023)]
)
def test_parked_vehicles_settle_at_ride_height(tmp_path, robot, height_m):
    summary, rows = drive(tmp_path, robot, FAST, 2.0, "worlds/training.json")
    assert summary["health"]["verdict"] == "ok"
    assert rows[-1]["position"][2] == pytest.approx(height_m, abs=0.003)
    assert np.linalg.norm(np.array(rows[-1]["position"][:2])) < 1e-3
    assert summary["dropped"] == {} or not any(summary["dropped"].values())


def test_wheeled_lockstep_is_deterministic(tmp_path):
    layer = {
        **FAST,
        "algorithms": {
            "controller": "template",
            "controller_options": {"cruise_speed_mps": 0.8},
        },
    }
    first = drive(tmp_path / "a", "rover", layer, 3.0, "worlds/training.json")[1]
    second = drive(tmp_path / "b", "rover", layer, 3.0, "worlds/training.json")[1]
    assert first == second


def race(mode: str) -> dict:
    return {
        **FAST,
        "algorithms": {"controller": "track_follower"},
        "ground_vehicle": {
            "opponents": {
                "count": 2,
                "mode": mode,
                "per_opponent_options": [
                    {"grip_fraction": 0.6},
                    {"grip_fraction": 0.55},
                ],
            }
        },
    }


@pytest.mark.parametrize("mode", ["kinematic", "dynamic"])
def test_race_with_opponents_is_deterministic(tmp_path, mode):
    layer = race(mode)
    first, rows_a = drive(tmp_path / "a", "car", layer, 8.0, "worlds/race_circuit.json")
    second, rows_b = drive(
        tmp_path / "b", "car", layer, 8.0, "worlds/race_circuit.json"
    )
    assert rows_a == rows_b
    assert first["opponents"] == second["opponents"]
    odometry = first["messages"]["odometry"]
    # Kinematic opponents report at 50 Hz, dynamic ones every 4 ms period.
    expected = odometry if mode == "dynamic" else (odometry - 1) // 5
    assert first["messages"]["opponent:car_1"] == expected
    assert first["message_ordering"]["barrier_timeouts"] == 0
    assert "stale_opponent_states" not in first["messages"]
    assert first["track"]["progress_m"] > 150
    for opponent in first["opponents"]:
        assert opponent["mode"] == mode
        assert opponent["track"]["progress_m"] > 100
        assert opponent["track"]["off_track_episodes"] == 0


class EncoderCounter:
    """Test estimator: counts wheel-encoder samples (no estimate)."""

    inputs = ("wheel_encoders",)

    def __init__(self, context, options):
        self.samples = 0
        self.max_speed_rad_s = 0.0
        self.last_ns = -1
        self.wheels = 0

    def on_sample(self, kind, sample):
        assert sample.sim_time_ns > self.last_ns
        self.last_ns = sample.sim_time_ns
        self.samples += 1
        self.wheels = len(sample.payload.wheel_speed_rad_s)
        self.max_speed_rad_s = max(
            self.max_speed_rad_s, float(np.max(sample.payload.wheel_speed_rad_s))
        )

    def latest(self):
        return None

    def drain_innovations(self):
        return []

    def summary(self):
        return dict(
            samples=self.samples,
            wheels=self.wheels,
            max_speed_rad_s=self.max_speed_rad_s,
        )


def test_wheel_encoders_reach_an_estimator_on_the_bus(tmp_path):
    layer = {
        **CRUISE,
        "algorithms": {
            **CRUISE["algorithms"],
            "estimator": "test_gazebo_wheeled:EncoderCounter",
        },
        # A dynamic opponent shares the wheel model: the bus must carry
        # only the ego's four wheels.
        "ground_vehicle": {"opponents": {"count": 1, "mode": "dynamic"}},
    }
    summary, _ = drive(tmp_path, "car", layer, 3.0, "worlds/race_circuit.json")
    encoders = summary["sensor_bus"]
    assert (
        encoders["submitted"]["wheel_encoders"] == summary["wheel_encoders"]["readings"]
    )
    counter = summary["algorithms"]["estimator_summary"]
    assert counter["samples"] == encoders["delivered"]["wheel_encoders"]
    assert counter["samples"] >= 290  # 100 Hz for 3 s
    assert counter["wheels"] == 4
    # 10 m/s cruise on 0.33 m tyres: about 30 rad/s once up to speed.
    assert counter["max_speed_rad_s"] > 15


class PloughDriver:
    """Test controller: constant drive torque; implement down after 5 s."""

    def __init__(self, context, options):
        self.vehicle = context.wheeled
        self.time_s = 0.0

    def update(self, state, request, dt_s):
        from robotlab.core.wheeled import WheelCommands

        self.time_s = state.sim_time_ns * 1e-9
        n = self.vehicle.wheel_count
        drive = np.where(self.vehicle.mask("driven"), 4000.0, 0.0)
        return WheelCommands(tuple(drive), (0.0,) * n, (0.0,) * n)

    def implement_lowered(self):
        return self.time_s >= 5.0


def test_trailer_static_loads(tmp_path):
    layer = {**FAST, "ground_vehicle": {"trailer": {"enabled": True}}}
    summary, rows = drive(tmp_path, "tractor", layer, 3.0, "worlds/airfield.json")
    assert summary["health"]["verdict"] == "ok"
    assert summary["trailer"]["stale_periods"] == 0
    wheels = [json.loads(line) for line in (tmp_path / "run" / "wheels.jsonl").open()]
    last = wheels[-1]
    trailer = load_drone_config(tmp_path / "layer.json", robot="tractor").data[
        "ground_vehicle"
    ]["trailer"]
    # Axle 0.3 m behind the trailer's centre of mass, drawbar 3 m ahead.
    axle_n = trailer["mass_kg"] * 9.81 * 3.0 / 3.3
    assert sum(last["trailer"]["normal_load_n"]) == pytest.approx(axle_n, rel=0.03)
    total = (3500.0 + trailer["mass_kg"] + 1.0) * 9.81
    assert sum(last["normal_load_n"]) + sum(
        last["trailer"]["normal_load_n"]
    ) == pytest.approx(total, rel=0.005)


def test_plough_draft_slows_the_tractor(tmp_path):
    layer = {
        **FAST,
        "ground_vehicle": {"implement": {"enabled": True}},
        "algorithms": {"controller": "test_gazebo_wheeled:PloughDriver"},
    }
    summary, rows = drive(tmp_path, "tractor", layer, 20.0, "worlds/airfield.json")
    assert summary["health"]["verdict"] == "ok"
    wheels = [json.loads(line) for line in (tmp_path / "run" / "wheels.jsonl").open()]
    last = wheels[-1]["implement"]
    # Verified 2026-10-22: lowered at 5 s from top speed, ploughing at
    # 0.20 m depth and 2.15 m/s against 21.5 kN.
    assert last["depth_m"] == pytest.approx(0.2, abs=0.005)
    assert last["tool_speed_mps"] == pytest.approx(2.153, abs=0.05)
    assert last["draft_n"] == pytest.approx(21466.0, rel=0.02)
    assert summary["implement"]["worked_area_m2"] > 20.0


class ScriptedKeys:
    """Stand-in for the Tk driving window: keys by wall time since start."""

    def __init__(self, script):
        import time as _time

        self._time = _time
        self._start = _time.monotonic()
        self._script = script  # [(until_wall_s, keys, presses)]
        self._pressed: set[int] = set()
        self.closed = False

    def _entry(self):
        elapsed = self._time.monotonic() - self._start
        for index, (until, keys, presses) in enumerate(self._script):
            if elapsed < until:
                return index, keys, presses
        self.closed = True
        return len(self._script), frozenset(), ()

    def poll_keys(self):
        return frozenset(self._entry()[1])

    def drain_presses(self):
        index, _, presses = self._entry()
        if index in self._pressed:
            return []
        self._pressed.add(index)
        return list(presses)

    def close(self):
        self.closed = True


def test_keyboard_driving_path_in_real_time(tmp_path, monkeypatch):
    """Scripted keys through the real keyboard path (DriverInput ramps,
    KeyboardDriver, real-time pacing): accelerate, steer left, brake, and
    end by closing the window."""
    import robotlab.visualization.driving_keyboard as driving

    script = [
        (1.0, set(), ()),
        (5.0, {"Up"}, ()),
        (8.0, {"Up", "Left"}, ()),
        (11.0, {"space"}, ()),
    ]
    monkeypatch.setattr(
        driving, "DrivingKeyboardWindow", lambda title, layout: ScriptedKeys(script)
    )
    output = tmp_path / "run"
    run(
        GazeboConfig(
            robot="car",
            scene="worlds/airfield.json",
            output=str(output),
            headless=True,
            keyboard=True,
            duration=60.0,
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["exit_reason"] == "keyboard_closed"
    assert summary["health"]["verdict"] == "ok"
    assert summary["algorithms"]["controller"] == "keyboard"
    commands = [json.loads(line) for line in (output / "commands.jsonl").open()]
    driver = [c["driver"] for c in commands]
    assert max(d["throttle"] for d in driver) == pytest.approx(1.0)
    assert max(d["steer"] for d in driver) > 0.9
    assert driver[-1]["brake"] == 1.0
    assert summary["max_speed_mps"] > 10.0
    # Python paces the unthrottled world to real time.
    assert summary["real_time"]["achieved_factor"] == pytest.approx(1.0, abs=0.08)


def test_randomized_car_follows_a_route(tmp_path):
    layer = {
        **FAST,
        "algorithms": {"controller": "track_follower"},
        "ground_vehicle": {
            "route": {"waypoints_m": [[0, 0], [60, 0], [80, -20]], "width_m": 6.0},
            "randomization": {
                "enabled": True,
                "seed": 2,
                "mass_fraction_std": 0.05,
                "inertia_fraction_std": 0.1,
                "center_of_mass_std_m": [0.05, 0.02, 0.0],
                "tyre_friction_fraction_std": 0.05,
                "drive_torque_fraction_std": 0.05,
            },
        },
    }
    summary, _ = drive(tmp_path, "car", layer, 15.0, "worlds/airfield.json")
    assert summary["health"]["verdict"] == "ok"
    # The run simulates the true (randomized) car.
    nominal = load_drone_config(robot="car").wheeled.mass_kg
    factor = summary["randomization"]["mass_factor"]
    assert factor != 1.0
    assert summary["robot"]["mass_kg"] == pytest.approx(nominal * factor)
    track = summary["track"]
    assert track["closed"] is False
    assert track["finished_s"] is not None and track["finished_s"] < 15.0
    assert track["progress_m"] > track["length_m"] - 1.5
    assert track["off_track_episodes"] == 0
    assert track["rms_lateral_offset_m"] < 1.0
