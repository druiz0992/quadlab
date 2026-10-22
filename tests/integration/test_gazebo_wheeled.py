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
    integrator (two 1 ms steps per 2 ms period, as Gazebo)."""
    config = load_drone_config(layer_path, robot="car")
    vehicle = config.wheeled
    ground = GroundModel.from_scene(load_scene("worlds/airfield.json"), None)
    pose = spawn_pose(vehicle, ground, 0.0, 0.0, 0.0)
    body = RigidBody.at_rest(vehicle.mass_kg, vehicle.inertia_kg_m2, pose.position_m)
    model = WheeledVehicleModel(vehicle, ground)
    controller = TemplateWheeledController(
        synthetic_context(wheeled=vehicle), {"cruise_speed_mps": 10.0}
    )
    for period in range(periods):
        state = body.state()
        if period % 5 == 0:
            control = WheeledControlState(
                period * 2_000_000,
                state.position_m,
                state.velocity_world_mps,
                state.quaternion_xyzw,
                state.body_rates_rad_s,
                model.wheel_speed_rad_s.copy(),
                model.actuators.steer_rad.copy(),
            )
            commands = controller.update(control, None, 0.01 if period else 0.0)
            model.actuators.accept(commands, 0)
        wrench = model.step(state, 0.002)
        for _ in range(2):
            body.step(wrench, 0.001)
    return body.position_m


def test_car_cruise_matches_python_reference(tmp_path):
    summary, rows = drive(tmp_path, "car", CRUISE, 6.0, "worlds/airfield.json")
    assert summary["health"]["verdict"] == "ok"
    assert summary["lockstep"]["steps_per_burst"] == 2
    final = np.array(rows[-1]["position"])
    # Verified 2026-10-22: 48.531 m after 6 s (template controller, 10 m/s).
    assert final[0] == pytest.approx(48.531, abs=0.05)
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
