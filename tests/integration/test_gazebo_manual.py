"""Opt-in real Gazebo checks: ROBOTLAB_TEST_GAZEBO=1 pytest this_file."""

import json
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.control.manual import FlightCommand
from robotlab.experiments.gazebo import GazeboConfig, run


@pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo transport and physics",
)
def test_manual_thrust_flight(tmp_path):
    def pilot(t):
        if 2 <= t < 4:
            return FlightCommand(forward_mps=1)
        if 4 <= t < 6:
            return FlightCommand(yaw_rate_rps=0.7)
        if 6 <= t < 8:
            return FlightCommand.from_keys({"q"})
        if 8 <= t < 10:
            return FlightCommand.from_keys({"a"})
        if 10 <= t < 12:
            return FlightCommand(forward_mps=-1)
        return FlightCommand()

    run(
        GazeboConfig(
            headless=True,
            control_mode="rotor-thrust",
            duration=15,
            output=str(tmp_path),
            scene="worlds/circuit.json",
        ),
        command_source=pilot,
    )
    telemetry = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    times = np.array([s["time"] for s in telemetry])
    times -= times[0]

    def at(t):
        return telemetry[np.abs(times - t).argmin()]

    def position(t):
        return np.array(at(t)["position"])

    def yaw(t):
        q = at(t)["quaternion"]
        r = Rotation.from_quat(q).as_matrix()
        return np.arctan2(r[1, 0], r[0, 0])

    assert position(4)[0] - position(2)[0] > 0.7
    assert yaw(6) - yaw(4) > 0.7
    assert position(8)[2] - position(6)[2] > 0.5
    assert position(10)[2] - position(8)[2] < -0.5
    heading = np.array([np.cos(yaw(10)), np.sin(yaw(10)), 0])
    assert np.dot(position(12) - position(10), heading) < -0.7
    assert np.linalg.norm(position(15) - position(14)) < 0.2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"]
    assert not summary["collisions"]
    assert not summary["dropped"]
    assert summary["saturated_commands"] == 0
    assert summary["control_mode"] == "rotor-thrust"
    events = [
        json.loads(line)
        for line in (tmp_path / "command-events.jsonl").read_text().splitlines()
    ]
    published = [event for event in events if event["status"] == "published"]
    assert published and all(len(event["thrust_n"]) == 4 for event in published)
    assert events[-1]["status"] == "stopped"
    assert events[-1]["thrust_n"] == [0] * 4


@pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo transport and physics",
)
@pytest.mark.parametrize("roll_deg", [0, 180])
def test_configured_mass_limits_and_imu_error(tmp_path, monkeypatch, roll_deg):
    from robotlab.visualization.gazebo_rerun import GazeboRerunRecorder

    config = tmp_path / "drone.json"
    config.write_text(
        json.dumps(
            {
                "vehicle": {"body_mass_kg": 1.6},
                "motors": {"max_thrust_n": 6.0},
                "sensors": {
                    "gps": {"enabled": False},
                    "imu": {
                        "rate_hz": 100,
                        "mount": {"rotation_deg": [roll_deg, 0, 0]},
                        "acceleration_mps2": {
                            "bias": [0.2, -0.3, 0.5],
                            "noise_stddev": [0.05] * 3,
                        },
                    },
                },
            }
        )
    )
    readings = []
    original = GazeboRerunRecorder.log

    def capture(self, kind, message, t):
        if kind == "imu" and t > 2:
            v = message.linear_acceleration
            readings.append([v.x, v.y, v.z])
        original(self, kind, message, t)

    monkeypatch.setattr(GazeboRerunRecorder, "log", capture)
    output = tmp_path / "run"
    run(
        GazeboConfig(
            drone=str(config),
            output=str(output),
            headless=True,
            control_mode="rotor-thrust",
            duration=4,
        )
    )
    values = np.asarray(readings)
    assert len(values) > 100
    expected_z = 9.81 * (1 if roll_deg == 0 else -1) + 0.5
    np.testing.assert_allclose(values.mean(axis=0), [0.2, -0.3, expected_z], atol=0.08)
    assert np.all(values.std(axis=0) > 0.02)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] and "gps" not in summary["messages"]
    assert summary["final_position"][2] == pytest.approx(1.0, abs=0.2)
    commands = [
        json.loads(line)
        for line in (output / "commands.jsonl").read_text().splitlines()
    ]
    assert max(max(command["thrust_n"]) for command in commands) <= 6
