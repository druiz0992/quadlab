import json
import os

import numpy as np
import pytest

from robotlab.control.manual import FlightCommand
from robotlab.experiments.run import GazeboConfig, run


@pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo physics/transport",
)
@pytest.mark.parametrize("noisy", [False, True])
def test_live_ekf_tracks_moving_drone_with_unaligned_sensor_rates(tmp_path, noisy):
    config = tmp_path / "drone.json"
    data = {"sensors": {"imu": {"rate_hz": 160}, "gps": {"rate_hz": 7}}}
    if noisy:
        data["sensors"]["imu"].update(
            {
                "mount": {"rotation_deg": [180, 0, 0]},
                "acceleration_mps2": {"noise_stddev": [0.08] * 3},
            }
        )
        data["sensors"]["gps"].update(
            {
                "horizontal_position_m": {"noise_stddev": 0.3},
                "vertical_position_m": {"noise_stddev": 0.5},
            }
        )
    config.write_text(json.dumps(data))
    output = tmp_path / "run"

    def pilot(t):
        return FlightCommand(
            forward_mps=0.5 if 1 <= t < 3 else 0, climb_mps=0.3 if 3 <= t < 4 else 0
        )

    run(
        GazeboConfig(
            drone=str(config),
            output=str(output),
            duration=5,
            headless=True,
            control_mode="rotor-thrust",
            ekf=True,
        ),
        command_source=pilot,
    )
    summary = json.loads((output / "summary.json").read_text())
    estimates = [
        json.loads(line)
        for line in (output / "estimates.jsonl").read_text().splitlines()
    ]
    assert summary["completed"]
    assert summary["ekf"]["gps_updates"] > 20
    assert summary["ekf"]["rejected_gps_timing"] <= 1  # GPS may precede first IMU
    assert not summary["dropped"]
    assert len(estimates) > 400
    assert estimates[0]["position_m"] == [0, 0, 0]  # declared prior, not spawn truth
    assert np.linalg.norm(
        np.array(estimates[-1]["position_m"]) - summary["final_position"]
    ) < (0.8 if noisy else 0.4)
    assert "ground truth" in summary["ekf"]["control_feedback"]


@pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo physics/transport",
)
def test_native_wind_and_modeled_gps_outage_recovery(tmp_path):
    from scipy.spatial.transform import Rotation

    config = tmp_path / "disturbances.json"
    config.write_text(
        json.dumps(
            {
                "wind": {
                    "enabled": True,
                    "velocity_enu_mps": [3, 0, 0],
                    "random_speed_std_mps": 0.1,
                    "random_direction_std_deg": 2,
                    "schedule": [
                        {"time_s": 1, "velocity_enu_mps": [0, 3, 0]},
                        {"time_s": 2, "velocity_enu_mps": [3, 0, 0]},
                    ],
                },
                "sensors": {
                    "gps": {
                        "dropout": {
                            "probability": 0.1,
                            "seed": 7,
                            "outages_s": [[1, 3]],
                        }
                    }
                },
            }
        )
    )
    output = tmp_path / "run"
    run(
        GazeboConfig(
            drone=str(config),
            output=str(output),
            duration=5,
            headless=True,
            control_mode="rotor-thrust",
            ekf=True,
        ),
        command_source=lambda t: FlightCommand(),
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"]
    losses = summary["gps_dropout"]
    assert losses["scheduled_drops"] >= 19
    assert losses["random_drops"] > 0
    assert (
        losses["delivered"] + losses["random_drops"] + losses["scheduled_drops"]
        == losses["received"]
    )
    assert (
        summary["ekf"]["gps_modeled_drops"]
        == losses["random_drops"] + losses["scheduled_drops"]
    )
    assert not summary["dropped"]
    wind_commands = [
        json.loads(line)
        for line in (output / "wind-commands.jsonl").read_text().splitlines()
    ]
    assert len(wind_commands) >= 9
    north = [w for w in wind_commands if 1e9 <= w["sim_time_ns"] < 2e9]
    assert north and all(w["velocity_enu_mps"][1] > 2 for w in north)
    estimates = [
        json.loads(line)
        for line in (output / "estimates.jsonl").read_text().splitlines()
    ]
    during = [e for e in estimates if 1.2e9 < e["sim_time_ns"] < 2.8e9]
    assert len(during) > 100
    assert summary["ekf"]["gps_updates"] > 10
    telemetry = [
        json.loads(line)
        for line in (output / "telemetry.jsonl").read_text().splitlines()
    ]
    # The controller must lean west against an eastward force, rather than
    # merely accepting an inert wind tag with no physical effect.
    pitch = Rotation.from_quat([s["quaternion"] for s in telemetry[-20:]]).as_euler(
        "xyz"
    )[:, 1]
    assert np.mean(pitch) < -0.005
