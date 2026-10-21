"""Opt-in Gazebo regression suite: short lockstep runs pinning verified numbers.

Run with ``DRONELAB_TEST_GAZEBO=1`` (about 5 minutes wall time); the
quadplane mission (about 17 minutes) also needs ``DRONELAB_TEST_GAZEBO_SLOW=1``.
Each run uses lockstep (``configs/control/lockstep.json``) so results are
repeatable, and a fresh pytest temporary directory. The expected values are
the ones verified by hand and recorded in the numbered docs; tolerances
leave room for floating-point differences across machines, not for
behaviour changes.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dronelab.experiments.gazebo import GazeboConfig, run

pytestmark = pytest.mark.skipif(
    os.environ.get("DRONELAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo transport and physics (DRONELAB_TEST_GAZEBO=1)",
)
slow = pytest.mark.skipif(
    os.environ.get("DRONELAB_TEST_GAZEBO_SLOW") != "1",
    reason="long Gazebo mission (DRONELAB_TEST_GAZEBO_SLOW=1)",
)

LOCKSTEP = "configs/control/lockstep.json"
HOLD_1M = {"mission": {"type": "hold", "center_m": [0, 0, 1]}}


def fly(
    directory: Path,
    layers: list[str | dict],
    duration_s: float,
    scene: str = "worlds/training.json",
) -> dict:
    """Run one headless lockstep rotor-thrust mission; return summary.json.

    Dict layers are written as JSON files in ``directory``; the run's
    outputs go to ``directory / "run"``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = [LOCKSTEP]
    for index, layer in enumerate(layers):
        if isinstance(layer, dict):
            path = directory / f"layer{index}.json"
            path.write_text(json.dumps(layer))
            layer = str(path)
        paths.append(layer)
    output = directory / "run"
    run(
        GazeboConfig(
            scene=scene,
            drone=tuple(paths),
            output=str(output),
            headless=True,
            control_mode="rotor-thrust",
            duration=duration_s,
        )
    )
    return json.loads((output / "summary.json").read_text())


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def capture(monkeypatch, method: str, key: str, convert) -> list:
    """Collect ``convert(...)`` for recorder calls whose first argument is ``key``."""
    from dronelab.visualization.gazebo_rerun import GazeboRerunRecorder

    values = []
    original = getattr(GazeboRerunRecorder, method)

    def spy(self, name, *args, **kwargs):
        if name == key:
            values.append(convert(*args))
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(GazeboRerunRecorder, method, spy)
    return values


def test_center_of_mass_offset_steady_state(tmp_path):
    # Integral-free controller: predicted g·tan(e)/K_pos = 0.69 m; verified 0.67 m.
    summary = fly(
        tmp_path,
        [HOLD_1M, {"vehicle": {"center_of_mass_offset_m": [0.01, 0, 0]}}],
        20,
    )
    x, y, z = summary["final_position"]
    assert x == pytest.approx(0.67, abs=0.03)
    assert abs(y) < 0.02 and z == pytest.approx(1.0, abs=0.05)
    assert summary["health"]["verdict"] == "ok"


def _mean_field_z_ut(tmp_path, monkeypatch, dipoles) -> float:
    """Mean measured z field (µT) after the hover settles (t > SETTLED_S)."""
    readings = capture(
        monkeypatch,
        "log",
        "magnetometer",
        lambda message, t: (t, message.field_tesla.z),
    )
    layer = {"sensors": {"magnetometer": {"enabled": True, "dipoles": dipoles}}}
    fly(tmp_path, [HOLD_1M, layer], 8)
    values = [z for t, z in readings if t > SETTLED_S]
    assert len(values) > 200
    return float(np.mean(values)) * 1e6


def _dipole_z_ut(position_m, dipole) -> float:
    """z of μ₀/4π · (3r̂(m·r̂) − m)/|r|³ in µT (level vehicle, mount at origin)."""
    r = np.asarray(position_m, float) - dipole["position_m"]
    d = np.linalg.norm(r)
    m = np.asarray(dipole["moment_am2"], float)
    return float((1e-7 * (3 * r * (m @ r) / d**2 - m) / d**3)[2] * 1e6)


SETTLED_S = 3.0


def test_dipole_beside_hover_changes_vertical_field(tmp_path, monkeypatch):
    # Equatorial dipole field −μ₀/4π·m/r³ = −1e-7·2000/2³ T = −25 µT.
    dipole = {"position_m": [2, 0, 1], "moment_am2": [0, 0, 2000]}
    with_dipole = _mean_field_z_ut(tmp_path / "with", monkeypatch, [dipole])
    monkeypatch.undo()
    without = _mean_field_z_ut(tmp_path / "without", monkeypatch, [])
    shift = with_dipole - without
    # Exact against the field at the true positions (hover height varies).
    telemetry = jsonl(tmp_path / "with" / "run" / "telemetry.jsonl")
    expected = np.mean(
        [
            _dipole_z_ut(s["position"], dipole)
            for s in telemetry
            if s["time"] > SETTLED_S
        ]
    )
    assert shift == pytest.approx(expected, abs=0.02)
    assert shift == pytest.approx(-25.0, abs=0.5)


def test_magnetometer_calibration_recovers_injected_errors(tmp_path):
    from dronelab.experiments.magnetometer_calibration import run_calibration

    report = run_calibration(["configs/sensors/magnetometer.json"], tmp_path)
    layer = json.loads((tmp_path / "magnetometer-calibration.json").read_text())
    section = layer["sensors"]["magnetometer"]["calibration"]
    np.testing.assert_allclose(
        section["hard_iron_offset_ut"], [4.0, -2.5, 1.5], atol=0.1
    )
    np.testing.assert_allclose(
        section["thrust_compensation_ut_per_n"], [0.15, -0.05, 0.10], atol=0.01
    )
    after, before = report["evaluation"]["after"], report["evaluation"]["before"]
    assert after["heading_rms_deg"] < 2.0 < before["heading_rms_deg"]


def test_imu_calibration_recovers_scale_and_writes_priors(tmp_path):
    from dronelab.experiments.imu_calibration import run_imu_calibration

    report = run_imu_calibration(
        ["configs/sensors/imu_consumer_mems.json"], tmp_path, startup_gyro_s=2
    )
    layer = json.loads((tmp_path / "imu-calibration.json").read_text())
    section = layer["sensors"]["imu"]["calibration"]
    distortion = np.asarray(report["injected_truth"]["accel_scale_and_misalignment"])
    # Correction × distortion = identity (verified to within 0.014 %).
    np.testing.assert_allclose(
        np.asarray(section["accel_correction"]) @ distortion, np.eye(3), atol=5e-4
    )
    assert section["enabled"] and section["startup_gyro_calibration_s"] == 2
    for key in ("filter_accel_bias_std_mps2", "filter_gyro_bias_std_rad_s"):
        assert 0 < section[key] == report["filter_priors"][key]
    after, before = report["evaluation"]["after"], report["evaluation"]["before"]
    assert after["accel_error_rms_mps2"] < 0.1 * before["accel_error_rms_mps2"]


def test_ekf_in_the_loop_tracks_like_truth_control(tmp_path):
    # Verified: 0.102 m on truth, 0.103 m on the estimate, 4 mm estimation RMSE.
    layers = ["configs/missions/square.json"]
    truth = fly(
        tmp_path / "truth", layers + ["configs/algorithms/ekf_observer.json"], 45
    )
    estimate = fly(
        tmp_path / "estimate", layers + ["configs/algorithms/ekf_in_the_loop.json"], 45
    )
    on_truth = truth["metrics"]["control"]["tracking"]["position_m"]["rmse"]
    on_estimate = estimate["metrics"]["control"]["tracking"]["position_m"]["rmse"]
    assert on_truth == pytest.approx(0.102, abs=0.01)
    assert on_estimate == pytest.approx(on_truth, abs=0.01)
    errors = estimate["metrics"]["estimation"]["errors"]["position_m"]
    assert errors["rmse"] < 0.01
    assert estimate["health"]["verdict"] == "ok"
    assert truth["health"]["verdict"] == "ok"


def test_esc_telemetry_matches_commanded_speeds(tmp_path, monkeypatch):
    reports = capture(
        monkeypatch, "log_vector", "esc/rotor_speed_rad_s", lambda v, *a: list(v)
    )
    summary = fly(tmp_path, [HOLD_1M, "configs/control/esc_dshot.json"], 6)
    commands = jsonl(tmp_path / "run" / "commands.jsonl")
    commanded = np.mean(
        [c["motor_speed_rad_s"] for c in commands if c["time"] > 2], axis=0
    )
    measured = np.asarray(reports[len(reports) // 3 :])  # after take-off settles
    assert summary["esc"]["telemetry"]["samples"] == len(reports) > 300
    # Verified: mean within 0.3 rad/s, with the configured 3 rad/s noise.
    np.testing.assert_allclose(measured.mean(axis=0), commanded, atol=1.0)
    assert np.all(measured.std(axis=0) > 2.0)


def test_lockstep_runs_are_deterministic(tmp_path):
    layers = ["configs/missions/step_x.json", "configs/algorithms/ekf_in_the_loop.json"]
    first = fly(tmp_path / "a", layers, 8)
    second = fly(tmp_path / "b", layers, 8)
    for name in ("telemetry.jsonl", "commands.jsonl", "estimates.jsonl"):
        a = (tmp_path / "a" / "run" / name).read_text()
        assert a and a == (tmp_path / "b" / "run" / name).read_text(), name
    assert first["metrics"] == second["metrics"]
    assert first["health"] == second["health"]


def test_battery_crash_is_reported_as_failed(tmp_path):
    summary = fly(
        tmp_path,
        [
            {
                "battery": {"enabled": True, "capacity_mah": 1500, "initial_soc": 0.01},
                "simulation": {"spawn_position_m": [0, 0, 2]},
                "mission": {"type": "hold", "center_m": [0, 0, 2]},
            }
        ],
        8,
    )
    health = summary["health"]
    assert health["verdict"] == "failed"
    impact = next(f for f in health["findings"] if f["kind"] == "hard_impact")
    assert impact["value"] > 4.0 and 2.0 < impact["time_s"] < 6.0
    assert summary["final_position"][2] < 0.3


@slow
def test_quadplane_full_mission(tmp_path):
    summary = fly(
        tmp_path,
        [
            "configs/vehicles/kite_quadplane.json",
            "configs/algorithms/quadplane_autopilot.json",
        ],
        160,
        scene="worlds/airfield.json",
    )
    # Verified: lands within centimetres of home; the only finding is the
    # planned touchdown (about 152 s).
    assert summary["completed"]
    assert summary["health"]["verdict"] == "warning", summary["health"]
    assert {f["kind"] for f in summary["health"]["findings"]} == {"ground_contact"}
    x, y, z = summary["final_position"]
    assert np.hypot(x, y) < 1.0 and z < 0.5
