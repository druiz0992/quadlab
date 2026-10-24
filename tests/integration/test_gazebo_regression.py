"""Opt-in Gazebo regression suite: short lockstep runs pinning verified numbers.

Run with ``ROBOTLAB_TEST_GAZEBO=1`` (about 2 minutes wall time); the
quadplane mission (about 17 minutes) also needs ``ROBOTLAB_TEST_GAZEBO_SLOW=1``.
Each run uses lockstep (``configs/control/lockstep.json``) so results are
repeatable, and a fresh pytest temporary directory. The EKF comparison
flies its two missions at once in separate processes (``fly_concurrently``);
the determinism test flies its pair one after the other, without the extra
load. The expected values are
the ones verified by hand and recorded in the numbered docs; tolerances
leave room for floating-point differences across machines, not for
behaviour changes.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robotlab.experiments.gazebo import GazeboConfig, run

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo transport and physics (ROBOTLAB_TEST_GAZEBO=1)",
)
slow = pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO_SLOW") != "1",
    reason="long Gazebo mission (ROBOTLAB_TEST_GAZEBO_SLOW=1)",
)

LOCKSTEP = "configs/control/lockstep.json"
HOLD_1M = {"mission": {"type": "hold", "center_m": [0, 0, 1]}}


def layer_paths(directory: Path, layers: list[str | dict]) -> list[str]:
    """Lockstep first, then ``layers``; dict layers become JSON files in
    ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [LOCKSTEP]
    for index, layer in enumerate(layers):
        if isinstance(layer, dict):
            path = directory / f"layer{index}.json"
            path.write_text(json.dumps(layer))
            layer = str(path)
        paths.append(layer)
    return paths


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
    output = directory / "run"
    run(
        GazeboConfig(
            scene=scene,
            drone=tuple(layer_paths(directory, layers)),
            output=str(output),
            headless=True,
            control_mode="rotor-thrust",
            duration=duration_s,
        )
    )
    return json.loads((output / "summary.json").read_text())


def fly_concurrently(
    flights: list[tuple[Path, list[str | dict], float]],
) -> list[dict]:
    """Run several ``fly`` missions at once, each a separate ``robotlab``
    process with its own transport partition, like ``robotlab-batch``.

    Lockstep results do not depend on machine load, so this only saves
    wall time. Returns the summaries in order; fails with the run's output
    if any process exits non-zero. (In-process spies such as ``capture``
    do not see these runs.)
    """
    processes = []
    for directory, layers, duration_s in flights:
        command = [sys.executable, "-m", "robotlab", "--headless"]
        command += ["--control-mode", "rotor-thrust", "--duration", str(duration_s)]
        for path in layer_paths(directory, layers):
            command += ["--drone", path]
        command += ["--output", str(directory / "run")]
        log = (directory / "robotlab.log").open("w")
        processes.append(
            (
                directory,
                log,
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT),
            )
        )
    failures = []
    for directory, log, process in processes:
        if process.wait() != 0:
            failures.append(directory)
        log.close()
    assert not failures, "\n".join(
        (directory / "robotlab.log").read_text() for directory in failures
    )
    return [
        json.loads((directory / "run" / "summary.json").read_text())
        for directory, _, _ in flights
    ]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def capture(monkeypatch, method: str, key: str, convert) -> list:
    """Collect ``convert(...)`` for recorder calls whose first argument is ``key``."""
    from robotlab.visualization.gazebo_rerun import GazeboRerunRecorder

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
    from robotlab.experiments.magnetometer_calibration import run_calibration

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
    from robotlab.experiments.imu_calibration import run_imu_calibration

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
    truth, estimate = fly_concurrently(
        [
            (tmp_path / "truth", layers + ["configs/algorithms/ekf_observer.json"], 45),
            (
                tmp_path / "estimate",
                layers + ["configs/algorithms/ekf_in_the_loop.json"],
                45,
            ),
        ]
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


@pytest.mark.parametrize(
    "layers",
    [
        ["configs/missions/step_x.json", "configs/algorithms/ekf_in_the_loop.json"],
        # Each varied between identical runs before deterministic ordering,
        # the start-up subscriber barrier, and simulation-time republishing.
        [
            "configs/vehicles/battery_low.json",
            "configs/control/esc_dshot.json",
            "configs/scenarios/ground_start.json",
            "configs/sensors/imu_vibration.json",
            "configs/sensors/sensor_errors.json",
            "configs/sensors/magnetometer.json",
            "configs/missions/square.json",
            "configs/algorithms/ekf_observer.json",
        ],
        # Python-published wind needs the RobotlabLockstep plugin barrier.
        ["configs/environment/turbulence.json", "configs/missions/step_x.json"],
    ],
    ids=["ekf-step", "ground-battery-esc-errors", "turbulence-plugin"],
)
def test_lockstep_runs_are_deterministic(tmp_path, layers):
    from robotlab.simulation.gazebo.plugins import lockstep_plugin_available

    if "turbulence" in layers[0] and not lockstep_plugin_available():
        pytest.skip("needs the lockstep plugin (robotlab-build-plugins)")
    # One after the other, not concurrently: the extra load of a second
    # Gazebo once delayed a GPS sample past the orderer's 5 s wait, which
    # releases periods non-deterministically (reported, but not this test).
    first = fly(tmp_path / "a", layers, 8)
    second = fly(tmp_path / "b", layers, 8)
    for summary in (first, second):
        ordering = summary["message_ordering"]
        assert ordering["barrier_timeouts"] == 0, (
            "machine too loaded for a determinism check: lockstep orderer "
            f"timed out on {ordering['timeout_streams']}"
        )
    compared = 0
    for name in ("telemetry.jsonl", "commands.jsonl", "estimates.jsonl"):
        a, b = (tmp_path / side / "run" / name for side in ("a", "b"))
        assert a.exists() == b.exists(), name
        if a.exists():
            assert a.stat().st_size > 0, name
            # Compare cheaply: pytest's diff of multi-megabyte strings can
            # take hours. Report the first differing line instead.
            difference = first_difference(a, b)
            assert difference is None, f"{name} differs: {difference}"
            compared += 1
    assert compared >= 2
    assert first["metrics"] == second["metrics"]
    assert first["health"] == second["health"]


def first_difference(a: Path, b: Path) -> str | None:
    """None when the files are identical, else the first differing line."""
    if a.read_bytes() == b.read_bytes():
        return None
    lines_a, lines_b = a.read_text().splitlines(), b.read_text().splitlines()
    for number, (line_a, line_b) in enumerate(zip(lines_a, lines_b), start=1):
        if line_a != line_b:
            return f"line {number}: {line_a[:200]!r} != {line_b[:200]!r}"
    return f"lengths differ: {len(lines_a)} vs {len(lines_b)} lines"


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
    # Verified: lands within centimetres of home after the planned touchdown
    # (about 150 s at the site air density, 152 s at 1.225 kg/m³).
    # Tracking is scored against the autopilot's own path (cross-track,
    # tracking_reference): the baseline L1 guidance swings up to 65 m wide
    # of the 300 m square's corners (~70 m minimum turn radius at 22 m/s and
    # 35° roll), so tracking_lost is a real finding (from 44 s, transition).
    assert summary["completed"]
    assert summary["health"]["verdict"] == "warning", summary["health"]
    assert {f["kind"] for f in summary["health"]["findings"]} == {
        "ground_contact",
        "tracking_lost",
    }
    control = summary["metrics"]["control"]
    assert control["tracking_reference"] == "controller"
    position = control["tracking"]["position_m"]
    assert position["rmse"] == pytest.approx(20.7, abs=0.5)
    assert position["max"] == pytest.approx(65.1, abs=1.0)
    x, y, z = summary["final_position"]
    assert np.hypot(x, y) < 1.0 and z < 0.5


def test_replay_reproduces_live_estimates(tmp_path):
    """Recorded bus deliveries replayed offline give the live EKF output."""
    from robotlab.experiments.replay import replay

    live = fly(
        tmp_path,
        [
            "configs/sensors/sensor_errors.json",
            "configs/missions/square.json",
            "configs/algorithms/ekf_observer.json",
            "configs/algorithms/record_replay.json",
        ],
        12,
    )
    run_dir = tmp_path / "run"
    summary = replay(run_dir / "replay.jsonl", tmp_path / "replay")
    assert summary["estimates"] > 1000
    assert (tmp_path / "replay" / "estimates.jsonl").read_bytes() == (
        run_dir / "estimates.jsonl"
    ).read_bytes()
    replayed = json.loads((tmp_path / "replay" / "metrics.json").read_text())
    assert replayed["estimation"] == live["metrics"]["estimation"]


def test_gazebo_environment_matches_python_models(tmp_path, monkeypatch):
    # Audited 2026-10-21 (docs/2_simulation.md#simulator-environment-audit):
    # Gazebo's barometer and NavSat include the 650 m site elevation and
    # agree with the Python atmosphere and geodesy models; the magnetic
    # table gives 45.1 µT, inclination 55.4°, zero declination at Madrid;
    # a 200 Hz IMU on 2 ms physics steps alternates 4 and 6 ms intervals.
    from robotlab.sensors.barometer import pressure_to_altitude

    stamp = lambda message: message.header.stamp  # noqa: E731
    ns = lambda s: s.sec * 1_000_000_000 + s.nsec  # noqa: E731
    baro = capture(monkeypatch, "log", "baro", lambda m, t: (t, m.pressure))
    gps = capture(monkeypatch, "log", "gps", lambda m, t: (t, m.altitude))
    field = capture(
        monkeypatch,
        "log",
        "magnetometer",
        lambda m, t: [m.field_tesla.x, m.field_tesla.y, m.field_tesla.z],
    )
    imu = capture(monkeypatch, "log", "imu", lambda m, t: ns(stamp(m)))
    no_errors = {"error_model": {"enabled": False}}
    fly(
        tmp_path,
        [
            HOLD_1M,
            "configs/sensors/baro.json",
            "configs/sensors/magnetometer.json",
            {"sensors": {"baro": no_errors, "magnetometer": no_errors}},
        ],
        4,
    )
    telemetry = jsonl(tmp_path / "run" / "telemetry.jsonl")
    heights = {round(r["time"], 3): r["position"][2] for r in telemetry}
    t, pressure = baro[-1]
    assert pressure_to_altitude(pressure) == pytest.approx(
        650 + heights[round(t, 3)], abs=0.01
    )
    t, altitude = gps[-1]
    assert altitude == pytest.approx(650 + heights[round(t, 3)], abs=0.01)
    # Level, facing east: sensor axes are world east/north/up.
    np.testing.assert_allclose(
        np.asarray(field[-1]) * 1e6, [0.0, 25.613, -37.152], atol=0.01
    )
    intervals_ms = set(np.diff(imu) // 1_000_000)
    assert intervals_ms == {4, 6}
    assert np.mean(np.diff(imu)) == pytest.approx(5e6, rel=0.01)


def _ground_texture_std(image_path: Path) -> float:
    """Median standard deviation of 8×8 grey blocks in the bottom 50 rows
    of a camera frame (the ground below the horizon)."""
    from matplotlib.image import imread

    grey = imread(image_path)[..., :3].mean(axis=2) * 255
    rows = grey[-48:, : grey.shape[1] // 8 * 8]
    blocks = rows.reshape(6, 8, -1, 8).std(axis=(1, 3))
    return float(np.median(blocks))


def test_textured_yard_renders_its_textures(tmp_path):
    """Verified 2026-10-24: the textured yard's first camera frame has a
    median local ground texture of 2.9 grey levels (training world: 0.3),
    and Gazebo loads every textured mesh without warnings."""
    output = tmp_path / "run"
    run(
        GazeboConfig(
            scene="worlds/textured_yard.json",
            drone=tuple(layer_paths(tmp_path, [HOLD_1M])),
            output=str(output),
            headless=True,
            control_mode="rotor-thrust",
            duration=3.0,
            render_sensors=True,
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["health"]["verdict"] == "ok"
    assert _ground_texture_std(output / "camera-first.png") > 1.5
    assert "Missing material" not in (output / "gazebo.log").read_text()
