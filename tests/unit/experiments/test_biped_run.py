"""Biped missions in MuJoCo with the built-in test biped (mujoco extra)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robotlab.experiments.biped_loop import BipedLoop
from robotlab.experiments.biped_run import _parts, run_biped
from robotlab.experiments.gazebo import GazeboConfig, run
from robotlab.simulation.mujoco.config import BUILTIN_MODELS, load_biped_config


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(ekf=True), "--ekf"),
        (dict(world="saved.sdf"), "--world"),
        (dict(keyboard=True), "--keyboard"),
        (dict(rerun=True), "--rerun"),
        (dict(control_mode="rotor-thrust"), "multirotors only"),
        (dict(scene="worlds/yard.json"), "biped.model_path"),
        (dict(target=(1.0, 0.0, 1.0)), "multirotor missions"),
        (dict(duration=0.0), "Duration"),
    ],
)
def test_option_conflicts_fail_before_mujoco(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    args: dict[str, Any] = dict(robot="biped", output=str(tmp_path / "out"))
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run(GazeboConfig(**args))
    assert not (tmp_path / "out").exists()


mujoco = pytest.importorskip("mujoco")


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_generate_only_writes_scene(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "out"
    assert run(GazeboConfig(robot="biped", output=str(out), generate_only=True)) is None
    assert (out / "scene.xml").exists() and "scene.xml" in capsys.readouterr().out
    assert not (out / "summary.json").exists()


def test_standing_mission_outputs_and_timing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run(GazeboConfig(robot="biped", output=str(out), duration=1.0))
    assert summary is not None and summary["completed"]
    assert summary["health"]["verdict"] == "ok"
    assert summary == json.loads((out / "summary.json").read_text())
    assert summary["timing"]["steps"] == 500
    telemetry = lines(out / "telemetry.jsonl")
    # 50 Hz from t = 0, plus the final state at exactly 1 s.
    stamps = [line["sim_time_ns"] for line in telemetry]
    assert stamps == [k * 20_000_000 for k in range(50)] + [1_000_000_000]
    assert len(lines(out / "commands.jsonl")) == 50
    assert summary["bus"]["delivered"] == {"imu": 250, "joint_encoders": 50}
    assert summary["robot"]["joint_names"][0] == "left_hip_pitch"
    assert abs(telemetry[-1]["position_m"][2] - 0.166) < 0.002


def mission(
    tmp_path: Path, layer: dict | None = None, duration_s: float = 1.0
) -> BipedLoop:
    paths = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    if layer is not None:
        path = tmp_path / "layer.json"
        path.write_text(json.dumps(layer))
        paths.append(path)
    config = load_biped_config(paths)
    from robotlab.simulation.mujoco.scene import build_scene

    scene = build_scene(config.data, config.model_path)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return BipedLoop(_parts(config, scene, out, duration_s))


def test_imu_samples_on_the_bus_read_specific_force(tmp_path: Path) -> None:
    loop = mission(tmp_path)
    subscription = loop.parts.bus.channel("imu").subscribe(capacity=1000)
    loop.run()
    samples = [subscription.next(timeout_wall_s=0) for _ in range(250)]
    assert [s.sim_time_ns for s in samples] == [k * 4_000_000 for k in range(250)]
    settled = np.array([s.payload.specific_force_mps2 for s in samples[100:]])
    np.testing.assert_allclose(settled.mean(axis=0), [0, 0, 9.81], atol=0.05)


def test_runs_are_deterministic_with_noise(tmp_path: Path) -> None:
    layer = {
        "sensors": {
            "imu": {
                "error_model": {
                    "enabled": True,
                    "seed": 4,
                    "gyro_noise_density_rad_s_per_sqrt_hz": [0.01] * 3,
                }
            },
            "joint_encoders": {"position_noise_std_rad": 0.002},
        }
    }
    first, second = mission(tmp_path / "a", layer), mission(tmp_path / "b", layer)
    one, two = first.run(), second.run()
    for name in ("telemetry.jsonl", "commands.jsonl"):
        assert (tmp_path / "a/out" / name).read_text() == (
            tmp_path / "b/out" / name
        ).read_text()
    assert one["sensors"] == two["sensors"]


def test_fall_is_a_failed_verdict(tmp_path: Path) -> None:
    # Start the test biped tilted 50 degrees about x: it falls over.
    text = (
        BUILTIN_MODELS["test_biped"]
        .read_text()
        .replace('qpos="0 0 0.167 1 0 0 0', 'qpos="0 0 0.2 0.906 0.423 0 0')
    )
    model = tmp_path / "tilted.xml"
    model.write_text(text)
    summary = mission(tmp_path, {"biped": {"model_path": str(model)}}, 2.0).run()
    kinds = [finding["kind"] for finding in summary["health"]["findings"]]
    assert summary["health"]["verdict"] == "failed" and "fallen" in kinds


def test_interrupt_and_error_still_write_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = mission(tmp_path)
    calls = {"n": 0}
    original = loop.parts.encoders.read

    def interrupt(*args: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 10:
            raise KeyboardInterrupt
        return original(*args)

    monkeypatch.setattr(loop.parts.encoders, "read", interrupt)
    summary = loop.run()
    assert not summary["completed"] and summary["exit_reason"] == "interrupted"
    assert summary["health"]["verdict"] == "warning"
    assert summary["timing"]["steps"] == 90

    failing = mission(tmp_path / "f")
    monkeypatch.setattr(failing.parts.encoders, "read", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        failing.run()
    written = json.loads((tmp_path / "f/out/summary.json").read_text())
    assert (
        written["exit_reason"] == "error" and written["health"]["verdict"] == "failed"
    )


def test_run_biped_missing_model_explains_checkout(tmp_path: Path) -> None:
    layer = tmp_path / "duck.json"
    layer.write_text(
        json.dumps(
            {
                "robot": {"kind": "biped"},
                "biped": {"model_path": str(tmp_path / "x.xml")},
            }
        )
    )
    args = GazeboConfig(drone=str(layer), output=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="microduck_rl"):
        run_biped(args)
    assert BUILTIN_MODELS["test_biped"].exists()


class _Recorder:
    """Controller that records its inputs and requests out-of-range targets."""

    def __init__(self, target: float = 0.0) -> None:
        self.states: list[Any] = []
        self.dts: list[float] = []
        self.target = target

    def update(self, state: Any, request: Any, dt_s: float) -> Any:
        from robotlab.core.legged import JointTargets

        self.states.append(state)
        self.dts.append(dt_s)
        return JointTargets((self.target,) * 6)

    def summary(self) -> dict:
        return {"calls": len(self.states)}


def test_controller_sees_samples_truth_and_dt(tmp_path: Path) -> None:
    loop = mission(tmp_path, duration_s=0.2)
    recorder = _Recorder()
    loop.parts.controller, loop.parts.controller_spec = recorder, "recorder"
    summary = loop.run()
    first, later = recorder.states[0], recorder.states[3]
    # Encoders are read before control at t = 0; the first IMU sample
    # (measured at 0) arrives one physics step later.
    assert first.joints.sim_time_ns == 0 and first.imu is None
    assert later.imu.sim_time_ns == later.sim_time_ns - 4_000_000
    assert later.trunk is not None and later.source == "truth"
    assert recorder.dts[:3] == [0.0, 0.02, 0.02]
    assert summary["control"]["controller"] == "recorder"
    assert summary["control"]["controller_summary"] == {"calls": 10}


def test_sensors_only_source_hides_the_trunk(tmp_path: Path) -> None:
    loop = mission(tmp_path, {"algorithms": {"control_state_source": "sensors"}}, 0.1)
    recorder = _Recorder()
    loop.parts.controller = recorder
    summary = loop.run()
    assert all(s.trunk is None and s.source == "sensors" for s in recorder.states)
    assert summary["control"]["state_source"] == "sensors"


def test_out_of_range_targets_are_clipped_and_counted(tmp_path: Path) -> None:
    loop = mission(tmp_path, duration_s=0.1)
    loop.parts.controller = _Recorder(target=2.0)
    summary = loop.run()
    assert set(summary["control"]["limited_commands"].values()) == {5}
    line = lines(tmp_path / "out" / "commands.jsonl")[0]
    assert line["joint_target_rad"] == [1.5] * 6
    assert line["requested_rad"] == [2.0] * 6 and line["limited"] == list(range(6))


def test_invalid_controller_output_fails_the_run(tmp_path: Path) -> None:
    loop = mission(tmp_path, duration_s=0.1)

    class WrongLength(_Recorder):
        def update(self, state: Any, request: Any, dt_s: float) -> Any:
            from robotlab.core.legged import JointTargets

            return JointTargets((0.0,) * 5)

    loop.parts.controller, loop.parts.controller_spec = WrongLength(), "wrong"
    with pytest.raises(RuntimeError, match="wrong: invalid JointTargets"):
        loop.run()
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert written["health"]["verdict"] == "failed"


def test_template_controller_runs_from_config(tmp_path: Path) -> None:
    layer = {
        "algorithms": {
            "controller": "template",
            "controller_options": {"lean_gains": {"left_ankle_pitch": 0.5}},
        }
    }
    summary = mission(tmp_path, layer, 0.5).run()
    assert summary["control"]["controller"] == "template"
    assert summary["health"]["verdict"] == "ok"


@pytest.mark.parametrize(
    "spec, message",
    [
        ("nope.module:Class", "Cannot import"),
        (  # Constructs from (context, options) but has no update().
            "robotlab.testing.plugins:LeggedControllerReport",
            "does not implement LeggedController",
        ),
    ],
)
def test_bad_controller_specs_raise(tmp_path: Path, spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mission(tmp_path, {"algorithms": {"controller": spec}})
