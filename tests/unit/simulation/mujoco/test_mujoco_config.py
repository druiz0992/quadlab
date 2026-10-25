"""Biped settings: defaults, layering, and validation (no MuJoCo needed)."""

import json
from pathlib import Path

import pytest

from robotlab.simulation.mujoco.config import (
    BUILTIN_MODELS,
    load_biped_config,
    period_steps,
    resolve_model_path,
)
from robotlab.simulation.gazebo.config import load_drone_config, requested_robot_kind


def write(tmp_path: Path, name: str, layer: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(layer))
    return path


def test_defaults_use_the_builtin_test_biped() -> None:
    config = load_biped_config()
    assert config.kind == "biped"
    assert config.model_path == BUILTIN_MODELS["test_biped"]
    assert config.model_path.is_file()
    assert config.sources == ()


def test_layers_merge_and_record_sources(tmp_path: Path) -> None:
    base = write(tmp_path, "base.json", {"biped": {"control_rate_hz": 100.0}})
    top = write(
        tmp_path,
        "top.json",
        {"extends": "base.json", "sensors": {"imu": {"rate_hz": 500.0}}},
    )
    config = load_biped_config(top)
    assert config.data["biped"]["control_rate_hz"] == 100.0
    assert config.data["sensors"]["imu"]["rate_hz"] == 500.0
    assert config.sources == (str(base.parent / "base.json"), str(top))


@pytest.mark.parametrize(
    "layer, message",
    [
        ({"biped": {"unknown": 1}}, "Unknown"),
        ({"biped": {"step_s": -0.001}}, "positive"),
        ({"biped": {"add_floor": 1}}, "boolean"),
        ({"biped": {"control_rate_hz": 300.0}}, "whole number"),
        ({"sensors": {"imu": {"rate_hz": 200.0}}}, "whole number"),
        (
            {"sensors": {"joint_encoders": {"counts_per_revolution": 0}}},
            "positive integer",
        ),
        (
            {"sensors": {"joint_encoders": {"position_noise_std_rad": -1.0}}},
            "nonnegative",
        ),
        ({"sensors": {"joint_encoders": {"seed": 1.5}}}, "nonnegative integer"),
        ({"sensor_timing": {"imu": {"latency_s": -0.1}}}, "nonnegative"),
        ({"health": {"fall_height_fraction": 1.0}}, r"\(0, 1\)"),
        ({"robot": {"kind": "rover"}}, "MuJoCo run"),
        ({"biped": {"model_path": "builtin:nope"}}, "unknown built-in"),
        ({"biped": {"model_path": "/no/such/model.xml"}}, "not found"),
        (
            {
                "sensors": {
                    "imu": {
                        "error_model": {
                            "accel_noise_density_mps2_per_sqrt_hz": [0, -1, 0]
                        }
                    }
                }
            },
            "nonnegative",
        ),
    ],
)
def test_invalid_settings_raise(tmp_path: Path, layer: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_biped_config(write(tmp_path, "bad.json", layer))


def test_period_steps_requires_whole_steps() -> None:
    assert period_steps(50.0, 2_000_000, "control") == 10
    assert period_steps(500.0, 2_000_000, "imu") == 1
    with pytest.raises(ValueError, match="whole number"):
        period_steps(1000.0, 2_000_000, "imu")  # half a step


def test_model_path_expands_user_and_checks_existence(tmp_path: Path) -> None:
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")
    assert resolve_model_path(str(model)) == model


def test_robot_kind_routing(tmp_path: Path) -> None:
    biped = write(tmp_path, "biped.json", {"robot": {"kind": "biped"}})
    assert requested_robot_kind("biped") == "biped"
    assert requested_robot_kind(None, biped) == "biped"
    assert requested_robot_kind(None) == "multirotor"
    with pytest.raises(ValueError, match="conflicts"):
        requested_robot_kind("car", biped)
    # The Gazebo/batch loader refuses MuJoCo kinds with a pointer to the CLI.
    with pytest.raises(ValueError, match="runs in MuJoCo"):
        load_drone_config(biped)


def test_example_biped_configs_load(tmp_path: Path) -> None:
    # External models are not in the repository: point them at the
    # built-in biped to check everything else in each file.
    builtin = write(
        tmp_path, "builtin.json", {"biped": {"model_path": "builtin:test_biped"}}
    )
    paths = sorted(Path("configs/bipeds").glob("*.json"))
    assert paths
    for path in paths:
        assert load_biped_config([path, builtin]).kind == "biped"
