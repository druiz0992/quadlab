"""Layered drone configuration: repeated files, extends, and provenance."""

import json

import pytest

from robotlab.simulation.gazebo.config import load_drone_config


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def test_later_files_override_earlier_and_objects_merge(tmp_path):
    first = write(
        tmp_path / "a.json", {"pilot": {"climb_mps": 0.4, "forward_mps": 2.0}}
    )
    second = write(tmp_path / "b.json", {"pilot": {"climb_mps": 0.8}})
    pilot = load_drone_config([first, second]).data["pilot"]
    assert pilot["climb_mps"] == 0.8  # later wins
    assert pilot["forward_mps"] == 2.0  # untouched keys survive
    assert pilot["yaw_rate_rad_s"] == 0.7  # built-in default


def test_lists_replace_rather_than_combine(tmp_path):
    zone = {"min_m": [0, 0, 0], "max_m": [1, 1, 1]}
    other = {"min_m": [5, 5, 0], "max_m": [6, 6, 1]}
    first = write(
        tmp_path / "a.json", {"sensors": {"gps": {"dropout": {"zones": [zone]}}}}
    )
    second = write(
        tmp_path / "b.json", {"sensors": {"gps": {"dropout": {"zones": [other]}}}}
    )
    zones = load_drone_config([first, second]).data["sensors"]["gps"]["dropout"][
        "zones"
    ]
    assert zones == [other]


def test_extends_applies_parents_first_relative_to_file(tmp_path):
    write(
        tmp_path / "parts/base.json", {"pilot": {"climb_mps": 0.3, "forward_mps": 1.5}}
    )
    write(tmp_path / "parts/extra.json", {"pilot": {"forward_mps": 2.5}})
    scenario = write(
        tmp_path / "scenarios/s.json",
        {
            "extends": ["../parts/base.json", "../parts/extra.json"],
            "pilot": {"climb_mps": 0.9},
        },
    )
    config = load_drone_config(scenario)
    assert config.data["pilot"]["forward_mps"] == 2.5
    assert config.data["pilot"]["climb_mps"] == 0.9  # the file itself wins over parents
    assert [s.rsplit("/", 1)[-1] for s in config.sources] == [
        "base.json",
        "extra.json",
        "s.json",
    ]
    assert config.provenance["pilot.climb_mps"].endswith("s.json")
    assert config.provenance["pilot.forward_mps"].endswith("extra.json")
    assert "pilot.yaw_rate_rad_s" not in config.provenance  # defaults are not listed


def test_nested_extends_and_single_string(tmp_path):
    write(tmp_path / "root.json", {"pilot": {"climb_mps": 0.2}})
    write(
        tmp_path / "mid.json", {"extends": "root.json", "pilot": {"forward_mps": 3.0}}
    )
    leaf = write(tmp_path / "leaf.json", {"extends": "mid.json"})
    pilot = load_drone_config(leaf).data["pilot"]
    assert (pilot["climb_mps"], pilot["forward_mps"]) == (0.2, 3.0)


@pytest.mark.parametrize(
    "files, message",
    [
        ({"a.json": {"extends": "b.json"}, "b.json": {"extends": "a.json"}}, "cycle"),
        ({"a.json": {"extends": "a.json"}}, "cycle"),
        ({"a.json": {"extends": 3}}, "extends must be"),
        ({"a.json": {"extends": "missing.json"}}, "Cannot load"),
        ({"a.json": [1, 2]}, "JSON object"),
        ({"a.json": {"pilot": {"unknown": 1}}}, "a.json"),  # error names the file
    ],
)
def test_invalid_layers_are_rejected(tmp_path, files, message):
    for name, data in files.items():
        write(tmp_path / name, data)
    with pytest.raises(ValueError, match=message):
        load_drone_config(tmp_path / "a.json")


def test_layer_need_not_be_valid_alone_only_merged_result(tmp_path):
    # Turbulence alone requires wind.enabled; a later layer supplies it.
    turbulence = write(tmp_path / "t.json", {"wind": {"turbulence": {"enabled": True}}})
    wind = write(tmp_path / "w.json", {"wind": {"enabled": True}})
    with pytest.raises(ValueError, match="turbulence requires"):
        load_drone_config(turbulence)
    assert load_drone_config([turbulence, wind]).data["wind"]["turbulence"]["enabled"]


def test_disturbance_scenario_matches_its_building_blocks():
    scenario = load_drone_config("configs/scenarios/disturbances.json")
    blocks = load_drone_config(
        ["configs/environment/gusty_wind.json", "configs/faults/gps_dropout.json"]
    )
    assert scenario.data == blocks.data
    assert scenario.content_hash == blocks.content_hash  # hash depends on settings only
    assert scenario.data["wind"]["random_seed"] == 42
    assert scenario.data["sensors"]["gps"]["dropout"]["outages_s"] == [[10, 15]]


def test_all_example_configs_load_and_combine():
    from pathlib import Path

    # Experiment files (robotlab-batch) and MuJoCo biped settings are not
    # drone layers; tested separately.
    paths = sorted(
        str(p)
        for p in Path("configs").rglob("*.json")
        if p.parent.name not in ("experiments", "bipeds")
    )
    assert len(paths) >= 15
    # Layers may depend on others; the battery monitor needs a battery.
    needs_battery = {
        "configs/sensors/battery_monitor.json",
        "configs/control/failsafe_rth.json",
    }
    for path in paths:
        if path in needs_battery:
            load_drone_config(["configs/vehicles/racing_5in.json", path])
        else:
            load_drone_config(path)
    combined = load_drone_config(
        [
            "configs/vehicles/10in_4s.json",
            "configs/sensors/baro.json",
            "configs/environment/turbulence.json",
        ]
    )
    assert (
        combined.data["battery"]["enabled"]
        and combined.data["sensors"]["baro"]["enabled"]
    )
    assert combined.data["wind"]["turbulence"]["enabled"]


def test_example_experiments_expand():
    from pathlib import Path

    from robotlab.experiments.batch import (
        expand_runs,
        load_experiment,
        resolved_seed_paths,
    )

    for path in Path("configs/experiments").glob("*.json"):
        experiment = load_experiment(path)
        assert expand_runs(experiment, resolved_seed_paths(experiment))
