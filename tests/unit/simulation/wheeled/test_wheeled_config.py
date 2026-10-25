import json

import pytest

from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.wheeled.config import PRESETS, preset_layer


def write(tmp_path, data, name="layer.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("kind", ["rover", "car", "tractor"])
def test_presets_load(kind):
    config = load_drone_config(robot=kind)
    assert config.kind == kind and config.wheeled is not None
    assert config.wheeled.model_name == kind
    assert config.data["simulation"]["lockstep"]
    assert config.sources == (f"robot preset {kind}",)


def test_multirotor_default_is_unchanged():
    config = load_drone_config()
    assert config.kind == "multirotor" and config.wheeled is None
    assert preset_layer("multirotor") == {}


def test_tractor_matches_published_dimensions():
    tractor = load_drone_config(robot="tractor").wheeled
    assert tractor.mass_kg == 3500.0
    assert tractor.wheelbase_m == pytest.approx(2.2)
    assert tractor.tyre.radius_m == pytest.approx(0.68)
    assert tractor.drive.max_wheel_speed_rad_s * 0.68 * 3.6 == pytest.approx(
        40, abs=0.1
    )
    assert sum(tractor.mask("driven")) == 4 and sum(tractor.mask("steered")) == 4


def test_robot_kind_from_layer_and_conflict(tmp_path):
    layer = write(tmp_path, {"robot": {"kind": "rover"}})
    assert load_drone_config(layer).kind == "rover"
    with pytest.raises(ValueError, match="conflicts"):
        load_drone_config(layer, robot="car")
    with pytest.raises(ValueError, match="robot"):
        load_drone_config(robot="boat")


def test_layer_overrides_preset(tmp_path):
    layer = write(tmp_path, {"ground_vehicle": {"chassis": {"mass_kg": 900.0}}})
    config = load_drone_config(layer, robot="car")
    assert config.wheeled.mass_kg == 900.0
    assert config.provenance["ground_vehicle.chassis.mass_kg"] == str(layer)


@pytest.mark.parametrize(
    "layer, message",
    [
        ({"simulation": {"lockstep": False}}, "lockstep"),
        ({"control": {"odometry_rate_hz": 100.0}}, "odometry_rate_hz"),
        ({"payload": {"enabled": True}}, "multirotor-only"),
        ({"wind": {"enabled": True}}, "multirotor-only"),
        ({"ground_vehicle": {"steering_layout": "tank"}}, "steering_layout"),
        ({"ground_vehicle": {"tyre": {"lateral": {"c": 2.5}}}}, "c must be"),
        ({"ground_vehicle": {"chassis": {"mass_kg": -1}}}, "positive"),
        ({"ground_vehicle": {"wheels": []}}, "at least 3 wheels"),
        ({"ground_vehicle": {"model_name": "my car"}}, "identifier"),
        ({"ground_vehicle": {"spawn_index": -1}}, "integer"),
        ({"ground_vehicle": {"opponents": {"count": 9}}}, "count"),
        (
            {
                "ground_vehicle": {
                    "opponents": {"count": 1, "per_opponent_options": [{}, {}]}
                }
            },
            "more",
        ),
        (
            {"ground_vehicle": {"opponents": {"count": 1, "controller": ""}}},
            "kinematic mode",
        ),
        (
            {
                "ground_vehicle": {
                    "opponents": {"count": 1, "mode": "dynamic", "controller": ""}
                }
            },
            "name a controller",
        ),
        ({"ground_vehicle": {"opponents": {"mode": "ghost"}}}, "mode must be"),
        (
            {"ground_vehicle": {"randomization": {"mass_fraction_std": 0.4}}},
            "below 1/3",
        ),
        (
            {
                "ground_vehicle": {
                    "randomization": {"center_of_mass_std_m": [0, -0.1, 0]}
                }
            },
            ">= 0",
        ),
        ({"ground_vehicle": {"route": {"waypoints_m": [[0, 0]]}}}, "two"),
        (
            {
                "ground_vehicle": {
                    "route": {"waypoints_m": [[0, 0], [5, 0]], "width_m": 0}
                }
            },
            "width_m",
        ),
        (
            {"ground_vehicle": {"opponents": {"count": 1, "controller": "x:Y"}}},
            "kinematic mode",
        ),
        ({"ground_vehicle": {"opponents": {"update_rate_hz": 60.0}}}, "divide"),
        ({"ground_vehicle": {"opponents": {"update_rate_hz": 500.0}}}, "divide"),
        ({"ground_vehicle": {"opponents": {"update_rate_hz": 0}}}, "update_rate_hz"),
    ],
)
def test_invalid_wheeled_settings(tmp_path, layer, message):
    with pytest.raises(ValueError, match=message):
        load_drone_config(write(tmp_path, layer), robot="car")


def test_skid_layout_rejects_steered_wheels(tmp_path):
    wheels = PRESETS["rover"]["ground_vehicle"]["wheels"]
    wheels = [dict(w, steered=True) for w in wheels]
    layer = write(tmp_path, {"ground_vehicle": {"wheels": wheels}})
    with pytest.raises(ValueError, match="skid"):
        load_drone_config(layer, robot="rover")


def test_lockstep_needs_two_steps_per_period(tmp_path):
    layer = write(
        tmp_path, {"simulation": {"step_s": 0.004}}
    )  # one step per 250 Hz period
    with pytest.raises(ValueError, match="two physics steps"):
        load_drone_config(layer, robot="car")


def test_opponent_options_merge():
    from robotlab.experiments.wheeled.opponents import opponent_options

    settings = {
        "ground_vehicle": {
            "opponents": {
                "controller_options": {"a": 1, "b": 2},
                "per_opponent_options": [{"b": 3}],
            }
        }
    }
    assert opponent_options(settings, 0) == {"a": 1, "b": 3}
    assert opponent_options(settings, 1) == {"a": 1, "b": 2}
