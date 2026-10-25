import pytest

from robotlab.experiments.run import GazeboConfig, run
from robotlab.experiments.wheeled.wheeled_run import build_wheeled_controller
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.testing.plugins import synthetic_context


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(ekf=True), "algorithms.estimator"),
        (dict(world="saved.sdf"), "generated world"),
        (dict(control_mode="rotor-thrust"), "multirotors only"),
        (dict(duration=0.0), "Duration"),
        (dict(keyboard=True, edit=True), "mission mode"),
    ],
)
def test_option_conflicts_fail_before_gazebo(tmp_path, kwargs, message):
    args = dict(robot="car", output=str(tmp_path / "out"), headless=True)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run(GazeboConfig(**args))


def test_generate_only_writes_world(tmp_path, capsys):
    run(GazeboConfig(robot="tractor", output=str(tmp_path / "out"), generate_only=True))
    assert (tmp_path / "out" / "world.sdf").exists()
    assert "world.sdf" in capsys.readouterr().out


def test_controller_selection():
    config = load_drone_config(robot="car")
    context = synthetic_context(wheeled=config.wheeled)
    assert build_wheeled_controller(config.data, context, keyboard=False) is None
    driver = build_wheeled_controller(config.data, context, keyboard=True)
    assert type(driver).__name__ == "KeyboardDriver"
    data = dict(
        config.data, algorithms=dict(config.data["algorithms"], controller="template")
    )
    assert (
        type(build_wheeled_controller(data, context, False)).__name__
        == "TemplateWheeledController"
    )
    bad = dict(
        config.data,
        algorithms=dict(
            config.data["algorithms"],
            controller="robotlab.control.template:TemplateController",
        ),
    )
    with pytest.raises(ValueError, match="WheeledController"):
        build_wheeled_controller(bad, context, False)


@pytest.mark.parametrize(
    "layer, message",
    [
        ({"randomization": {"enabled": True}}, "ground_vehicle.randomization"),
        (
            {"mission": {"type": "waypoints", "waypoints_m": [[1, 0, 1]]}},
            "ground_vehicle.route",
        ),
    ],
)
def test_multirotor_sections_point_to_the_wheeled_ones(tmp_path, layer, message):
    import json

    path = tmp_path / "layer.json"
    path.write_text(json.dumps(layer))
    args = dict(robot="car", output=str(tmp_path / "out"), drone=(str(path),))
    with pytest.raises(ValueError, match=message):
        run(GazeboConfig(**args, generate_only=True))


def test_route_replaces_the_scene_track():
    from robotlab.experiments.wheeled.wheeled_run import reference_track
    from robotlab.simulation.gazebo.world import load_scene

    settings = load_drone_config(robot="car").data
    scene = load_scene("worlds/race_circuit.json")
    assert reference_track(settings, scene).length_m > 1000
    settings["ground_vehicle"]["route"]["waypoints_m"] = [[0, 0], [30, 0]]
    route = reference_track(settings, scene)
    assert route.length_m == pytest.approx(30.0) and not route.closed
    assert reference_track(load_drone_config(robot="car").data, {}) is None


def test_randomized_world_uses_the_true_vehicle(tmp_path):
    import json
    import xml.etree.ElementTree as ET

    path = tmp_path / "layer.json"
    path.write_text(
        json.dumps(
            {
                "ground_vehicle": {
                    "randomization": {
                        "enabled": True,
                        "seed": 3,
                        "mass_fraction_std": 0.1,
                    }
                }
            }
        )
    )
    out = tmp_path / "out"
    run(
        GazeboConfig(
            robot="car", output=str(out), drone=(str(path),), generate_only=True
        )
    )
    mass = float(ET.parse(out / "world.sdf").find('.//model[@name="car"]//mass').text)
    resolved = json.loads((out / "drone-resolved.json").read_text())
    nominal = load_drone_config(robot="car").wheeled.mass_kg
    assert mass != pytest.approx(nominal)
    factor = mass / nominal
    assert json.dumps(resolved).count("mass_factor") >= 1
    assert 0.7 <= factor <= 1.3
