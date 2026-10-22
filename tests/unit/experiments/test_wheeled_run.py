import pytest

from robotlab.experiments.gazebo import GazeboConfig, run
from robotlab.experiments.wheeled_run import build_wheeled_controller
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
