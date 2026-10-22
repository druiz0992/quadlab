"""Per-run vehicle randomization and centre-of-mass offset."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.gazebo.world import build_world
from robotlab.simulation.randomization import randomize_config


def config_with(**randomization):
    config = load_drone_config()
    config.data["randomization"].update(randomization)
    return config


def test_disabled_returns_nominal_config():
    config = load_drone_config()
    true, draws = randomize_config(config)
    assert true is config and draws == {}


def test_draws_are_seeded_truncated_and_change_only_truth():
    settings = dict(
        enabled=True,
        seed=4,
        mass_fraction_std=0.05,
        inertia_fraction_std=0.1,
        center_of_mass_std_m=[0.01, 0.01, 0.0],
        rotor_thrust_fraction_std=0.03,
        rotor_moment_fraction_std=0.03,
        motor_time_constant_fraction_std=0.1,
        imu_mount_rotation_std_deg=1.0,
    )
    config = config_with(**settings)
    true, draws = randomize_config(config)
    again, draws_again = randomize_config(config_with(**settings))
    assert draws == draws_again
    assert config.data["vehicle"]["body_mass_kg"] == 1.5  # nominal untouched
    assert true.properties.body_mass == pytest.approx(1.5 * draws["mass_factor"])
    assert true.properties.thrust_coefficient == config.properties.thrust_coefficient
    assert np.all(np.abs(np.array(draws["rotor_thrust_factors"]) - 1) <= 0.09 + 1e-12)
    assert true.data["vehicle"]["center_of_mass_offset_m"][2] == 0.0
    assert true.data["sensors"]["imu"]["mount"]["rotation_deg"] != [0.0, 0.0, 0.0]
    other, _ = randomize_config(config_with(**dict(settings, seed=5)))
    assert other.properties.body_mass != true.properties.body_mass


def test_true_vehicle_reaches_the_sdf(tmp_path):
    config = load_drone_config()
    config.data["vehicle"]["center_of_mass_offset_m"] = [0.02, -0.01, 0.0]
    config.data["motors"]["rotor_thrust_scale"] = [1.0, 0.9, 1.0, 1.1]
    path = build_world(
        {"obstacles": []},
        tmp_path / "w.sdf",
        config=config,
        control_mode="rotor-thrust",
    )
    root = ET.parse(path).getroot()
    base = root.find(".//model[@name='drone']/link[@name='base_link']")
    assert base.findtext("inertial/pose") == "0.02 -0.01 0.0 0 0 0"
    constants = [
        float(p.findtext("motorConstant"))
        for p in root.iter("plugin")
        if p.get("name") == "gz::sim::systems::MulticopterMotorModel"
    ]
    np.testing.assert_allclose(constants, np.array([1.0, 0.9, 1.0, 1.1]) * 8.55e-6)


def test_validation(tmp_path):
    def load(layer):
        path = tmp_path / "x.json"
        path.write_text(json.dumps(layer))
        return load_drone_config(path)

    with pytest.raises(ValueError, match="half the arm"):
        load({"vehicle": {"center_of_mass_offset_m": [0.2, 0, 0]}})
    with pytest.raises(ValueError, match="one value per rotor"):
        load({"motors": {"rotor_thrust_scale": [1, 1, 1]}})
    with pytest.raises(ValueError, match="below 1/3"):
        load({"randomization": {"mass_fraction_std": 0.4}})


def wheeled_layer(tmp_path, randomization):
    path = tmp_path / "wheeled.json"
    path.write_text(json.dumps({"ground_vehicle": {"randomization": randomization}}))
    return path


def test_wheeled_randomization_disabled_returns_the_nominal_config():
    config = load_drone_config(robot="car")
    assert randomize_config(config) == (config, {})


def test_wheeled_draws_are_seeded_and_applied(tmp_path):
    settings = dict(
        enabled=True,
        seed=4,
        mass_fraction_std=0.05,
        inertia_fraction_std=0.1,
        center_of_mass_std_m=[0.05, 0.02, 0.0],
        tyre_friction_fraction_std=0.05,
        drive_torque_fraction_std=0.05,
    )
    config = load_drone_config(wheeled_layer(tmp_path, settings), robot="tractor")
    true, draws = randomize_config(config)
    again, same = randomize_config(config)
    assert draws == same and true.wheeled == again.wheeled
    nominal, real = config.wheeled, true.wheeled
    assert real.mass_kg == pytest.approx(nominal.mass_kg * draws["mass_factor"])
    second = (sum(nominal.inertia_kg_m2) - 2 * np.array(nominal.inertia_kg_m2)) / 2
    scaled = second * draws["second_moment_factors"]
    np.testing.assert_allclose(real.inertia_kg_m2, scaled.sum() - scaled)
    shift = np.array(draws["center_of_mass_shift_m"])
    assert shift[2] == 0.0 and np.all(np.abs(shift) <= 3 * np.array([0.05, 0.02, 0]))
    np.testing.assert_allclose(
        real.hub_positions_m(), nominal.hub_positions_m() - shift, atol=1e-12
    )
    hitch = true.data["ground_vehicle"]["implement"]["hitch_position_m"]
    nominal_hitch = config.data["ground_vehicle"]["implement"]["hitch_position_m"]
    np.testing.assert_allclose(hitch, np.array(nominal_hitch) - shift, atol=1e-12)
    factor = draws["tyre_friction_factor"]
    assert real.tyre.friction_lateral == pytest.approx(
        nominal.tyre.friction_lateral * factor
    )
    assert real.drive.max_torque_nm == pytest.approx(
        nominal.drive.max_torque_nm * draws["drive_torque_factor"]
    )
    # The nominal configuration is untouched (algorithms keep it).
    assert (
        config.wheeled
        == load_drone_config(wheeled_layer(tmp_path, settings), robot="tractor").wheeled
    )


def test_wheeled_inertia_draws_stay_physical(tmp_path):
    """Large inertia draws on every preset keep the triangle inequality
    (scaling the inertias directly broke it for the car: Gazebo refused)."""
    for kind in ("rover", "car", "tractor"):
        for seed in range(50):
            settings = dict(enabled=True, seed=seed, inertia_fraction_std=0.3)
            config = load_drone_config(wheeled_layer(tmp_path, settings), robot=kind)
            i = np.array(randomize_config(config)[0].wheeled.inertia_kg_m2)
            assert np.all(i > 0)
            assert np.all(i.sum() - i >= i - 1e-12)


def test_invalid_nominal_inertia_is_rejected(tmp_path):
    from robotlab.simulation.randomization import _scaled_inertia

    with pytest.raises(ValueError, match="triangle"):
        _scaled_inertia([1.0, 1.0, 3.0], np.ones(3))
