"""Per-run vehicle randomization and centre-of-mass offset."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.world import build_world
from dronelab.simulation.randomization import randomize_config


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
