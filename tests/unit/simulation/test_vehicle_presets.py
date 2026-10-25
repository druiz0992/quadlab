"""Vehicle presets: physical plausibility and gain derivation."""

import math

import pytest

from robotlab.simulation.gazebo.config import load_drone_config

# preset: (mass range kg, thrust-to-weight range, attitude ωn and ζ used for roll/pitch)
PRESETS = {
    "micro": ((0.025, 0.035), (1.7, 2.5), 6.0, 1.0),
    "racing_5in": ((0.5, 0.9), (7.0, 13.0), 20.0, 0.7),
    "freestyle_5in": ((0.65, 0.9), (6.0, 10.0), 18.0, 0.7),
    "10in_4s": ((1.3, 1.8), (2.0, 3.0), None, None),
    "heavy_lift": ((4.0, 8.0), (1.8, 2.5), 5.0, 0.7),
}


@pytest.mark.parametrize("name", PRESETS)
def test_preset_mass_thrust_and_hover_margin(name):
    (low, high), (tw_low, tw_high), _, _ = PRESETS[name]
    config = load_drone_config(f"configs/vehicles/{name}.json")
    p = config.properties
    assert low <= p.mass <= high
    thrust_to_weight = 4 * p.max_rotor_thrust / (p.mass * 9.81)
    assert tw_low <= thrust_to_weight <= tw_high
    # Hover leaves speed headroom for attitude control.
    hover_speed = math.sqrt(p.mass * 9.81 / 4 / p.thrust_coefficient)
    assert hover_speed < 0.8 * p.max_rotor_speed
    # Propellers must not overlap: diameter below the rotor spacing.
    assert p.propeller_diameter < 2 * config.data["vehicle"]["arm_offset_m"] * 1.01


@pytest.mark.parametrize("name", [n for n, v in PRESETS.items() if v[2]])
def test_attitude_gains_follow_inertia_and_stay_below_motor_bandwidth(name):
    _, _, wn, zeta = PRESETS[name]
    data = load_drone_config(f"configs/vehicles/{name}.json").data
    inertia = data["vehicle"]["body_inertia_kg_m2"][0]
    kp, kd = (
        data["controller"]["attitude_gain"][0],
        data["controller"]["angular_rate_gain"][0],
    )
    assert kp == pytest.approx(inertia * wn**2, rel=1e-3)
    assert kd == pytest.approx(2 * zeta * wn * inertia, rel=1e-3)
    # Closed-loop attitude bandwidth well below the motor lag bandwidth.
    assert wn < 0.5 / data["motors"]["time_constant_up_s"]


def test_racing_fpv_scenario_composes_airframe_camera_and_no_gps():
    config = load_drone_config("configs/scenarios/racing_fpv.json")
    camera = config.data["sensors"]["camera"]
    assert camera["enabled"] and camera["mount"]["rotation_deg"][1] == -30
    assert math.degrees(camera["horizontal_fov_rad"]) == pytest.approx(150, abs=0.1)
    assert camera["lens"]["type"] == "equidistant"  # native Gazebo fisheye
    assert not config.data["sensors"]["gps"]["enabled"]
    assert config.data["control"]["control_rate_hz"] == 250
    assert config.provenance["sensors.camera.enabled"].endswith("fpv_camera.json")


def test_battery_presets_are_rejected_in_velocity_mode(tmp_path):
    from robotlab.experiments.run import GazeboConfig, run

    with pytest.raises(ValueError, match="rotor-thrust"):
        run(
            GazeboConfig(
                drone="configs/vehicles/racing_5in.json",
                output=str(tmp_path / "out"),
                duration=1,
            )
        )
