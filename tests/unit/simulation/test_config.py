import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dronelab.control.manual import FlightCommand, ManualController
from dronelab.core.vehicle import DEFAULT_DRONE
from dronelab.simulation.gazebo.actuators import thrust_to_motor_speeds
from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.world import build_world
from dronelab.experiments.gazebo import GazeboConfig, run


def config_file(tmp_path, data):
    path = tmp_path / "drone.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "vibration",
    [
        {"enabled": 1},
        {"frequency_hz": 0},
        {"frequency_hz": float("nan")},
        {"accel_amplitude_mps2": [-1, 0, 0]},
        {"accel_amplitude_mps2": [1, 2]},
        {"random_std_mps2": [0, -1, 0]},
        {"phase_deg": [0, float("inf"), 0]},
        {"random_correlation_s": 0},
        {"clip_limit_mps2": 0},
        {"clip_limit_mps2": -1},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_accelerometer_vibration_configuration(tmp_path, vibration):
    with pytest.raises(ValueError):
        load_drone_config(
            config_file(tmp_path, {"sensors": {"imu": {"vibration": vibration}}})
        )


@pytest.mark.parametrize(
    "dropout",
    [
        {"probability": -0.1},
        {"probability": 1.1},
        {"probability": True},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": True},
        {"outages_s": [[2, 1]]},
        {"outages_s": [[1, 1]]},
        {"outages_s": [[-1, 2]]},
        {"outages_s": [[0, float("inf")]]},
        {"outages_s": [1, 2]},
        {"outages_s": "invalid"},
    ],
)
def test_invalid_gps_dropout_configuration(tmp_path, dropout):
    with pytest.raises(ValueError):
        load_drone_config(
            config_file(tmp_path, {"sensors": {"gps": {"dropout": dropout}}})
        )


@pytest.mark.parametrize(
    "wind",
    [
        {"enabled": 1},
        {"force_scale_per_s": 0},
        {"rise_time_s": 0},
        {"rise_time_s": 1e-8},
        {"gust_amplitude_fraction": -1},
        {"gust_amplitude_fraction": 1.1},
        {"gust_period_s": 0},
        {"velocity_enu_mps": [1, 2]},
        {"velocity_enu_mps": [0, float("nan"), 0]},
    ],
)
def test_invalid_wind_configuration(tmp_path, wind):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, {"wind": wind}))


@pytest.mark.parametrize(
    "wind",
    [
        {"random_seed": -1},
        {"random_seed": 1.2},
        {"random_seed": True},
        {"random_speed_std_mps": -1},
        {"random_direction_std_deg": -1},
        {"random_speed_std_mps": float("nan")},
        {"random_update_period_s": 0},
        {"random_update_period_s": 1e-12},
        {"schedule": "invalid"},
        {"schedule": [{"time_s": 1}]},
        {"schedule": [{"time_s": -1, "velocity_enu_mps": [1, 0, 0]}]},
        {"schedule": [{"time_s": 1, "velocity_enu_mps": [1, 0]}]},
        {"schedule": [{"time_s": 1, "velocity_enu_mps": [1, 0, 0], "oops": 1}]},
        {
            "schedule": [
                {"time_s": 2, "velocity_enu_mps": [1, 0, 0]},
                {"time_s": 1, "velocity_enu_mps": [1, 0, 0]},
            ]
        },
    ],
)
def test_invalid_dynamic_wind_configuration(tmp_path, wind):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, {"wind": wind}))


def test_native_wind_generation_and_disabled_default(tmp_path):
    default = ET.parse(build_world({"obstacles": []}, tmp_path / "default.sdf"))
    assert default.find("./world/wind") is None
    assert default.find('.//plugin[@name="gz::sim::systems::WindEffects"]') is None
    config = load_drone_config(
        config_file(
            tmp_path,
            {
                "wind": {
                    "enabled": True,
                    "velocity_enu_mps": [3, -2, 1],
                    "force_scale_per_s": 0.2,
                    "rise_time_s": 2,
                    "gust_amplitude_fraction": 0.25,
                    "gust_period_s": 8,
                }
            },
        )
    )
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "wind.sdf", config=config)
    )
    assert root.findtext("./world/wind/linear_velocity") == "3 -2 1"
    wind = root.find('./world/plugin[@name="gz::sim::systems::WindEffects"]')
    assert wind.get("filename") == "gz-sim-wind-effects-system"
    assert float(wind.findtext("force_approximation_scaling_factor")) == 0.2
    assert float(wind.findtext("horizontal/magnitude/sin/amplitude_percent")) == 0.25
    assert float(wind.findtext("horizontal/magnitude/sin/period")) == 8
    assert float(wind.findtext("vertical/time_for_rise")) == 2
    assert (
        root.findtext(
            './world/model[@name="drone"]/link[@name="base_link"]/enable_wind'
        )
        == "true"
    )


def test_defaults_and_example_files():
    assert load_drone_config().properties == DEFAULT_DRONE
    assert load_drone_config("configs/vehicles/quad_x.json").properties == DEFAULT_DRONE
    assert (
        load_drone_config("configs/sensors/noisy.json").data["sensors"]["imu"][
            "acceleration_mps2"
        ]["bias"][0]
        == 0.02
    )


def test_limits_mass_and_noise_reach_all_consumers(tmp_path):
    config = load_drone_config(
        config_file(
            tmp_path,
            {
                "vehicle": {"body_mass_kg": 2.0, "body_size_m": [0.6, 0.5, 0.2]},
                "motors": {"max_thrust_n": 7.0, "time_constant_up_s": 0.04},
                "sensors": {
                    "imu": {
                        "rate_hz": 100,
                        "acceleration_mps2": {
                            "bias": [0.1, -0.2, 0.3],
                            "noise_stddev": [0.01, 0.02, 0.03],
                        },
                    },
                    "gps": {"enabled": False},
                    "camera": {"enabled": True, "width": 640},
                    "lidar": {"enabled": False},
                },
            },
        )
    )
    root = ET.parse(
        build_world(
            {"obstacles": []},
            tmp_path / "world.sdf",
            config=config,
            control_mode="rotor-thrust",
        )
    )
    drone = root.find('./world/model[@name="drone"]')
    assert float(drone.findtext("./link/inertial/mass")) == 2
    assert drone.findtext("./link/collision/geometry/box/size") == "0.6 0.5 0.2"
    assert drone.find('./link/sensor[@name="gps"]') is None
    assert drone.find('./link/sensor[@name="lidar"]') is None
    assert drone.findtext('./link/sensor[@name="camera"]/camera/image/width') == "640"
    imu = drone.find('./link/sensor[@name="imu"]')
    assert float(imu.findtext("update_rate")) == 100
    assert float(imu.findtext("./imu/linear_acceleration/y/noise/mean")) == -0.2
    assert float(imu.findtext("./imu/linear_acceleration/z/noise/stddev")) == 0.03
    motor = next(p for p in drone.findall("plugin") if "MotorModel" in p.get("name"))
    assert float(motor.findtext("maxRotVelocity")) == pytest.approx(
        np.sqrt(7 / 8.55e-6)
    )
    assert float(motor.findtext("timeConstantUp")) == 0.04
    controller = ManualController(config.properties, config.data["controller"])
    forces = controller.update(
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
        (0, 0, 0, 1),
        np.zeros(3),
        FlightCommand(),
        0.02,
    )
    np.testing.assert_allclose(forces, 2.04 * 9.81 / 4)
    with pytest.raises(ValueError):
        thrust_to_motor_speeds([7.1] * 4, config.properties)


@pytest.mark.parametrize(
    "data",
    [
        {"weight": 2},
        {"schema_version": 2},
        {"vehicle": {"body_mass_kg": True}},
        {"vehicle": {"body_inertia_kg_m2": [1, 1, 3]}},
        {"motors": {"max_thrust_n": 10}},
        {"motors": {"max_thrust_n": 2}},
        {"vehicle": {"body_mass_kg": -1}},
        {"motors": {"thrust_coefficient_n_s2": float("nan")}},
        {"sensors": {"imu": {"acceleration_mps2": {"noise_stddev": [-1, 0, 0]}}}},
        {"sensors": {"gps": {"enabled": 1}}},
        {"sensors": {"camera": {"width": 12.5}}},
        {"sensors": {"lidar": {"min_range_m": 60}}},
        {"sensors": {"imu": {"rate_hz": 1000}}},
        {"control": {"command_lifetime_sim_s": 0.01}},
        {"control": {"publish_rate_hz": 1}},
        {"control": {"odometry_rate_hz": 5}},
        {"controller": {"angular_rate_gain": [1, 1]}},
    ],
)
def test_invalid_configs_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, data))


def test_generate_only_records_resolved_settings_and_cli_override(tmp_path):
    path = config_file(tmp_path, {"sensors": {"camera": {"enabled": False}}})
    output = tmp_path / "generated"
    run(
        GazeboConfig(
            drone=str(path),
            output=str(output),
            generate_only=True,
            control_mode="rotor-thrust",
            render_sensors=True,
        )
    )
    data = json.loads((output / "drone-resolved.json").read_text())
    assert data["sensors"]["camera"]["enabled"]
    assert data["sensors"]["lidar"]["enabled"]
    assert data["derived"]["total_mass_kg"] == pytest.approx(1.54)
    assert len(data["derived"]["config_sha256"]) == 64
    with pytest.raises(ValueError, match="generated world"):
        run(
            GazeboConfig(
                drone=str(path), world=str(output / "world.sdf"), generate_only=True
            )
        )


def test_rendered_sensor_noise_uses_legacy_sdf_type_element(tmp_path):
    config = load_drone_config("configs/sensors/noisy.json", render_sensors=True)
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "rendered.sdf", config=config)
    )
    for sensor in ("camera", "lidar"):
        noise = root.find(f'.//sensor[@name="{sensor}"]/{sensor}/noise')
        assert noise.get("type") is None
        assert noise.findtext("type") == "gaussian"
        assert float(noise.findtext("stddev")) > 0


def test_sensor_mounts_generate_body_relative_sdf_poses(tmp_path):
    mounts = {
        "imu": {"position_m": [0.1, -0.2, 0.03], "rotation_deg": [180, 0, 0]},
        "gps": {"position_m": [-0.1, 0, 0.2], "rotation_deg": [0, 0, 90]},
        "camera": {"position_m": [0.4, 0, 0.1], "rotation_deg": [0, 20, 0]},
        "lidar": {"position_m": [0, 0.1, 0.3], "rotation_deg": [10, 20, 30]},
    }
    config = load_drone_config(
        config_file(
            tmp_path,
            {
                "sensors": {
                    name: {"enabled": True, "mount": pose}
                    for name, pose in mounts.items()
                }
            },
        )
    )
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "mounted.sdf", config=config)
    )
    for name, mount in mounts.items():
        pose = list(
            map(float, root.findtext(f'.//sensor[@name="{name}"]/pose').split())
        )
        np.testing.assert_allclose(pose[:3], mount["position_m"])
        np.testing.assert_allclose(pose[3:], np.deg2rad(mount["rotation_deg"]))
    assert root.find('.//sensor[@name="contact"]/pose') is None


@pytest.mark.parametrize(
    "mount",
    [
        {"position_m": [1, 2]},
        {"rotation_deg": [0, True, 0]},
        {"position_m": [0, float("nan"), 0]},
        {"rotation_deg": [0, 0, float("inf")]},
        {"position_m": "0 0 0"},
        {"quaternion": [0, 0, 0, 1]},
        None,
    ],
)
def test_invalid_mount_rejected(tmp_path, mount):
    with pytest.raises(ValueError):
        load_drone_config(
            config_file(tmp_path, {"sensors": {"camera": {"mount": mount}}})
        )


def test_omitted_mount_preserves_existing_camera_and_lidar_offsets():
    from dronelab.simulation.gazebo.config import mount_pose

    config = load_drone_config()
    assert mount_pose(config.data["sensors"]["camera"]) == [0.3, 0, 0.1, 0, 0, 0]
    assert mount_pose(config.data["sensors"]["lidar"]) == [0, 0, 0.2, 0, 0, 0]


def test_gazebo_horizontal_gps_noise_metres_converted_to_native_degrees(tmp_path):
    config = load_drone_config(
        config_file(
            tmp_path,
            {
                "sensors": {
                    "gps": {
                        "horizontal_position_m": {"bias": 1.0, "noise_stddev": 2.0},
                        "vertical_position_m": {"noise_stddev": 3.0},
                    }
                }
            },
        )
    )
    root = ET.parse(build_world({"obstacles": []}, tmp_path / "gps.sdf", config=config))
    noise = root.find(".//navsat/position_sensing/horizontal/noise")
    assert float(noise.findtext("mean")) == pytest.approx(np.rad2deg(1 / 6378137.0))
    assert float(noise.findtext("stddev")) == pytest.approx(np.rad2deg(2 / 6378137.0))
    assert (
        float(root.findtext(".//navsat/position_sensing/vertical/noise/stddev")) == 3.0
    )


@pytest.mark.parametrize(
    "section",
    [
        {"sensors": {"imu": {"error_model": {"enabled": 1}}}},
        {"sensors": {"imu": {"error_model": {"seed": -1}}}},
        {
            "sensors": {
                "imu": {
                    "error_model": {"accel_noise_density_mps2_per_sqrt_hz": [-1, 0, 0]}
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "error_model": {"gyro_bias_random_walk_rad_s_per_sqrt_s": [0, 0]}
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "error_model": {"gyro_turn_on_bias_std_rad_s": [0, float("nan"), 0]}
                }
            }
        },
        {"sensors": {"imu": {"error_model": {"unknown": 1}}}},
        {"sensors": {"gps": {"error_model": {"horizontal_std_m": -0.1}}}},
        {"sensors": {"gps": {"error_model": {"correlation_time_s": 0}}}},
        {"sensors": {"gps": {"error_model": {"white_vertical_std_m": float("inf")}}}},
        {"sensors": {"gps": {"error_model": {"seed": 1.5}}}},
    ],
)
def test_invalid_sensor_error_model_configuration(tmp_path, section):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, section))


def test_sensor_error_example_matches_its_declared_ekf_assumptions():
    data = load_drone_config("configs/sensors/sensor_errors.json").data
    imu, ekf = data["sensors"]["imu"], data["ekf"]
    errors = imu["error_model"]
    assert errors["enabled"] and data["sensors"]["gps"]["error_model"]["enabled"]
    # Per-sample EKF noise is the density discretized at the IMU rate.
    rate = imu["rate_hz"] ** 0.5
    assert ekf["accel_std_mps2"] == pytest.approx(
        errors["accel_noise_density_mps2_per_sqrt_hz"][0] * rate, rel=0.01
    )
    assert ekf["gyro_std_rad_s"] == pytest.approx(
        errors["gyro_noise_density_rad_s_per_sqrt_hz"][0] * rate, rel=0.01
    )
    assert ekf["accel_bias_walk"] == errors["accel_bias_random_walk_mps2_per_sqrt_s"][0]
    assert ekf["gyro_bias_walk"] == errors["gyro_bias_random_walk_rad_s_per_sqrt_s"][0]
    # Defaults keep existing runs unchanged.
    defaults = load_drone_config().data["sensors"]
    assert not defaults["imu"]["error_model"]["enabled"]
    assert not defaults["gps"]["error_model"]["enabled"]


@pytest.mark.parametrize(
    "section",
    [
        {
            "sensors": {
                "gps": {
                    "dropout": {"zones": [{"min_m": [0, 0, 0], "max_m": [0, 1, 1]}]}
                }
            }
        },
        {"sensors": {"gps": {"dropout": {"zones": [{"min_m": [0, 0, 0]}]}}}},
        {"sensors": {"gps": {"dropout": {"zones": "box"}}}},
        {"sensors": {"gps": {"multipath": {"onset_rate_hz": -1}}}},
        {"sensors": {"gps": {"multipath": {"duration_s": 0}}}},
        {
            "sensors": {
                "gps": {"multipath": {"zones": [{"min_m": [0, 0], "max_m": [1, 1]}]}}
            }
        },
        {"sensors": {"imu": {"faults": {"events": [{"type": "explode"}]}}}},
        {
            "sensors": {
                "imu": {
                    "faults": {
                        "events": [
                            {
                                "type": "bias_step",
                                "instrument": "magnetometer",
                                "start_s": 0,
                                "end_s": None,
                                "value": [0, 0, 0],
                            }
                        ]
                    }
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "faults": {
                        "events": [
                            {
                                "type": "bias_step",
                                "instrument": "accel",
                                "start_s": 5,
                                "end_s": 5,
                                "value": [0, 0, 0],
                            }
                        ]
                    }
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "faults": {
                        "events": [
                            {
                                "type": "stuck",
                                "instrument": "gyro",
                                "start_s": 0,
                                "end_s": None,
                                "axes": [1, 0, 0],
                            }
                        ]
                    }
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "faults": {
                        "events": [
                            {
                                "type": "spikes",
                                "instrument": "gyro",
                                "start_s": 0,
                                "end_s": None,
                                "value": [1, 1, 1],
                                "probability": 2,
                            }
                        ]
                    }
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "faults": {
                        "events": [
                            {
                                "type": "noise",
                                "instrument": "gyro",
                                "start_s": 0,
                                "end_s": None,
                                "value": [-1, 0, 0],
                            }
                        ]
                    }
                }
            }
        },
        {
            "sensors": {
                "imu": {
                    "faults": {
                        "events": [
                            {
                                "type": "scale",
                                "instrument": "gyro",
                                "start_s": 0,
                                "end_s": None,
                                "value": [1, 1, 1],
                                "axes": [True] * 3,
                            }
                        ]
                    }
                }
            }
        },
    ],
)
def test_invalid_fault_configuration(tmp_path, section):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, section))


def test_fault_examples_load_and_defaults_inject_nothing():
    gps = load_drone_config("configs/faults/gps_faults.json").data["sensors"]["gps"]
    assert gps["dropout"]["zones"] and gps["multipath"]["enabled"]
    imu = load_drone_config("configs/faults/imu_faults.json").data["sensors"]["imu"]
    assert {event["type"] for event in imu["faults"]["events"]} == {
        "bias_step",
        "spikes",
        "stuck",
    }
    defaults = load_drone_config().data["sensors"]
    assert defaults["gps"]["dropout"]["zones"] == []
    assert not defaults["gps"]["multipath"]["enabled"]
    assert defaults["imu"]["faults"]["events"] == []


@pytest.mark.parametrize(
    "control",
    [
        {"control_rate_hz": 5},
        {"control_rate_hz": 100},  # above odometry rate
        {"odometry_rate_hz": 200, "control_rate_hz": 200},  # above publish rate
        {"position_rate_hz": 30},  # does not divide 50 Hz
        {"position_rate_hz": 100},
        {"state_latency_s": -0.01},
        {"command_latency_s": 0.6},
        {"publish_on_send": 1},
    ],
)
def test_invalid_control_timing(tmp_path, control):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, {"control": control}))


def test_valid_fast_control_timing_with_publish_on_send(tmp_path):
    control = {
        "odometry_rate_hz": 250,
        "control_rate_hz": 250,
        "position_rate_hz": 50,
        "publish_on_send": True,
        "state_latency_s": 0.01,
        "command_latency_s": 0.008,
    }
    assert (
        load_drone_config(config_file(tmp_path, {"control": control})).data["control"][
            "position_rate_hz"
        ]
        == 50
    )


def test_control_timing_example_and_defaults():
    control = load_drone_config("configs/control/control_timing.json").data["control"]
    assert control["control_rate_hz"] == 250 and control["publish_on_send"]
    defaults = load_drone_config().data["control"]
    assert defaults["control_rate_hz"] == defaults["odometry_rate_hz"] == 50
    assert defaults["state_latency_s"] == defaults["command_latency_s"] == 0
    assert defaults["position_rate_hz"] is None and not defaults["publish_on_send"]


@pytest.mark.parametrize(
    "baro",
    [
        {"enabled": 1},
        {"rate_hz": 0},
        {"rate_hz": 1000},  # above physics rate
        {"pressure_pa": {"noise_stddev": -1}},
        {"error_model": {"white_std_m": -0.1}},
        {"error_model": {"drift_std_m": float("nan")}},
        {"error_model": {"drift_correlation_time_s": 0}},
        {"error_model": {"offset_m": float("inf")}},
        {"error_model": {"seed": -1}},
    ],
)
def test_invalid_barometer_configuration(tmp_path, baro):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, {"sensors": {"baro": baro}}))


def test_barometer_sdf_only_when_enabled_and_example_loads(tmp_path):
    from dronelab.simulation.gazebo.world import world_elevation_m

    default = build_world({"obstacles": []}, tmp_path / "default.sdf")
    root = ET.parse(default).getroot()
    assert root.find(".//sensor[@name='baro']") is None
    assert world_elevation_m(default) == 650.0
    config = load_drone_config("configs/sensors/baro.json")
    assert config.data["sensors"]["baro"]["error_model"]["offset_m"] == 1.5
    path = build_world({"obstacles": []}, tmp_path / "baro.sdf", config=config)
    root = ET.parse(path).getroot()
    sensor = root.find(".//sensor[@name='baro']")
    assert sensor.get("type") == "air_pressure"
    assert sensor.findtext("topic") == "/dronelab/baro"
    assert sensor.findtext("update_rate") == "50"
    # Datum is applied in Python; Gazebo 8 ignores reference_altitude.
    assert sensor.find(".//reference_altitude") is None
    assert root.find(".//plugin[@name='gz::sim::systems::AirPressure']") is not None


@pytest.mark.parametrize(
    "data",
    [
        {"battery": {"cells": 0}},
        {"battery": {"cells": 4.0}},
        {"battery": {"initial_soc": 1.2}},
        {"battery": {"initial_soc": 0}},
        {"battery": {"powertrain_efficiency": 1.5}},
        {"battery": {"internal_resistance_ohm": 0}},
        {"battery": {"avionics_power_w": -1}},
        {"vehicle": {"propeller_diameter_m": 0}},
        {"vehicle": {"rotor_height_m": -0.1}},
        {"wind": {"turbulence": {"enabled": True}}},  # requires wind.enabled
        {"wind": {"enabled": True, "turbulence": {"intensity_mps": [-1, 0, 0]}}},
        {"wind": {"enabled": True, "turbulence": {"length_scale_m": [0, 1, 1]}}},
        {"wind": {"enabled": True, "turbulence": {"update_period_s": 0.0001}}},
        {"wind": {"still_air_drag": 1}},
    ],
)
def test_invalid_power_and_aero_configuration(tmp_path, data):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, data))


def test_still_air_drag_generates_wind_effects_with_zero_wind(tmp_path):
    config = load_drone_config(
        config_file(tmp_path, {"wind": {"still_air_drag": True}})
    )
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "w.sdf", config=config)
    ).getroot()
    assert root.find(".//plugin[@name='gz::sim::systems::WindEffects']") is not None
    assert root.findtext("./world/wind/linear_velocity") == "0.0 0.0 0.0"
    assert root.findtext(".//link[@name='base_link']/enable_wind") == "true"
    default = ET.parse(build_world({"obstacles": []}, tmp_path / "d.sdf")).getroot()
    assert default.find(".//plugin[@name='gz::sim::systems::WindEffects']") is None


def test_battery_and_ground_effect_rejected_outside_rotor_mode(tmp_path):
    path = config_file(tmp_path, {"battery": {"enabled": True}})
    with pytest.raises(ValueError, match="rotor-thrust"):
        run(GazeboConfig(drone=str(path), output=str(tmp_path / "out"), duration=1))


def test_physics_examples_load():
    for path in (
        "configs/vehicles/battery_low.json",
        "configs/environment/turbulence.json",
        "configs/scenarios/ground_effect_hover.json",
        "configs/vehicles/10in_4s.json",
    ):
        load_drone_config(path)
    assert (
        load_drone_config("configs/vehicles/quad_x.json").data
        == load_drone_config().data
    )


@pytest.mark.parametrize(
    "sensors_",
    [
        {"lidar": {"vertical_samples": 0}},
        {"lidar": {"vertical_samples": 16}},  # needs distinct min/max angles
        {
            "lidar": {
                "vertical_samples": 16,
                "vertical_min_angle_deg": 10,
                "vertical_max_angle_deg": -10,
            }
        },
        {"lidar": {"vertical_min_angle_deg": -95}},
        {"rangefinder": {"min_range_m": 5, "max_range_m": 1}},
        {"optical_flow": {"min_distance_m": 9}},
        {"optical_flow": {"rate_hz": 100}},  # above 50 Hz odometry
        {"optical_flow": {"noise_std_rad_s": -0.1}},
        {"magnetometer": {"error_model": {"soft_iron": [[1, 0], [0, 1]]}}},
        {
            "magnetometer": {
                "error_model": {"soft_iron": [[1, 0, 0], [0, 1, 0], [0, 0, "x"]]}
            }
        },
        {"magnetometer": {"error_model": {"white_std_ut": [-1, 0, 0]}}},
    ],
)
def test_invalid_new_sensor_configuration(tmp_path, sensors_):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, {"sensors": sensors_}))


def test_new_sensors_generate_sdf_only_when_enabled(tmp_path):
    default = ET.parse(build_world({"obstacles": []}, tmp_path / "d.sdf")).getroot()
    for name in ("magnetometer", "rangefinder"):
        assert default.find(f".//sensor[@name='{name}']") is None
    plugin = default.find(".//plugin[@name='gz::sim::systems::Magnetometer']")
    assert plugin.findtext("use_units_gauss") == "false"
    assert plugin.findtext("use_earth_frame_ned") == "false"
    data = {
        "sensors": {
            "magnetometer": {
                "enabled": True,
                "field_ut": {"noise_stddev": [0.5, 0.5, 0.5]},
            },
            "rangefinder": {"enabled": True},
            "lidar": {
                "enabled": True,
                "vertical_samples": 16,
                "vertical_min_angle_deg": -15,
                "vertical_max_angle_deg": 15,
            },
        }
    }
    config = load_drone_config(config_file(tmp_path, data))
    assert config.render_sensors
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "s.sdf", config=config)
    ).getroot()
    mag = root.find(".//sensor[@name='magnetometer']")
    assert mag.get("type") == "magnetometer"
    assert float(mag.findtext("./magnetometer/x/noise/stddev")) == pytest.approx(0.5e-6)
    ranger = root.find(".//sensor[@name='rangefinder']")
    assert (
        ranger.get("type") == "gpu_lidar"
        and ranger.findtext("topic") == "/dronelab/rangefinder"
    )
    assert ranger.findtext(".//horizontal/samples") == "1"
    assert ranger.findtext("pose") == "0.0 0.0 -0.07 0.0 1.5707963267948966 0.0"
    vertical = root.find(".//sensor[@name='lidar']//vertical")
    assert vertical.findtext("samples") == "16"
    assert float(vertical.findtext("min_angle")) == pytest.approx(-np.radians(15))


def test_new_sensor_examples_load():
    for path in (
        "configs/sensors/magnetometer.json",
        "configs/sensors/downward.json",
        "configs/sensors/lidar_3d.json",
    ):
        load_drone_config(path)


@pytest.mark.parametrize(
    "data",
    [
        {"simulation": {"real_time_factor": 0}},  # unlimited needs lockstep
        {"simulation": {"lockstep": True, "step_s": 0.003}},  # 20 ms not a multiple
        {"simulation": {"lockstep": 1}},
        {"simulation": {"real_time_factor": -1}},
    ],
)
def test_invalid_lockstep_configuration(tmp_path, data):
    with pytest.raises(ValueError):
        load_drone_config(config_file(tmp_path, data))


def test_lockstep_example_and_world_real_time_factor(tmp_path):
    config = load_drone_config("configs/control/lockstep.json")
    assert config.data["simulation"]["lockstep"]
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "w.sdf", config=config)
    ).getroot()
    assert root.findtext(".//physics/real_time_factor") == "0"


@pytest.mark.parametrize(
    "kwargs, drone, message",
    [
        (dict(keyboard=True), {"simulation": {"lockstep": True}}, "keyboard"),
        (
            dict(control_mode="rotor-thrust"),
            {"simulation": {"lockstep": True}},
            "publish_on_send",
        ),
    ],
)
def test_lockstep_runner_restrictions(tmp_path, kwargs, drone, message):
    path = config_file(tmp_path, drone)
    with pytest.raises(ValueError, match=message):
        run(
            GazeboConfig(
                drone=str(path), output=str(tmp_path / "out"), duration=1, **kwargs
            )
        )


def test_health_thresholds_validate_in_config(tmp_path):
    layer = tmp_path / "health.json"
    layer.write_text('{"health": {"saturated_fraction": 2.0}}')
    with pytest.raises(ValueError, match="saturated_fraction"):
        load_drone_config(layer)
    layer.write_text('{"health": {"hard_impact_speed_mps": 0}}')
    with pytest.raises(ValueError, match="positive"):
        load_drone_config(layer)
    layer.write_text('{"health": {"flip_tilt_deg": 120, "max_dropped_messages": 5}}')
    assert load_drone_config(layer).data["health"]["flip_tilt_deg"] == 120


def test_sample_timing_warnings_name_unaligned_gazebo_streams():
    from dronelab.simulation.gazebo.config import sample_timing_warnings

    data = load_drone_config().data
    # Default: 200 Hz IMU on 2 ms steps; everything else aligned.
    (warning,) = sample_timing_warnings(data)
    assert warning.startswith("sensors.imu.rate_hz = 200 Hz")
    assert "between 4 and 6 ms" in warning
    data["sensors"]["imu"]["rate_hz"] = 250
    assert sample_timing_warnings(data) == []
    data["control"]["odometry_rate_hz"] = 300  # 3.33 ms
    data["sensors"]["baro"].update(enabled=False, rate_hz=30)  # disabled: ignored
    (warning,) = sample_timing_warnings(data)
    assert warning.startswith("control.odometry_rate_hz = 300 Hz")


def test_site_air_density_default_and_overrides(tmp_path):
    from dronelab.simulation.aerodynamics import site_air_density_kg_m3

    # Standard sea level gives the classic value; the 650 m site at 25 °C less.
    assert site_air_density_kg_m3(0.0, 15.0) == pytest.approx(1.225, abs=5e-4)
    assert site_air_density_kg_m3(650.0, 25.0) == pytest.approx(1.0953, abs=5e-4)
    with pytest.raises(ValueError):
        site_air_density_kg_m3(0.0, -300.0)

    aero = load_drone_config().data["aerodynamics"]
    assert aero["air_density_kg_m3"] == pytest.approx(1.0953, abs=5e-4)
    sea = load_drone_config(site_elevation_m=0.0).data["aerodynamics"]
    assert sea["air_density_kg_m3"] == pytest.approx(1.1839, abs=5e-4)  # 25 °C

    def layer(value):
        path = tmp_path / f"{value}.json"
        path.write_text(json.dumps({"aerodynamics": {"air_density_kg_m3": value}}))
        return path

    explicit = load_drone_config(layer(1.225)).data["aerodynamics"]
    assert explicit["air_density_kg_m3"] == 1.225  # migration: old default
    with pytest.raises(ValueError):
        load_drone_config(layer("sea"))
    with pytest.raises(ValueError):
        load_drone_config(layer(-1.0))
