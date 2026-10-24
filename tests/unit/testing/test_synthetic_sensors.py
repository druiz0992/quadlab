"""Synthetic scene and sensors (robotlab.testing.synthetic_*), noise-free."""

import dataclasses
import math

import numpy as np
import pytest

from robotlab.sensors.bus import STREAMS
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.testing.scenarios import (
    CAR_BODY_HEIGHT_M,
    SCENARIOS,
    SENSOR_KINDS,
    scenario_context,
    scenario_samples,
)
from robotlab.testing.trajectories import circling
from robotlab.testing.synthetic_scene import (
    BOX_RGB,
    DEFAULT_SCENE,
    SKY_RGB,
    SceneBox,
    SyntheticScene,
)

QUIET = {
    "sensors": {
        "baro": {"pressure_pa": {"noise_stddev": 0.0}},
        "magnetometer": {"field_ut": {"noise_stddev": [0.0, 0.0, 0.0]}},
        "rangefinder": {"range_m": {"noise_stddev": 0.0}},
        "lidar": {"range_m": {"noise_stddev": 0.0}},
        "camera": {"intensity": {"noise_stddev": 0.0}},
        "optical_flow": {"noise_std_rad_s": 0.0},
        "airspeed": {"noise_std_pa": 0.0},
        "battery_monitor": {"voltage_noise_std_v": 0.0, "current_noise_std_a": 0.0},
        "wheel_encoders": {"steering_noise_std_rad": 0.0},
        "gps": {"heading": {"noise_std_deg": 0.0}},
    },
    "motors": {"esc": {"telemetry": {"noise_std_rad_s": 0.0}}},
}


def quiet(name, duration_s=1.0, **changes):
    return dataclasses.replace(
        SCENARIOS[name],
        **changes,
        imu_noise=(0.0, 0.0),
        gps_noise_m=0.0,
        settings=QUIET,
        duration_s=duration_s,
    )


def first(name, kind, duration_s=1.0, **changes):
    samples = scenario_samples(quiet(name, duration_s, **changes), (kind,))
    return next(s for k, s in samples if k == kind)


def test_scene_hits_ground_boxes_and_nothing():
    scene = SyntheticScene((SceneBox((5, -1, 0), (6, 1, 2)),))
    hits = scene.cast(
        np.array([0.0, 0.0, 1.0]),
        np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
        50.0,
    )
    np.testing.assert_allclose(hits.distance_m, [5, 1, np.inf, np.inf])
    np.testing.assert_array_equal(hits.surface, [1, 0, -1, -1])
    np.testing.assert_allclose(hits.normal[:2], [[-1, 0, 0], [0, 0, 1]])
    assert scene.cast(np.zeros(3) + [0, 0, 1], [[1.0, 0, 0]], 4.0).surface[0] == -1


def test_scene_keeps_the_built_in_circles_clear():
    """Every pillar is farther than 13 m from the origin (circles ≤ 12 m)."""
    for box in DEFAULT_SCENE.boxes[4:]:
        nearest = np.clip(0.0, box.min_m[:2], box.max_m[:2])
        assert np.linalg.norm(nearest) > 13.0


def test_baro_magnetometer_and_rangefinder_at_hover():
    baro = first("hover", "baro").payload
    assert baro.altitude_m == pytest.approx(650.0 + 1.0, abs=1e-6)
    field = first("hover", "magnetometer").payload.field_ut
    # Facing east: body y is north; inclination 55.4° down, 45.1 µT.
    strength, inclination = 45.1, math.radians(55.4)
    np.testing.assert_allclose(
        field,
        [0, strength * math.cos(inclination), -strength * math.sin(inclination)],
        atol=1e-9,
    )
    # Down-looking 7 cm below the body origin at 1 m.
    assert first("hover", "rangefinder").payload.distance_m == pytest.approx(0.93)


def test_lidar_sees_the_walls_from_hover():
    scan = first("hover", "lidar").payload
    assert len(scan.points_m) == 360  # horizontal ring, every beam hits a wall
    np.testing.assert_allclose(scan.points_m[0], [20.0, 0.0, 0.2], atol=1e-9)
    np.testing.assert_allclose(np.abs(scan.points_m[90]), [0.0, 20.0, 0.2], atol=1e-9)
    assert np.all(scan.reflectivity == 0.6)


def test_camera_shows_sky_above_and_ground_below():
    image = first("hover", "camera").payload.image
    assert image.shape == (240, 320, 3)
    np.testing.assert_array_equal(image[0, 160], np.rint(SKY_RGB))
    # Facing east at the x = 20 m wall (box 2), centre pixel is the wall
    # (shaded, never brighter than its flat colour).
    assert np.all(image[110, 160] <= BOX_RGB[2]) and np.any(image[110, 160] > 0)
    bottom = image[-1].astype(float)
    assert not np.allclose(bottom, SKY_RGB, atol=5)


LEVEL = dict(trajectory=circling(5.0, 2.0))  # level circle, 2 m/s, from (5, 0)


def test_flow_and_airspeed_on_a_level_circle():
    flow = first("circle", "optical_flow", **LEVEL).payload
    assert flow.valid
    # Pitched 90°: forward 2 m/s at 0.93 m reads as -v_z_sensor / d ...
    assert abs(flow.flow_rad_s[1]) == pytest.approx(2.0 / 0.93, rel=1e-6)
    airspeed = first("circle", "airspeed", **LEVEL).payload
    assert airspeed.indicated_airspeed_mps == pytest.approx(
        2.0
        * math.sqrt(
            load_drone_config().data["aerodynamics"]["air_density_kg_m3"] / 1.225
        )
    )


def test_esc_speeds_carry_the_banked_thrust():
    """On the flown circle, thrust supplies gravity plus 0.8 m/s² inward."""
    esc = first("circle", "esc").payload.rotor_speed_rad_s
    vehicle = load_drone_config().properties
    thrust = vehicle.mass * math.hypot(9.81, 0.8)
    np.testing.assert_allclose(esc, math.sqrt(thrust / 4 / vehicle.thrust_coefficient))


def test_battery_monitor_and_gps_heading():
    battery = first("hover", "battery_monitor").payload
    assert battery.current_a == pytest.approx(10.0)
    assert battery.voltage_v == pytest.approx(4 * 4.2 - 10 * 0.02)
    heading = first("circle", "gps_heading", **LEVEL).payload
    # 0.1 s into the circle (yaw rate 0.4 rad/s, starting north).
    assert heading.heading_rad == pytest.approx(math.pi / 2 + 0.04)


def test_car_wheels_roll_and_front_wheels_steer_into_the_turn():
    samples = scenario_samples(quiet("car_circle"), ("wheel_encoders",))
    reading = [s.payload for k, s in samples][-1]
    front_left, front_right, rear_left, rear_right = reading.steer_angle_rad
    assert front_left > front_right > 0  # turning left: inner wheel steers more
    assert rear_left == rear_right == 0  # unsteered
    speeds = reading.wheel_speed_rad_s
    # Outer (right) wheels roll faster; quantized by 1024 counts per turn.
    assert speeds[1] > speeds[0] and speeds[3] > speeds[2]
    assert np.mean(speeds[2:]) == pytest.approx(5.0 / 0.3, rel=0.02)


def test_car_body_height_matches_the_nominal_car():
    car = load_drone_config(robot="car").wheeled
    hub_z = car.hub_positions_m()[:, 2].mean()
    assert CAR_BODY_HEIGHT_M == pytest.approx(car.tyre.radius_m - hub_z)


def test_selection_is_validated():
    hover, car = SCENARIOS["hover"], SCENARIOS["car_circle"]
    for scenario, sensors, message in (
        (hover, ("imu", "sonar"), "distinct kinds"),
        (hover, ("imu", "imu"), "distinct kinds"),
        (hover, (), "distinct kinds"),
        (hover, ("wheel_encoders",), "wheeled vehicle"),
        (car, ("esc",), "multirotor"),
    ):
        with pytest.raises(ValueError, match=message):
            scenario_samples(scenario, sensors)


def test_every_bus_stream_can_be_synthesized():
    assert set(SENSOR_KINDS) == set(STREAMS)


def test_more_sensors_leave_other_streams_unchanged():
    scenario = dataclasses.replace(SCENARIOS["circle"], duration_s=1.0)
    alone = [s for k, s in scenario_samples(scenario, ("imu",))]
    mixed = [
        s for k, s in scenario_samples(scenario, ("imu", "baro", "lidar")) if k == "imu"
    ]
    assert len(alone) == len(mixed)
    for a, b in zip(alone, mixed):
        np.testing.assert_array_equal(
            a.payload.specific_force_mps2, b.payload.specific_force_mps2
        )


def test_context_holds_exactly_the_selected_sensors_with_config_mounts():
    context = scenario_context(SCENARIOS["hover"], ("imu", "gps", "rangefinder"))
    assert sorted(context.sensors) == ["gps", "imu", "rangefinder"]
    rangefinder = context.sensors["rangefinder"]
    assert rangefinder.mount_position_m == (0.0, 0.0, -0.07)
    assert rangefinder.rate_hz == 20.0
    car = scenario_context(SCENARIOS["car_circle"], ("imu", "wheel_encoders"))
    assert car.wheeled is not None and car.wheeled.kind == "car"


def test_samples_follow_rates_priority_and_sequences():
    samples = scenario_samples(quiet("hover", 0.2), ("gps", "imu", "baro"))
    times = [(s.sim_time_ns, k) for k, s in samples]
    assert times[:2] == [(0, "imu"), (0, "baro")]  # IMU first at equal times
    assert times == sorted(times, key=lambda x: (x[0], list(STREAMS).index(x[1])))
    for kind, count in (("imu", 41), ("baro", 11), ("gps", 2)):
        stream = [s for k, s in samples if k == kind]
        assert [s.sequence for s in stream] == list(range(1, count + 1))
