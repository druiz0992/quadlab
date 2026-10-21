"""Magnetometer errors, synthetic optical flow, 3D lidar and rangefinder decoding."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronelab.sensors.gazebo import (
    lidar_points,
    magnetometer_errors_message,
    rangefinder_distance,
)
from dronelab.sensors.magnetometer import MagnetometerErrorModel
from dronelab.sensors.optical_flow import OpticalFlowModel
from dronelab.simulation.gazebo.config import load_drone_config

FIELD_T = np.array([0.0, 25.6e-6, -37.2e-6])
LEVEL = [0.0, 0.0, 0.0, 1.0]


def sensors():
    return load_drone_config().data["sensors"]


# --- Magnetometer -----------------------------------------------------------


def mag_settings(**overrides):
    return dict(sensors()["magnetometer"]["error_model"], enabled=True, **overrides)


def test_magnetometer_disabled_is_passthrough():
    model = MagnetometerErrorModel(dict(mag_settings(), enabled=False))
    np.testing.assert_array_equal(model.apply(FIELD_T, 20.0), FIELD_T)
    message = NS(field_tesla=NS(x=0.0, y=1.0, z=2.0))
    assert magnetometer_errors_message(message, model, 0.0) is message


def test_hard_soft_iron_and_motor_interference_in_microtesla():
    soft = [[1.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.05, 1.0]]
    model = MagnetometerErrorModel(
        mag_settings(
            soft_iron=soft,
            hard_iron_ut=[5.0, -3.0, 2.0],
            motor_interference_ut_per_n=[0.2, 0.0, -0.1],
        )
    )
    measured = model.apply(FIELD_T, total_thrust_n=15.0)
    expected = np.array(soft) @ FIELD_T + np.array([5.0 + 3.0, -3.0, 2.0 - 1.5]) * 1e-6
    np.testing.assert_allclose(measured, expected, atol=1e-15)
    assert model.summary()["max_motor_interference_ut"] == pytest.approx(
        np.hypot(3.0, 1.5)
    )


def test_magnetometer_white_noise_std_and_seed():
    config = mag_settings(white_std_ut=[0.5, 0.5, 0.5], seed=4)
    first, second = MagnetometerErrorModel(config), MagnetometerErrorModel(config)
    samples = np.array([first.apply(FIELD_T) for _ in range(20000)]) - FIELD_T
    np.testing.assert_allclose(samples.std(axis=0), 0.5e-6, rtol=0.03)
    np.testing.assert_array_equal(second.apply(FIELD_T), samples[0] + FIELD_T)


# --- Optical flow -----------------------------------------------------------


def flow_model(**overrides):
    return OpticalFlowModel(dict(sensors()["optical_flow"], enabled=True, **overrides))


def test_level_forward_motion_gives_negative_z_flow_scaled_by_distance():
    model = flow_model()
    reading = model.sample(0, [0, 0, 2.07], [1.0, 0, 0], LEVEL, [0, 0, 0])
    # Sensor 0.07 m below the body: 2.0 m to the ground along the down axis.
    assert reading.valid and reading.distance_m == pytest.approx(2.0)
    # Body forward maps to sensor +z; flow_z = -v_z / d.
    np.testing.assert_allclose(reading.flow_rad_s, [0.0, -0.5], atol=1e-12)


def test_sideways_motion_and_pure_rotation():
    model = flow_model(rate_hz=50.0)
    sideways = model.sample(0, [0, 0, 2.07], [0, 1.0, 0], LEVEL, [0, 0, 0])
    np.testing.assert_allclose(sideways.flow_rad_s, [-0.5, 0.0], atol=1e-12)
    # Pitch rate with no translation: raw flow equals the body rate (sensor
    # y axis is body y), plus the lever-arm velocity of the offset mount.
    rotating = model.sample(20_000_000, [0, 0, 2.07], [0, 0, 0], LEVEL, [0, 0.3, 0])
    lever_v_x = 0.3 * -0.07  # (ω × r)_x for r = (0, 0, -0.07)
    np.testing.assert_allclose(
        rotating.flow_rad_s, [0.0, 0.3 - lever_v_x / 2.0], atol=1e-12
    )


def test_tilted_sensor_distance_uses_optical_axis():
    model = flow_model()
    pitch = Rotation.from_euler("y", 20, degrees=True).as_quat()
    reading = model.sample(0, [0, 0, 2.0], [0, 0, 0], pitch, [0, 0, 0])
    offset = Rotation.from_quat(pitch).apply([0, 0, -0.07])
    height = 2.0 + offset[2]
    axis = (
        Rotation.from_quat(pitch) * Rotation.from_euler("y", 90, degrees=True)
    ).apply([1, 0, 0])
    assert reading.distance_m == pytest.approx(-height / axis[2])


@pytest.mark.parametrize(
    "position, velocity, quaternion",
    [
        ([0, 0, 0.1], [0, 0, 0], LEVEL),  # 0.03 m below minimum distance
        ([0, 0, 20.0], [0, 0, 0], LEVEL),  # beyond maximum distance
        ([0, 0, 0.5], [5.0, 0, 0], LEVEL),  # 11 rad/s exceeds the 7 rad/s limit
        (
            [0, 0, 2.0],
            [0, 0, 0],
            Rotation.from_euler("y", -100, degrees=True).as_quat(),
        ),
    ],
)
def test_invalid_samples_report_nan(position, velocity, quaternion):
    model = flow_model()
    reading = model.sample(0, position, velocity, quaternion, [0, 0, 0])
    assert not reading.valid and np.isnan(reading.flow_rad_s).all()
    assert model.summary()["invalid"] == 1


def test_flow_fixed_rate_grid_and_seeded_noise():
    model = flow_model(rate_hz=50.0, noise_std_rad_s=0.1, seed=3)
    times = [i * 5_000_000 for i in range(4000)]  # 200 Hz offers
    readings = [
        model.sample(t, [0, 0, 2.07], [0, 0, 0], LEVEL, [0, 0, 0]) for t in times
    ]
    kept = [r for r in readings if r is not None]
    assert len(kept) == 1000  # 50 Hz of 20 s
    noise = np.array([r.flow_rad_s for r in kept])
    np.testing.assert_allclose(noise.std(axis=0), 0.1, rtol=0.1)
    again = flow_model(rate_hz=50.0, noise_std_rad_s=0.1, seed=3)
    first = again.sample(0, [0, 0, 2.07], [0, 0, 0], LEVEL, [0, 0, 0])
    np.testing.assert_array_equal(first.flow_rad_s, kept[0].flow_rad_s)


# --- Lidar and rangefinder decoding -----------------------------------------


def scan(ranges, *, count, vertical_count=1, vmin=0.0, vstep=0.0, rmin=0.1, rmax=30.0):
    return NS(
        ranges=ranges,
        angle_min=0.0,
        angle_step=np.pi / 2,
        count=count,
        vertical_count=vertical_count,
        vertical_angle_min=vmin,
        vertical_angle_step=vstep,
        range_min=rmin,
        range_max=rmax,
    )


def test_3d_lidar_rows_use_vertical_angles():
    # Two rows (-30°, +30°) of two columns (0°, 90°), each 2 m.
    message = scan(
        [2.0] * 4, count=2, vertical_count=2, vmin=-np.pi / 6, vstep=np.pi / 3
    )
    points = lidar_points(message, position_m=(0, 0, 0), rotation_rad=(0, 0, 0))
    c, s = 2 * np.cos(np.pi / 6), 2 * np.sin(np.pi / 6)
    np.testing.assert_allclose(
        points, [[c, 0, -s], [0, c, -s], [c, 0, s], [0, c, s]], atol=1e-12
    )


def test_rangefinder_distance_and_out_of_range():
    assert rangefinder_distance(scan([1.25], count=1)) == 1.25
    assert np.isnan(rangefinder_distance(scan([float("inf")], count=1)))
    assert np.isnan(rangefinder_distance(scan([0.05], count=1)))
