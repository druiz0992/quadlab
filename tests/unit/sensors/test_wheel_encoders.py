import numpy as np
import pytest

from robotlab.sensors.wheel_encoders import WheelEncoderModel

SETTINGS = dict(
    enabled=True,
    rate_hz=100.0,
    counts_per_revolution=100,
    steering_noise_std_rad=0.0,
    seed=0,
)


def test_speed_from_counts_is_quantized():
    model = WheelEncoderModel(SETTINGS, 2)
    first = model.read(0, np.zeros(2), np.full(2, 9.0), np.zeros(2))
    np.testing.assert_array_equal(first.wheel_speed_rad_s, 0.0)  # no interval yet
    angle = np.array([1.0, 0.03])
    reading = model.read(10_000_000, angle, np.zeros(2), np.zeros(2))
    step = 2 * np.pi / 100
    assert reading.counts == (15, 0)
    np.testing.assert_allclose(
        reading.wheel_speed_rad_s, np.array([15, 0]) * step / 0.01
    )


def test_negative_rotation_counts_down():
    model = WheelEncoderModel(SETTINGS, 1)
    model.read(0, np.zeros(1), np.zeros(1), np.zeros(1))
    reading = model.read(10_000_000, np.array([-0.5]), np.zeros(1), np.zeros(1))
    assert reading.counts[0] == -8
    assert reading.wheel_speed_rad_s[0] < 0


def test_steering_noise_is_seeded_and_disabled_passes_truth():
    noisy = dict(SETTINGS, steering_noise_std_rad=0.01)
    a = WheelEncoderModel(noisy, 3).read(0, np.zeros(3), np.zeros(3), np.zeros(3))
    b = WheelEncoderModel(noisy, 3).read(0, np.zeros(3), np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(a.steer_angle_rad, b.steer_angle_rad)
    assert np.std(a.steer_angle_rad) > 0
    off = WheelEncoderModel(dict(SETTINGS, enabled=False), 1)
    reading = off.read(0, np.array([3.0]), np.array([4.0]), np.array([0.2]))
    assert reading.wheel_speed_rad_s[0] == 4.0 and reading.steer_angle_rad[0] == 0.2
    assert off.summary()["readings"] == 1
