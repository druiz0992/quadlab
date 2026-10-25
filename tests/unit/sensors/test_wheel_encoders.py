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


def test_readings_are_read_only():
    for enabled in (True, False):
        model = WheelEncoderModel({**SETTINGS, "enabled": enabled}, 4)
        reading = model.read(0, np.zeros(4), np.zeros(4), np.zeros(4))
        for array in (reading.wheel_speed_rad_s, reading.steer_angle_rad):
            with pytest.raises(ValueError):
                array[0] = 1.0


def test_encoder_readings_travel_on_the_sensor_bus():
    from robotlab.core.serialization import decode, encode
    from robotlab.sensors.bus import SensorBus
    from robotlab.simulation.gazebo.config import load_drone_config

    config = load_drone_config(robot="car")
    bus = SensorBus(config.data["sensor_timing"], ["wheel_encoders"])
    model = WheelEncoderModel(SETTINGS, 4)
    model.read(0, np.zeros(4), np.zeros(4), np.zeros(4))
    reading = model.read(10_000_000, np.full(4, 1.0), np.zeros(4), np.zeros(4))
    bus.submit("wheel_encoders", reading, 10_000_000)
    ((kind, sample),) = bus.advance(20_000_000)
    assert kind == "wheel_encoders" and sample.frame_id == "vehicle/wheels"
    assert sample.payload is reading and sample.sim_time_ns == 10_000_000
    restored = decode(encode(sample))
    np.testing.assert_array_equal(
        restored.payload.wheel_speed_rad_s, reading.wheel_speed_rad_s
    )


def test_wheeled_runs_carry_encoders_and_drones_do_not():
    from types import SimpleNamespace

    from robotlab.experiments.common.sensor_pipeline import SensorPipeline
    from robotlab.simulation.gazebo.config import load_drone_config

    def kinds(config):
        fake = SimpleNamespace(
            _sensors=config.data["sensors"],
            esc_telemetry=SimpleNamespace(enabled=False),
            gps_heading=SimpleNamespace(enabled=False),
        )
        return SensorPipeline.bus_kinds(fake)

    assert "wheel_encoders" in kinds(load_drone_config(robot="car"))
    assert "wheel_encoders" not in kinds(load_drone_config())


def test_recordings_from_before_the_move_still_decode() -> None:
    """Encoder readings recorded as robotlab.sensors.wheel_encoders types
    decode to the current core.measurements type."""
    from robotlab.core.measurements import EncoderReading
    from robotlab.core.serialization import decode, encode

    reading = EncoderReading(5, (1, 2), np.zeros(2), np.ones(2))
    old = encode(reading)
    old["__dataclass__"] = "robotlab.sensors.wheel_encoders:EncoderReading"
    restored = decode(old)
    assert isinstance(restored, EncoderReading) and restored.counts == (1, 2)
