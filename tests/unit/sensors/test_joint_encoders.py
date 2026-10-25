"""Joint encoders: quantization, velocity resolution, noise, pass-through."""

import numpy as np
import pytest

from robotlab.sensors.joint_encoders import JointEncoderModel

NAMES = ("hip", "knee")


def settings(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        enabled=True,
        rate_hz=50.0,
        seed=3,
        counts_per_revolution=4096,
        velocity_resolution_rad_s=0.1,
        position_noise_std_rad=0.0,
    )
    base.update(overrides)
    return base


def test_positions_round_to_counts_and_velocity_to_resolution() -> None:
    model = JointEncoderModel(settings(), NAMES)
    lsb = 2 * np.pi / 4096
    reading = model.sample(20_000_000, [10.4 * lsb, -3.6 * lsb], [0.26, -0.04])
    assert reading.sim_time_ns == 20_000_000
    assert reading.names == NAMES
    assert reading.counts == (10, -4)
    np.testing.assert_allclose(reading.position_rad, [10 * lsb, -4 * lsb])
    np.testing.assert_allclose(reading.velocity_rad_s, [0.3, 0.0], atol=1e-12)
    assert not reading.position_rad.flags.writeable
    assert model.summary()["readings"] == 1


def test_noise_is_seeded_and_flickers_counts() -> None:
    lsb = 2 * np.pi / 4096
    first = JointEncoderModel(settings(position_noise_std_rad=lsb), NAMES)
    second = JointEncoderModel(settings(position_noise_std_rad=lsb), NAMES)
    counts = [first.sample(i, [0.0, 0.0], [0.0, 0.0]).counts for i in range(50)]
    assert counts == [
        second.sample(i, [0.0, 0.0], [0.0, 0.0]).counts for i in range(50)
    ]
    assert len({c[0] for c in counts}) > 1


def test_disabled_passes_truth_through() -> None:
    model = JointEncoderModel(settings(enabled=False), NAMES)
    reading = model.sample(0, [0.123456, -1.0], [0.0101, 2.0])
    np.testing.assert_array_equal(reading.position_rad, [0.123456, -1.0])
    np.testing.assert_array_equal(reading.velocity_rad_s, [0.0101, 2.0])
    assert reading.counts == (0, 0)


def test_zero_velocity_resolution_keeps_velocity() -> None:
    model = JointEncoderModel(settings(velocity_resolution_rad_s=0.0), NAMES)
    np.testing.assert_array_equal(
        model.sample(0, [0, 0], [0.0123, 1.0]).velocity_rad_s, [0.0123, 1.0]
    )


def test_wrong_joint_count_raises() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        JointEncoderModel(settings(), NAMES).sample(0, [0.0], [0.0])
