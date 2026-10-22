import numpy as np
import pytest

from robotlab.core.measurements import GpsReading
from robotlab.sensors.geodesy import gps_to_enu
from robotlab.sensors.multipath import GpsMultipath
from robotlab.simulation.gazebo.config import load_drone_config

REFERENCE = GpsReading(40.4168, -3.7038, 650.0)
PERIOD_NS = 100_000_000  # 10 Hz fixes


def settings(**overrides):
    result = load_drone_config().data["sensors"]["gps"]["multipath"]
    result.update({"enabled": True, **overrides})
    return result


def offsets(model, samples, position=(0, 0, 1)):
    return np.array(
        [
            gps_to_enu(model.apply(REFERENCE, i * PERIOD_NS, position), REFERENCE)
            for i in range(samples)
        ]
    )


def test_disabled_and_zero_rate_are_passthrough():
    assert GpsMultipath(settings(enabled=False)).apply(REFERENCE, 0) is REFERENCE
    np.testing.assert_allclose(offsets(GpsMultipath(settings()), 50), 0, atol=1e-6)


def test_episode_offset_is_constant_for_its_duration_then_clears():
    model = GpsMultipath(
        settings(
            onset_rate_hz=1000.0,
            duration_s=0.5,
            horizontal_std_m=3.0,
            vertical_std_m=3.0,
            seed=1,
        )
    )
    result = offsets(model, 6)
    # First fix has dt = 0, so no onset; the episode starts at fix 1 and lasts 0.5 s.
    np.testing.assert_allclose(result[0], 0, atol=1e-6)
    for i in range(2, 6):
        np.testing.assert_allclose(result[i], result[1], atol=1e-6)
    assert np.linalg.norm(result[1]) > 0.1
    summary = model.summary()
    assert summary["episodes"] == 1 and summary["affected_fixes"] == 5
    assert summary["episode_log"][0]["start_sim_time_ns"] == PERIOD_NS
    # At 0.6 s the first episode has expired and a new one starts immediately.
    model.apply(REFERENCE, 6 * PERIOD_NS, (0, 0, 1))
    assert model.summary()["episodes"] == 2


def test_onset_rate_matches_poisson_process():
    rate_hz, duration_s = 0.05, 1.0
    model = GpsMultipath(
        settings(
            onset_rate_hz=rate_hz, duration_s=duration_s, horizontal_std_m=1.0, seed=3
        )
    )
    seconds = 20000.0
    for i in range(int(seconds * 10)):
        model.apply(REFERENCE, i * PERIOD_NS)
    # Onsets only occur outside episodes, so expected count uses the free time.
    expected = seconds * rate_hz / (1 + rate_hz * duration_s)
    assert model.summary()["episodes"] == pytest.approx(expected, rel=0.1)


def test_zones_restrict_onset_and_leaving_ends_episode():
    zone = {"min_m": [0, 0, 0], "max_m": [10, 10, 10]}
    model = GpsMultipath(
        settings(
            onset_rate_hz=1000.0,
            duration_s=100.0,
            horizontal_std_m=3.0,
            zones=[zone],
            seed=2,
        )
    )
    outside = offsets(model, 5, position=(20, 0, 1))
    np.testing.assert_allclose(outside, 0, atol=1e-6)
    inside = gps_to_enu(model.apply(REFERENCE, 5 * PERIOD_NS, (5, 5, 1)), REFERENCE)
    assert np.linalg.norm(inside) > 0.1
    left = gps_to_enu(model.apply(REFERENCE, 6 * PERIOD_NS, (20, 5, 1)), REFERENCE)
    np.testing.assert_allclose(left, 0, atol=1e-6)


def test_seeded_and_local_rng():
    config = settings(onset_rate_hz=2.0, duration_s=0.3, horizontal_std_m=2.0, seed=7)
    before = np.random.get_state()
    np.testing.assert_array_equal(
        offsets(GpsMultipath(config), 100), offsets(GpsMultipath(config), 100)
    )
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
