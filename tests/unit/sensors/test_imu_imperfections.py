"""IMU scale factor, cross-coupling, g-sensitivity, bias instability,
saturation, drops/outages, and timestamp jitter."""

import math

import numpy as np
import pytest

from robotlab.sensors.imu.imperfections import (
    ImuImperfections,
    ImuSaturation,
    ImuTiming,
)
from robotlab.simulation.gazebo.config import load_drone_config

MS = 1_000_000


def settings(**overrides):
    base = dict(
        load_drone_config().data["sensors"]["imu"]["imperfections"], enabled=True
    )
    base.update(overrides)
    return base


def test_disabled_is_exact_passthrough():
    model = ImuImperfections(
        load_drone_config().data["sensors"]["imu"]["imperfections"]
    )
    force, rate = model.apply([1, 2, 3], [0.1, 0.2, 0.3], 0)
    np.testing.assert_array_equal(force, [1, 2, 3])
    np.testing.assert_array_equal(rate, [0.1, 0.2, 0.3])


def test_scale_cross_coupling_and_g_sensitivity():
    coupling = [[0, 0.01, 0], [0, 0, 0], [0, 0, 0]]
    g_sens = [[0, 0, 0.001], [0, 0, 0], [0, 0, 0]]
    model = ImuImperfections(
        settings(
            accel_scale_factor=[0.01, 0, 0],
            accel_cross_coupling=coupling,
            gyro_scale_factor=[0, 0, -0.02],
            gyro_g_sensitivity_rad_s_per_mps2=g_sens,
        )
    )
    force, rate = model.apply([2.0, 1.0, 9.81], [0, 0, 1.0], 0)
    np.testing.assert_allclose(force, [2.02 + 0.01, 1.0, 9.81])
    np.testing.assert_allclose(rate, [0.00981, 0, 0.98])


def test_gauss_markov_bias_is_stationary_with_correlation_time():
    sigma, tau = 0.01, 5.0
    model = ImuImperfections(
        settings(
            accel_bias_instability_mps2=[sigma] * 3,
            bias_instability_correlation_s=tau,
            seed=3,
        )
    )
    biases = []
    for i in range(60000):
        force, _ = model.apply([0, 0, 0], [0, 0, 0], i * 10 * MS)
        biases.append(force[0])
    b = np.asarray(biases)
    assert b.std() == pytest.approx(sigma, rel=0.15)
    lag = 500  # 5 s at 100 Hz: autocorrelation e^{-1}
    corr = np.corrcoef(b[:-lag], b[lag:])[0, 1]
    assert corr == pytest.approx(math.exp(-1), abs=0.12)
    np.testing.assert_allclose(model.current_biases()[0][0], b[-1])


def test_saturation_clips_and_counts():
    sat = ImuSaturation(settings(accel_range_mps2=16 * 9.81, gyro_range_rad_s=34.9))
    force, rate = sat.apply([200, 0, 9.81], [0, 40, 0], 0)
    assert force[0] == pytest.approx(156.96) and rate[1] == 34.9
    assert sat.saturated_samples == 1
    assert not ImuSaturation(settings()).enabled


def test_drops_outages_and_bounded_jitter():
    timing = ImuTiming(
        settings(sample_drop_probability=0.1, outages_s=[[1.0, 1.5]]), rate_hz=200
    )
    lost = [timing.drop(i * 5 * MS) for i in range(400)]  # 0–2 s
    assert all(lost[200:300])  # the outage
    assert 0.05 < np.mean(lost[:200]) < 0.17
    jitter = ImuTiming(settings(timestamp_jitter_s=0.01), rate_hz=200)
    stamps = [jitter.stamp(i * 5 * MS) for i in range(1000)]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))  # never reverses
    offsets = np.array(stamps) - np.arange(1000) * 5 * MS
    assert np.abs(offsets).max() <= 0.45 * 5 * MS + 1


def test_cross_coupling_diagonal_must_be_zero(tmp_path):
    import json

    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {
                "sensors": {
                    "imu": {
                        "imperfections": {
                            "gyro_cross_coupling": [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
                        }
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="diagonal"):
        load_drone_config(path)
