import numpy as np
import pytest

from robotlab.sensors.imu.faults import ImuFaults

PERIOD_NS = 5_000_000
HOVER = np.array([0.0, 0.0, 9.81])


def faults(*events, seed=0):
    return ImuFaults({"seed": seed, "events": list(events)})


def event(kind, instrument="accel", start_s=0.0, end_s=None, **parameters):
    return dict(
        type=kind, instrument=instrument, start_s=start_s, end_s=end_s, **parameters
    )


def run(model, samples, force=HOVER, rate=(0.1, 0.2, 0.3)):
    out = [model.apply(force, rate, i * PERIOD_NS) for i in range(samples)]
    return np.array([a for a, _ in out]), np.array([g for _, g in out])


def test_no_events_is_passthrough():
    model = faults()
    assert not model.enabled
    accel, gyro = run(model, 3)
    np.testing.assert_array_equal(accel, np.tile(HOVER, (3, 1)))


def test_bias_step_applies_only_inside_window_and_to_its_instrument():
    model = faults(
        event("bias_step", "gyro", start_s=0.01, end_s=0.02, value=[0.1, 0, 0])
    )
    accel, gyro = run(model, 6)
    np.testing.assert_allclose(gyro[:, 0], [0.1, 0.1, 0.2, 0.2, 0.1, 0.1])
    np.testing.assert_array_equal(accel, np.tile(HOVER, (6, 1)))
    summary = model.summary()["events"][0]
    assert summary["samples"] == 2 and summary["first_sim_time_ns"] == 10_000_000


def test_scale_and_noise():
    accel, _ = run(faults(event("scale", value=[1, 1, 1.1])), 2)
    np.testing.assert_allclose(accel[:, 2], 9.81 * 1.1)
    accel, _ = run(faults(event("noise", value=[0.5, 0, 0]), seed=1), 20000)
    assert accel[:, 0].std() == pytest.approx(0.5, rel=0.03)
    np.testing.assert_allclose(accel[:, 1:], np.tile(HOVER[1:], (20000, 1)))


def test_stuck_axes_hold_first_faulted_value():
    model = faults(
        event("stuck", "gyro", start_s=0.005, end_s=0.015, axes=[True, False, False])
    )
    readings = [
        model.apply(HOVER, [float(i), float(i), 0], i * PERIOD_NS)[1] for i in range(5)
    ]
    np.testing.assert_array_equal([r[0] for r in readings], [0, 1, 1, 3, 4])
    np.testing.assert_array_equal([r[1] for r in readings], [0, 1, 2, 3, 4])


def test_spike_probability_and_seeded_repeatability():
    spike = event("spikes", value=[0, 0, 50], probability=0.1)
    accel, _ = run(faults(spike, seed=4), 10000)
    spiked = np.abs(accel[:, 2] - 9.81) > 1e-12
    assert spiked.mean() == pytest.approx(0.1, abs=0.01)
    again, _ = run(faults(spike, seed=4), 10000)
    np.testing.assert_array_equal(accel, again)
    assert faults(spike, seed=4).summary()["events"][0]["spikes"] == 0


def test_events_compose_in_order_and_time_must_increase():
    model = faults(event("bias_step", value=[1, 0, 0]), event("scale", value=[2, 1, 1]))
    accel, _ = model.apply([0, 0, 0], [0, 0, 0], 0)
    assert accel[0] == 2  # (0 + 1) * 2
    with pytest.raises(ValueError, match="strictly increasing"):
        model.apply([0, 0, 0], [0, 0, 0], 0)
