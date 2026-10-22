"""ESC command resolution/update rate and telemetry."""

import numpy as np
import pytest

from robotlab.sensors.esc_telemetry import EscTelemetryModel
from robotlab.simulation.esc import EscModel

MS = 1_000_000


def esc(rate=None, steps=0, clock=lambda: 0):
    return EscModel(dict(update_rate_hz=rate, resolution_steps=steps), 1000.0, clock)


def test_ideal_esc_passes_through():
    model = esc()
    assert not model.enabled
    np.testing.assert_array_equal(
        model.accept(np.array([123.4, 0, 5, 999])), [123.4, 0, 5, 999]
    )


def test_resolution_quantizes_to_steps_of_max_speed():
    model = esc(steps=2000)  # 0.5 rad/s per step
    np.testing.assert_allclose(
        model.accept(np.array([100.26, 100.24, 0.1, 1000.2])),
        [100.5, 100.0, 0.0, 1000.0],
    )


def test_update_rate_holds_commands_between_ticks_but_not_stops():
    now = {"t": 0}
    model = esc(rate=100.0, clock=lambda: now["t"])
    results = []
    for t in range(0, 30, 4):  # new commands every 4 ms, ESC every 10 ms
        now["t"] = t * MS
        results.append(model.accept(np.full(4, 500.0 + t)) is not None)
    # Forwarded at 0, 12 (≥ 10), 24 (≥ 22) ms.
    assert results == [True, False, False, True, False, False, True, False]
    assert model.accept(np.full(4, 524.0)) is not None  # unchanged: no tick used
    now["t"] = 29 * MS
    assert model.accept(np.zeros(4)) is not None  # stop always passes
    assert model.summary() == dict(forwarded=5, held=5)


def test_telemetry_rescales_decimates_and_adds_noise():
    settings = dict(enabled=True, rate_hz=100.0, noise_std_rad_s=0.0, seed=0)
    model = EscTelemetryModel(settings, slowdown=10)
    reports = [
        model.sample(t * MS, [60.0, 60.0, -65.0, -65.0]) for t in range(0, 30, 4)
    ]
    kept = [r for r in reports if r is not None]
    assert len(kept) == 3  # 0, 12 (≥10), 24 (≥20) ms
    np.testing.assert_allclose(kept[0].rotor_speed_rad_s, [600, 600, 650, 650])
    noisy = EscTelemetryModel(dict(settings, noise_std_rad_s=5.0), slowdown=10)
    values = np.array(
        [noisy.sample(t * 10 * MS, [60.0] * 4).rotor_speed_rad_s for t in range(2000)]
    )
    assert values.std() == pytest.approx(5.0, rel=0.05)
