import numpy as np
import pytest

from robotlab.simulation.aerodynamics import DrydenTurbulence, ground_effect_factor
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.gazebo.wind import WindProfile

MS = 1_000_000


def test_ground_effect_cheeseman_bennett_and_clamp():
    radius = 0.1
    assert ground_effect_factor(0.1, radius) == pytest.approx(
        1 / (1 - (0.1 / 0.4) ** 2)
    )
    assert ground_effect_factor(0.0, radius) == pytest.approx(4 / 3)
    assert ground_effect_factor(0.02, radius) == pytest.approx(4 / 3)
    assert ground_effect_factor(2.6 * radius, radius) < 1.01


def turbulence_settings(**overrides):
    return dict(
        load_drone_config().data["wind"]["turbulence"], enabled=True, **overrides
    )


def test_turbulence_stationary_std_and_correlation_time():
    model = DrydenTurbulence(
        turbulence_settings(
            intensity_mps=[1.0, 0.5, 0.2], length_scale_m=[10, 10, 10], seed=3
        )
    )
    mean = np.array([5.0, 0.0, 0.0])  # tau = 10 m / 5 m/s = 2 s
    samples = []
    for i in range(200000):
        model.advance(i * 50 * MS, mean)
        samples.append(model.velocity_enu(mean))
    samples = np.array(samples)
    np.testing.assert_allclose(samples.std(axis=0), [1.0, 0.5, 0.2], rtol=0.1)
    assert np.corrcoef(samples[:-1, 0], samples[1:, 0])[0, 1] == pytest.approx(
        np.exp(-0.05 / 2.0), abs=0.01
    )


def test_turbulence_axes_follow_mean_wind_direction():
    model = DrydenTurbulence(turbulence_settings(intensity_mps=[1.0, 0.0, 0.0], seed=1))
    model.advance(0, np.array([0.0, 3.0, 0.0]))
    along = model.velocity_enu(np.array([0.0, 3.0, 0.0]))
    assert (
        along[0] == pytest.approx(0.0, abs=1e-12) and along[1] != 0
    )  # wind blows north


def test_turbulence_independent_of_caller_rate():
    mean = np.array([3.0, 0.0, 0.0])
    fast, slow = (
        DrydenTurbulence(turbulence_settings(intensity_mps=[1, 1, 1], seed=2))
        for _ in range(2)
    )
    for i in range(201):
        fast.advance(i * 10 * MS, mean)
    for i in range(0, 201, 7):
        slow.advance(i * 10 * MS, mean)
    slow.advance(2000 * MS, mean)
    np.testing.assert_array_equal(fast.velocity_enu(mean), slow.velocity_enu(mean))


def test_wind_profile_adds_turbulence_without_changing_random_gust_sequence():
    wind = load_drone_config().data["wind"]
    wind.update(
        enabled=True,
        velocity_enu_mps=[3, 0, 0],
        random_speed_std_mps=0.5,
        random_seed=4,
    )
    plain = WindProfile(wind)
    turbulent = WindProfile(
        dict(wind, turbulence=turbulence_settings(intensity_mps=[0.0, 0.0, 0.0]))
    )
    times = [i * 50 * MS for i in range(40)]
    plain_targets = [plain.target(t) for t in times]
    turbulent_targets = [turbulent.target(t) for t in times]
    # Zero-intensity turbulence publishes every 50 ms tick but adds nothing.
    assert all(target is not None for target in turbulent_targets)
    changed = [(t, v) for t, v in zip(times, plain_targets) if v is not None]
    for t, v in changed:
        np.testing.assert_allclose(turbulent_targets[times.index(t)], v)
