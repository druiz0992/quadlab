from types import SimpleNamespace as NS

import numpy as np
import pytest

from dronelab.sensors.barometer import (
    BaroErrorModel,
    altitude_to_pressure,
    pressure_to_altitude,
)
from dronelab.sensors.gazebo import baro_datum_message, baro_errors_message
from dronelab.simulation.gazebo.config import load_drone_config

PERIOD_NS = 20_000_000  # 50 Hz


def settings(**overrides):
    result = load_drone_config().data["sensors"]["baro"]["error_model"]
    result.update({"enabled": True, **overrides})
    return result


def test_standard_atmosphere_reference_values_and_round_trip():
    assert altitude_to_pressure(0.0) == pytest.approx(101325.0)
    # ISA table: 1000 m -> 89874.6 Pa.
    assert altitude_to_pressure(1000.0) == pytest.approx(89874.6, abs=1.0)
    for altitude in (-100.0, 0.0, 650.0, 3000.0):
        assert pressure_to_altitude(altitude_to_pressure(altitude)) == pytest.approx(
            altitude, abs=1e-6
        )
    # Near sea level pressure falls about 12 Pa per metre.
    gradient = altitude_to_pressure(1.0) - altitude_to_pressure(0.0)
    assert gradient == pytest.approx(-12.0, abs=0.2)
    with pytest.raises(ValueError):
        pressure_to_altitude(0.0)


def errors_m(model, samples, altitude=651.0):
    pressure = altitude_to_pressure(altitude)
    return np.array(
        [
            pressure_to_altitude(model.apply(pressure, i * PERIOD_NS)) - altitude
            for i in range(samples)
        ]
    )


def test_disabled_is_passthrough():
    model = BaroErrorModel(settings(enabled=False, white_std_m=1.0))
    assert model.apply(95000.0, 0) == 95000.0
    message = NS(pressure=95000.0)
    assert baro_errors_message(message, model, 0) is message


def test_offset_and_white_noise_in_metres():
    errors = errors_m(
        BaroErrorModel(settings(offset_m=2.0, white_std_m=0.3, seed=1)), 20000
    )
    assert errors.mean() == pytest.approx(2.0, abs=0.01)
    assert errors.std() == pytest.approx(0.3, rel=0.03)


def test_drift_is_stationary_and_correlated():
    tau_s = 2.0
    errors = errors_m(
        BaroErrorModel(
            settings(drift_std_m=0.5, drift_correlation_time_s=tau_s, seed=2)
        ),
        100000,
    )
    assert errors.std() == pytest.approx(0.5, rel=0.1)
    assert np.corrcoef(errors[:-1], errors[1:])[0, 1] == pytest.approx(
        np.exp(-0.02 / tau_s), abs=0.01
    )


def test_seeded_local_rng_and_ordered_time():
    config = settings(drift_std_m=0.5, white_std_m=0.1, seed=9)
    before = np.random.get_state()
    np.testing.assert_array_equal(
        errors_m(BaroErrorModel(config), 50), errors_m(BaroErrorModel(config), 50)
    )
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    model = BaroErrorModel(config)
    model.apply(95000.0, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.apply(95000.0, 5)


def test_datum_shift_moves_sea_level_reading_to_world_elevation():
    source = NS(header="h", pressure=altitude_to_pressure(1.0), variance=0.0)
    shifted = baro_datum_message(source, 650.0)
    assert pressure_to_altitude(shifted.pressure) == pytest.approx(651.0, abs=1e-6)
    assert source.pressure == altitude_to_pressure(1.0)
    assert shifted.header == "h" and shifted.variance == 0.0


def test_prop_wash_and_airspeed_errors_follow_dynamic_pressure():
    model = BaroErrorModel(
        settings(prop_wash_coefficient=0.1, airspeed_coefficient=0.05)
    )
    pressure = altitude_to_pressure(651.0)
    measured = pressure_to_altitude(
        model.apply(pressure, 0, induced_velocity_mps=5.6, airspeed_mps=6.0)
    )
    expected = 0.1 * 5.6**2 / (2 * 9.80665) + 0.05 * 6.0**2 / (2 * 9.80665)
    assert measured - 651.0 == pytest.approx(expected, abs=1e-6)
    # No airflow, no error from these terms.
    assert pressure_to_altitude(model.apply(pressure, 1)) == pytest.approx(
        651.0, abs=1e-6
    )
