from types import SimpleNamespace as NS

import numpy as np
import pytest

from robotlab.core.measurements import GpsReading
from robotlab.sensors.gazebo import gps_errors_message
from robotlab.sensors.geodesy import gps_to_enu
from robotlab.sensors.gps_errors import GpsErrorModel, offset_fix
from robotlab.simulation.gazebo.config import load_drone_config

REFERENCE = GpsReading(40.4168, -3.7038, 650.0)
PERIOD_NS = 100_000_000


def settings(**overrides):
    result = load_drone_config().data["sensors"]["gps"]["error_model"]
    result.update({"enabled": True, **overrides})
    return result


def errors_enu(model, samples):
    return np.array(
        [
            gps_to_enu(model.apply(REFERENCE, i * PERIOD_NS), REFERENCE)
            for i in range(samples)
        ]
    )


def test_offset_fix_round_trips_through_enu():
    offset = np.array([3.0, -2.0, 1.5])
    np.testing.assert_allclose(
        gps_to_enu(offset_fix(REFERENCE, offset), REFERENCE), offset, atol=1e-5
    )


def test_disabled_is_passthrough():
    model = GpsErrorModel(settings(enabled=False, horizontal_std_m=5.0))
    assert model.apply(REFERENCE, 0) is REFERENCE
    message = NS(latitude_deg=40, longitude_deg=-3, altitude=650)
    assert gps_errors_message(message, model, 0) is message


def test_correlated_error_has_stationary_std_and_exponential_correlation():
    tau_s = 5.0
    model = GpsErrorModel(
        settings(
            horizontal_std_m=1.0, vertical_std_m=2.0, correlation_time_s=tau_s, seed=1
        )
    )
    errors = errors_enu(model, 60000)
    np.testing.assert_allclose(errors.std(axis=0), [1.0, 1.0, 2.0], rtol=0.1)
    expected = np.exp(-PERIOD_NS * 1e-9 / tau_s)
    assert np.corrcoef(errors[:-1, 0], errors[1:, 0])[0, 1] == pytest.approx(
        expected, abs=0.01
    )


def test_white_error_is_isotropic_in_metres_and_uncorrelated():
    model = GpsErrorModel(
        settings(white_horizontal_std_m=0.5, white_vertical_std_m=0.8, seed=2)
    )
    errors = errors_enu(model, 20000)
    np.testing.assert_allclose(errors.std(axis=0), [0.5, 0.5, 0.8], rtol=0.03)
    assert abs(np.corrcoef(errors[:-1, 0], errors[1:, 0])[0, 1]) < 0.03


def test_seeded_and_local_rng_and_ordered_time():
    config = settings(horizontal_std_m=1.0, seed=9)
    before = np.random.get_state()
    first, second = errors_enu(GpsErrorModel(config), 20), errors_enu(
        GpsErrorModel(config), 20
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    model = GpsErrorModel(config)
    model.apply(REFERENCE, 10)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.apply(REFERENCE, 10)


def test_message_adapter_moves_position_only():
    model = GpsErrorModel(settings(horizontal_std_m=2.0, vertical_std_m=2.0, seed=4))
    source = NS(
        latitude_deg=REFERENCE.latitude_deg,
        longitude_deg=REFERENCE.longitude_deg,
        altitude=REFERENCE.altitude_m,
        velocity_east=1.0,
    )
    result = gps_errors_message(source, model, 0)
    assert source.latitude_deg == REFERENCE.latitude_deg
    assert result.latitude_deg != source.latitude_deg
    assert result.altitude != source.altitude
    assert result.velocity_east == 1.0
