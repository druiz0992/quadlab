"""Generic inertial estimator: converters, filter loading, the template filter."""

from typing import Any

import numpy as np
import pytest

from robotlab.core.algorithms import SensorDescription
from robotlab.core.measurements import (
    BaroReading,
    GpsReading,
    GpsStatus,
    MagnetometerReading,
)
from robotlab.estimation.inertial.estimator import InertialEstimator
from robotlab.estimation.inertial.filter import update_kinds
from robotlab.estimation.inertial.measurements import MeasurementConverter
from robotlab.estimation.inertial.template import TemplateInertialFilter
from robotlab.testing.plugins import check_estimator, synthetic_context
from robotlab.testing.scenarios import SCENARIOS, run_scenario

TEMPLATE = "robotlab.estimation.inertial.template:TemplateInertialFilter"
REFERENCE = GpsReading(40.4168, -3.7038, 650.0)


def converter(**options: Any) -> MeasurementConverter:
    s = np.sqrt(0.5)
    sensors = {
        "gps": SensorDescription("gps", "wgs84", 10.0, (0.1, 0.0, 0.2)),
        # Magnetometer yawed 90 degrees: sensor x is body y.
        "magnetometer": SensorDescription(
            "magnetometer", "drone/magnetometer", 50.0, (0, 0, 0), (0, 0, s, s)
        ),
    }
    return MeasurementConverter(sensors, REFERENCE, options)


def test_gps_fix_in_enu_with_lever_arm_and_velocity() -> None:
    status = GpsStatus("3d", 12, 1.0, 1.0, 2.0, 0.3)
    reading = GpsReading(40.4168, -3.7038, 650.0, np.array([1.0, 2.0, 0.0]), status)
    fix = converter().gps(reading)
    np.testing.assert_allclose(fix.position_enu_m, 0, atol=1e-6)
    np.testing.assert_allclose(fix.lever_arm_body_m, [0.1, 0, 0.2])
    np.testing.assert_allclose(fix.covariance_m2, np.diag([1, 1, 4]))
    assert fix.velocity_enu_mps is not None
    np.testing.assert_allclose(fix.velocity_enu_mps, [1, 2, 0])
    assert fix.velocity_variance_m2_s2 == pytest.approx(0.09)
    configured = converter(use_receiver_gps_accuracy=False, gps_std_m=[1, 2, 3])
    np.testing.assert_allclose(
        configured.gps(REFERENCE).covariance_m2, np.diag([1, 4, 9])
    )


def test_magnetometer_rotated_into_body_and_calibrated_preferred() -> None:
    reading = MagnetometerReading(np.array([10.0, 0, 0]), np.array([20.0, 0, 0]))
    field = converter(magnetometer_std_ut=2.0).magnetometer(reading)
    np.testing.assert_allclose(field.field_body_ut, [0, 20, 0], atol=1e-9)
    np.testing.assert_allclose(field.covariance_ut2, np.eye(3) * 4)


def test_baro_altitude_relative_to_reference() -> None:
    baro = converter(baro_std_m=0.3).baro(BaroReading(95_000.0, 652.5))
    assert baro.altitude_m == pytest.approx(2.5) and baro.variance_m2 == pytest.approx(
        0.09
    )


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(gps_std_m=[1, 1]), "gps_std_m"),
        (dict(baro_std_m=0), "baro_std_m"),
        (dict(magnetometer_std_ut=-1), "magnetometer_std_ut"),
    ],
)
def test_converter_rejects_bad_options(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        converter(**options)


def test_update_kinds_found_by_name() -> None:
    assert update_kinds(TemplateInertialFilter({})) == ("baro", "gps")


def test_template_passes_the_contract_kit_and_the_hover_scenario() -> None:
    options = {"filter": TEMPLATE, "filter_options": {"initial_position_m": [0, 0, 1]}}
    report = check_estimator(InertialEstimator, options)
    error = report.final_position_error_m
    assert error is not None and error < 0.5
    scenario = run_scenario(InertialEstimator, options, SCENARIOS["hover"])
    assert scenario.position_rmse_m is not None and scenario.position_rmse_m < 0.5


def test_inputs_follow_enabled_sensors() -> None:
    options = {"filter": TEMPLATE}
    gps_only = InertialEstimator(synthetic_context(("imu", "gps")), options)
    assert gps_only.inputs == ("imu", "gps")
    both = InertialEstimator(synthetic_context(("imu", "gps", "baro")), options)
    assert both.inputs == ("imu", "baro", "gps")
    assert both.summary()["inputs"] == ["imu", "baro", "gps"]
    with pytest.raises(ValueError, match="needs the imu"):
        InertialEstimator(synthetic_context(("gps",)), options)
    with pytest.raises(ValueError, match="needs one of the baro, gps"):
        InertialEstimator(synthetic_context(("imu",)), options)


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, 'needs option "filter"'),
        ({"filter": TEMPLATE, "gain": 1}, "unknown options"),
        ({"filter": "nope.mod:Filter"}, "Cannot import"),
        ({"filter": "robotlab.estimation.inertial.template:Missing"}, "no class"),
        (
            {"filter": "robotlab.core.measurements:BaroReading"},
            "InertialFilter|BaroReading",
        ),
        ({"filter": TEMPLATE, "filter_options": {"bad": 1}}, "unknown options"),
    ],
)
def test_filter_loading_errors(options: dict, message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        InertialEstimator(synthetic_context(), options)


def test_debug_series_use_the_name_prefix() -> None:
    logged: list[str] = []
    context = synthetic_context(log=lambda name, *_, **__: logged.append(name))
    estimator = InertialEstimator(context, {"filter": TEMPLATE, "name": "mine"})
    assert estimator.node.estimates.stream_id == "algorithms/mine/state"
    report = check_estimator(
        lambda c, o: InertialEstimator(
            synthetic_context(log=lambda name, *_, **__: logged.append(name)), o
        ),
        {"filter": TEMPLATE, "name": "mine"},
    )
    error = report.final_position_error_m
    assert error is not None and error < 1.0
    assert "mine/gps" in logged
