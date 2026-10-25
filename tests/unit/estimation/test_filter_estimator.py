"""Generic filter estimator: converters, filter loading, the example filter."""

from typing import Any

import numpy as np
import pytest

from robotlab.core.algorithms import SensorDescription
from robotlab.core.measurements import (
    BaroReading,
    GpsReading,
    GpsStatus,
    ImuReading,
    MagnetometerReading,
    RangeReading,
    EncoderReading,
)
from robotlab.estimation.filter.estimator import FilterEstimator
from robotlab.estimation.filter.protocol import predict_kind, update_kinds
from robotlab.estimation.filter.measurements import MeasurementConverter
from robotlab.estimation.examples.imu_gps_filter import ImuGpsFilter
from robotlab.testing.plugins import check_estimator, synthetic_context
from robotlab.testing.scenarios import SCENARIOS, run_scenario, scenario_context

EXAMPLE = "robotlab.estimation.examples.imu_gps_filter:ImuGpsFilter"
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
    assert update_kinds(ImuGpsFilter({})) == ("baro", "gps")


def test_example_passes_the_contract_kit_and_the_hover_scenario() -> None:
    options = {"filter": EXAMPLE, "filter_options": {"initial_position_m": [0, 0, 1]}}
    report = check_estimator(FilterEstimator, options)
    error = report.final_position_error_m
    assert error is not None and error < 0.5
    scenario = run_scenario(FilterEstimator, options, SCENARIOS["hover"])
    assert scenario.position_rmse_m is not None and scenario.position_rmse_m < 0.5


def test_inputs_follow_enabled_sensors() -> None:
    options = {"filter": EXAMPLE}
    gps_only = FilterEstimator(synthetic_context(("imu", "gps")), options)
    assert gps_only.inputs == ("imu", "gps")
    both = FilterEstimator(synthetic_context(("imu", "gps", "baro")), options)
    assert both.inputs == ("imu", "baro", "gps")
    assert both.summary()["inputs"] == ["imu", "baro", "gps"]
    with pytest.raises(ValueError, match="needs the imu"):
        FilterEstimator(synthetic_context(("gps",)), options)
    with pytest.raises(ValueError, match="needs one of the baro, gps"):
        FilterEstimator(synthetic_context(("imu",)), options)


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, 'needs option "filter"'),
        ({"filter": EXAMPLE, "gain": 1}, "unknown options"),
        ({"filter": "nope.mod:Filter"}, "Cannot import"),
        ({"filter": "robotlab.estimation.examples.imu_gps_filter:Missing"}, "no class"),
        (
            {"filter": "robotlab.core.measurements:BaroReading"},
            "RecursiveFilter|BaroReading",
        ),
        ({"filter": EXAMPLE, "filter_options": {"bad": 1}}, "unknown options"),
    ],
)
def test_filter_loading_errors(options: dict, message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        FilterEstimator(synthetic_context(), options)


def test_debug_series_use_the_name_prefix() -> None:
    logged: list[str] = []
    context = synthetic_context(log=lambda name, *_, **__: logged.append(name))
    estimator = FilterEstimator(context, {"filter": EXAMPLE, "name": "mine"})
    assert estimator.node.estimates.stream_id == "algorithms/mine/state"
    report = check_estimator(
        lambda c, o: FilterEstimator(
            synthetic_context(log=lambda name, *_, **__: logged.append(name)), o
        ),
        {"filter": EXAMPLE, "name": "mine"},
    )
    error = report.final_position_error_m
    assert error is not None and error < 1.0
    assert "mine/gps" in logged


class _Driven:
    """Filter stub: configurable predict/update methods for protocol checks."""

    def state(self) -> Any:
        return None


def _stub(**methods: Any) -> Any:
    return type("Stub", (_Driven,), methods)()


def _noop(self: Any, *args: Any) -> None:
    return None


def test_predict_kind_finds_the_single_driving_stream() -> None:
    assert predict_kind(_stub(predict_wheel_encoders=_noop, update_gps=_noop)) == (
        "wheel_encoders"
    )
    assert predict_kind(ImuGpsFilter({})) == "imu"


@pytest.mark.parametrize(
    "methods, message",
    [
        (dict(update_gps=_noop), "no predict_<kind>"),
        (dict(predict=_noop, update_gps=_noop), r"rename predict\(.*predict_imu"),
        (dict(predict_imu=_noop, predict_wheel_encoders=_noop), "several driving"),
        (dict(predict_gps=_noop, update_gps=_noop), "both predicts and updates"),
    ],
)
def test_predict_kind_rejects_invalid_filters(methods: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        predict_kind(_stub(**methods))


def test_imu_input_rotated_into_body_and_body_origin_required() -> None:
    s = np.sqrt(0.5)
    yawed = {
        "imu": SensorDescription("imu", "drone/imu", 250.0, (0, 0, 0), (0, 0, s, s))
    }
    imu = MeasurementConverter(yawed, REFERENCE, {}, kinds=("imu",)).imu(
        ImuReading(np.array([1.0, 0.0, 9.81]), np.array([0.1, 0.0, 0.0]))
    )
    np.testing.assert_allclose(imu.specific_force_mps2, [0, 1, 9.81], atol=1e-12)
    np.testing.assert_allclose(imu.angular_rate_rad_s, [0, 0.1, 0], atol=1e-12)
    offset = {"imu": SensorDescription("imu", "drone/imu", 250.0, (0.1, 0, 0))}
    with pytest.raises(ValueError, match="body-origin IMU"):
        MeasurementConverter(offset, REFERENCE, {}, kinds=("imu",))
    # An offset IMU is fine for a filter that does not use it.
    MeasurementConverter(offset, REFERENCE, {}, kinds=("gps",))


def test_wheel_motion_from_encoders_and_raw_passthrough() -> None:
    wheeled = scenario_context(SCENARIOS["car_circle"], ("wheel_encoders",)).wheeled
    assert wheeled is not None
    convert = MeasurementConverter(
        {}, REFERENCE, {}, kinds=("wheel_encoders",), wheeled=wheeled
    )
    n = wheeled.wheel_count
    steer = np.zeros(n)
    steer[0] = np.pi / 2
    reading = EncoderReading(0, (0,) * n, np.full(n, 10.0), steer)
    motion = convert.convert("wheel_encoders", reading)
    np.testing.assert_allclose(
        motion.hub_positions_body_m, wheeled.hub_positions_m()[:, :2]
    )
    np.testing.assert_allclose(motion.rolling_direction_body[0], [0, 1], atol=1e-12)
    np.testing.assert_allclose(motion.rolling_direction_body[1], [1, 0])
    np.testing.assert_allclose(motion.rolling_speed_mps, 10.0 * wheeled.tyre.radius_m)
    with pytest.raises(ValueError, match="needs 4 wheels"):
        convert.wheel_encoders(EncoderReading(0, (0,), np.zeros(1), np.zeros(1)))
    with pytest.raises(ValueError, match="wheeled vehicle"):
        MeasurementConverter({}, REFERENCE, {}, kinds=("wheel_encoders",))
    raw = RangeReading(2.0)
    assert convert.convert("rangefinder", raw) is raw
