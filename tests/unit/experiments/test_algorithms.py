"""Plug-in loading, context, runner dispatch, and estimate-driven state."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronelab.control.interface import ControlState, RotorController
from dronelab.control.manual import FlightCommand
from dronelab.control.references import ReferencePoint
from dronelab.core.measurements import EstimatedState, GpsReading, ImuReading, Sample
from dronelab.estimation.interface import StateEstimator
from dronelab.experiments.algorithms import (
    AlgorithmRunner,
    EstimateStateSource,
    StaleEstimate,
    build_context,
    build_controller,
    build_estimator,
    load_class,
)
from dronelab.sensors.bus import SensorBus
from dronelab.simulation.gazebo.config import load_drone_config

MS = 1_000_000
ORIGIN = GpsReading(40.4168, -3.7038, 650.0)


def settings(**algorithms):
    data = load_drone_config().data
    data["algorithms"].update(algorithms)
    data["sensor_timing"]["reorder_window_s"] = 0.0
    return data


def context(data):
    return build_context(data, load_drone_config().properties, ORIGIN, ("imu", "gps"))


def test_load_class_errors_are_explicit():
    with pytest.raises(ValueError, match="package.module:Class"):
        load_class("nonsense", {})
    with pytest.raises(ValueError, match="Cannot import"):
        load_class("no_such_module_xyz:Thing", {})
    with pytest.raises(ValueError, match="no class"):
        load_class("dronelab.control.manual:Missing", {})


def test_context_has_mounts_and_no_error_models():
    data = settings()
    data["sensors"]["imu"]["mount"]["rotation_deg"] = [0, 0, 90]
    ctx = context(data)
    imu = ctx.sensors["imu"]
    np.testing.assert_allclose(
        imu.mount_quaternion_xyzw, Rotation.from_euler("z", 90, degrees=True).as_quat()
    )
    assert imu.rate_hz == data["sensors"]["imu"]["rate_hz"]
    assert not hasattr(ctx, "settings")


def test_builtin_controller_follows_protocol_and_reference():
    data = settings()
    controller = build_controller(data, context(data))
    assert isinstance(controller, RotorController)
    state = ControlState(
        0, np.array([0, 0, 1.0]), np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3)
    )
    reference = ReferencePoint(np.array([0.5, 0, 1.0]), np.zeros(3), np.zeros(3), 0.3)
    controller.update_position(state, reference, 0.02)
    np.testing.assert_allclose(controller.setpoint_m, [0.5, 0, 1.0])
    thrust = controller.update_attitude(state)
    assert thrust.shape == (4,) and np.all(thrust > 0)
    controller.update_position(state, FlightCommand(), 0.02)


def test_builtin_ekf_estimator_runs_on_bus_samples_and_reports_nis():
    data = settings(estimator="gps_imu_ekf")
    data["ekf"]["initial_position_m"] = [0, 0, 1]
    estimator = build_estimator(data, context(data))
    assert isinstance(estimator, StateEstimator)
    bus = SensorBus(data["sensor_timing"], ("imu", "gps"))
    runner = AlgorithmRunner(bus, estimator)
    estimates, innovations = [], []
    for i in range(1, 101):
        bus.submit("imu", ImuReading([0, 0, 9.81], [0, 0, 0]), i * 4 * MS)
        if i % 25 == 0:
            fix = GpsReading(ORIGIN.latitude_deg, ORIGIN.longitude_deg, 651.0)
            bus.submit("gps", fix, i * 4 * MS)
        _, new, latest_innovations = runner.advance(i * 4 * MS)
        estimates += new
        innovations += latest_innovations
    assert [n.name for n in innovations] == ["gps"] * 4
    assert all(n.dof == 3 and n.accepted for n in innovations)
    assert len(estimates) >= 100  # one per IMU sample plus GPS updates
    assert estimator.summary()["gps_updates"] == 4
    np.testing.assert_allclose(
        runner.latest_estimate.payload.position_m, [0, 0, 1], atol=0.05
    )


def test_estimator_requires_its_sensors():
    data = settings(estimator="gps_imu_ekf")
    ctx = build_context(data, load_drone_config().properties, ORIGIN, ("imu",))
    with pytest.raises(ValueError, match="gps"):
        build_estimator(data, ctx)


def estimate_sample(t_ns, gyro_bias=(0, 0, 0.01)):
    state = EstimatedState(
        np.array([1.0, 2.0, 3.0]),
        np.zeros(3),
        np.array([0, 0, 0, 1.0]),
        np.zeros(3),
        np.array(gyro_bias),
        np.eye(15),
    )
    return Sample("algorithms/x/state", 1, 0, t_ns, 1, "local_enu", state)


def test_estimate_state_uses_rotated_gyro_minus_bias_and_detects_staleness():
    mount = tuple(Rotation.from_euler("z", 90, degrees=True).as_quat())
    source = EstimateStateSource(mount, max_age_s=0.05)
    assert source.state(0, estimate_sample(0)) is None  # no gyro yet
    gyro = Sample(
        "sensors/imu", 1, 0, 0, 1, "drone/imu", ImuReading([0, 0, 9.81], [0.1, 0, 0])
    )
    source.on_delivered("imu", gyro)
    state = source.state(10 * MS, estimate_sample(0))
    np.testing.assert_allclose(state.position_m, [1, 2, 3])
    np.testing.assert_allclose(state.body_rates_rad_s, [0, 0.1, -0.01], atol=1e-12)
    assert state.source == "estimate"
    with pytest.raises(StaleEstimate):
        source.state(60 * MS, estimate_sample(0))


def test_config_validation_of_state_source(tmp_path):
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps({"algorithms": {"control_state_source": "estimate"}}))
    with pytest.raises(ValueError, match="needs algorithms.estimator"):
        load_drone_config(path)
