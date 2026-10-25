"""IMU calibration: static fit, driver with startup gyro bias, fingerprints."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.estimation.imu_calibration import ImuCalibration, ImuCalibrationDriver
from robotlab.evaluation.imu_bias import EffectiveImuBias
from robotlab.experiments.common.sensor_pipeline import calibration_warnings
from robotlab.experiments.calibration.imu_calibration import (
    AXIS_POSES,
    imu_calibration_config,
    write_static_world,
)
from robotlab.simulation.gazebo.config import load_drone_config, sensor_fingerprint

MS = 1_000_000
A = (
    np.eye(3)
    + np.diag([0.005, -0.004, 0.006])
    + np.array([[0, 0.0035, -0.002], [-0.003, 0, 0.0025], [0.002, -0.003, 0]])
)
G = np.eye(3) * 5e-5
B_A, B_G = np.array([0.05, -0.03, 0.02]), np.array([0.003, -0.002, 0.001])
K_A, K_G = np.array([0.006, -0.004, 0.008]), np.array([1.5e-4, -1e-4, 2e-4])


def static_data(temperatures=(5.0, 25.0, 45.0), noise=0.0):
    rng = np.random.default_rng(1)
    poses = list(AXIS_POSES) + list(Rotation.random(8, random_state=3))
    rows = [(p, t) for t in temperatures for p in poses]
    forces = np.array([p.inv().apply([0, 0, 9.81]) for p, _ in rows])
    temps = np.array([t for _, t in rows])
    accel = (
        forces @ A.T
        + B_A
        + np.outer(temps - 25, K_A)
        + rng.normal(size=forces.shape) * noise
    )
    gyro = forces @ G.T + B_G + np.outer(temps - 25, K_G)
    return accel, gyro, forces, temps


def test_static_fit_recovers_scale_offsets_temperature_and_g_sensitivity():
    accel, gyro, forces, temps = static_data()
    cal = ImuCalibration.fit_static(accel, gyro, forces, temps)
    np.testing.assert_allclose(cal.accel_correction @ A, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(cal.accel_offset_mps2, B_A, atol=1e-9)
    np.testing.assert_allclose(cal.accel_temperature_mps2_per_c, K_A, atol=1e-9)
    np.testing.assert_allclose(cal.gyro_g_sensitivity, G, atol=1e-12)
    np.testing.assert_allclose(cal.gyro_temperature_rad_s_per_c, K_G, atol=1e-12)
    force, rate = cal.apply(accel[3], gyro[3], temps[3])
    np.testing.assert_allclose(force, forces[3], atol=1e-9)
    np.testing.assert_allclose(rate, 0, atol=1e-12)


def test_single_temperature_reports_no_slope_and_needs_spanning_poses():
    accel, gyro, forces, temps = static_data(temperatures=(25.0,))
    cal = ImuCalibration.fit_static(accel, gyro, forces, temps)
    assert not cal.accel_temperature_mps2_per_c.any()
    with pytest.raises(ValueError, match="spanning"):
        ImuCalibration.fit_static(accel[:2], gyro[:2], forces[:2], temps[:2])


def test_driver_learns_startup_gyro_bias_then_subtracts_it():
    accel, gyro, forces, temps = static_data()
    cal = ImuCalibration.fit_static(accel, gyro, forces, temps).with_startup(0.5)
    driver = ImuCalibrationDriver(cal)
    turn_on = np.array([0.002, -0.001, 0.0005])  # changes at every power-up
    for i in range(101):  # 1 s at 100 Hz, still and level
        driver.apply(
            A @ [0, 0, 9.81] + B_A, G @ [0, 0, 9.81] + B_G + turn_on, 25.0, i * 10 * MS
        )
    assert driver.ready
    _, rate = driver.apply(
        A @ [0, 0, 9.81] + B_A, G @ [0, 0, 9.81] + B_G + turn_on, 25.0, 2000 * MS
    )
    np.testing.assert_allclose(rate, 0, atol=1e-12)
    assert ImuCalibrationDriver(cal.with_startup(0.0)).ready


def test_config_round_trip_and_staleness_fingerprints(tmp_path):
    accel, gyro, forces, temps = static_data()
    section = ImuCalibration.fit_static(accel, gyro, forces, temps).to_config()
    sensors = load_drone_config().data["sensors"]
    section["fingerprint"] = sensor_fingerprint(sensors, "imu")
    path = tmp_path / "imu-calibration.json"
    path.write_text(json.dumps({"sensors": {"imu": {"calibration": section}}}))
    config = load_drone_config(path)
    assert calibration_warnings(config.data["sensors"]) == []
    # Seeds change at every run; they must not make a calibration stale.
    reseeded = load_drone_config(path)
    reseeded.data["sensors"]["imu"]["error_model"]["seed"] = 99
    assert calibration_warnings(reseeded.data["sensors"]) == []
    changed = load_drone_config([path, "configs/sensors/imu_consumer_mems.json"])
    assert "imu calibration" in calibration_warnings(changed.data["sensors"])[0]
    bypassed = load_drone_config(
        [
            path,
            "configs/sensors/imu_consumer_mems.json",
            "configs/sensors/no_calibration.json",
        ]
    )
    assert calibration_warnings(bypassed.data["sensors"]) == []


def test_startup_calibration_needs_ground_start(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {"sensors": {"imu": {"calibration": {"startup_gyro_calibration_s": 2.0}}}}
        )
    )
    with pytest.raises(ValueError, match="start_on_ground"):
        load_drone_config(path)


def test_effective_bias_tracks_delivered_minus_ideal_in_body_axes():
    nominal = Rotation.from_euler("z", 90, degrees=True)
    tracker = EffectiveImuBias(nominal, nominal, time_constant_s=1.0)
    rng = np.random.default_rng(0)
    for i in range(2000):
        ideal = np.array([0.0, 0.0, 9.81])
        delivered = ideal + [0.1, 0.0, 0.0] + rng.normal(size=3) * 0.05
        tracker.update(
            i * 5 * MS, ideal, np.zeros(3), delivered, np.array([0.01, 0, 0])
        )
    accel, gyro = tracker.current()
    np.testing.assert_allclose(
        accel, [0.0, 0.1, 0.0], atol=0.02
    )  # rotated by the mount
    np.testing.assert_allclose(gyro, [0.0, 0.01, 0.0], atol=1e-12)


def test_static_calibration_world_and_gravity(tmp_path):
    config = imu_calibration_config(["configs/sensors/imu_consumer_mems.json"])
    assert not config.data["sensors"]["imu"]["calibration"]["enabled"]
    root = ET.parse(write_static_world(config, tmp_path)).getroot()
    assert root.find(".//model[@name='drone']").findtext("static") == "true"
    assert root.find("world").findtext("gravity") == "0 0 -9.81"


def test_calibration_priors_override_ekf_defaults_only_when_enabled():
    from robotlab.experiments.harness.algorithms import gps_imu_ekf_options

    def priors(calibration):
        return gps_imu_ekf_options(ekf, calibration)["filter_options"]

    ekf = load_drone_config().data["ekf"]
    calibration = dict(filter_accel_bias_std_mps2=0.05, filter_gyro_bias_std_rad_s=4e-4)
    tightened = priors(calibration)
    assert tightened["initial_accel_bias_std_mps2"] == 0.05
    assert tightened["initial_gyro_bias_std_rad_s"] == 4e-4
    assert ekf["initial_accel_bias_std_mps2"] == 0.2  # input untouched
    bypassed = priors(None)  # defaults
    assert bypassed["initial_accel_bias_std_mps2"] == 0.2
    assert bypassed["initial_gyro_bias_std_rad_s"] == 0.02
    partial = priors(
        dict(filter_accel_bias_std_mps2=None, filter_gyro_bias_std_rad_s=1e-3)
    )
    assert partial["initial_accel_bias_std_mps2"] == 0.2


def test_filter_node_predicts_with_calibrated_imu_values():
    from robotlab.core.measurements import GpsReading, ImuReading
    from robotlab.estimation.filter import FilterNode
    from robotlab.sensors.streams import Publisher

    class Recorder:
        """Filter stub recording what prediction receives."""

        def __init__(self) -> None:
            self.forces: list[list[float]] = []

        def predict_imu(self, imu, dt_s):
            self.forces.append(list(imu.specific_force_mps2))

        def update_gps(self, fix):
            return None

        def state(self):
            from robotlab.core.measurements import EstimatedState

            return EstimatedState(position_m=[0.0, 0.0, 0.0])

    recorder = Recorder()
    node = FilterNode(recorder, name="test", gps_reference=GpsReading(40, -3, 650))
    imu = Publisher("sensors/imu", ImuReading, frame_id="drone/imu")
    for t_ns in (0, 5_000_000):
        reading = ImuReading(
            [1.0, 0.0, 9.0],
            [0.1, 0.0, 0.0],
            calibrated_specific_force_mps2=[0.0, 0.0, 9.81],
            calibrated_angular_rate_rad_s=[0.0, 0.0, 0.0],
        )
        node.on_driver(imu.publish(reading, sim_time_ns=t_ns))
    # The held first reading predicts over the interval: its calibrated values.
    assert recorder.forces == [[0.0, 0.0, 9.81]]
