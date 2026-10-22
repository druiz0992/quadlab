import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.estimation.gps_imu_ekf import GpsImuEkf


def ekf(**kwargs):
    values = dict(
        position_m=[0, 0, 0],
        velocity_mps=[0, 0, 0],
        quaternion_xyzw=[0, 0, 0, 1],
        initial_covariance=np.eye(15) * 0.1,
    )
    values.update(kwargs)
    return GpsImuEkf(**values)


def test_noiseless_hover_and_body_to_world_rotation():
    f = ekf(
        quaternion_xyzw=Rotation.from_euler(
            "xyz", [20, -10, 30], degrees=True
        ).as_quat()
    )
    force = f.rotation.inv().apply([0, 0, 9.81])
    for _ in range(100):
        f.predict(force, [0, 0, 0], 0.01)
    np.testing.assert_allclose(f.position, 0, atol=1e-12)
    np.testing.assert_allclose(f.velocity, 0, atol=1e-12)
    f = ekf()
    for _ in range(100):
        f.predict([0, 0, 9.81], [0, 0, 0.2], 0.01)
    assert f.rotation.as_euler("xyz")[2] == pytest.approx(0.2)


def test_gps_limits_drift_and_covariance_remains_positive():
    fused, inertial = ekf(), ekf()
    rng = np.random.default_rng(7)
    for i in range(500):
        force = np.array([0.08, -0.04, 9.85]) + rng.normal(0, 0.03, 3)
        gyro = np.array([0.001, -0.001, 0.002])
        for f in (fused, inertial):
            f.predict(force, gyro, 0.01)
        if i % 10 == 0:
            fused.correct_gps(rng.normal(0, 0.08, 3), np.eye(3) * 0.08**2)
        np.testing.assert_allclose(fused.P, fused.P.T, atol=1e-12)
        assert np.linalg.eigvalsh(fused.P).min() > 0
    assert np.linalg.norm(fused.position) < 0.3
    assert np.linalg.norm(fused.position) < np.linalg.norm(inertial.position) / 3


def test_receiver_lever_arm_and_gps_outlier_gate():
    f = ekf(quaternion_xyzw=Rotation.from_euler("z", 90, degrees=True).as_quat())
    p = f.P.copy()
    assert f.correct_gps([0, 1, 0], np.eye(3) * 0.1, receiver_position_body_m=[1, 0, 0])
    np.testing.assert_allclose(f.position, 0, atol=1e-12)
    assert np.trace(f.P) < np.trace(p)
    before = f.state()
    assert not f.correct_gps([1000] * 3, np.eye(3) * 0.1)
    np.testing.assert_array_equal(f.P, before.covariance)
    np.testing.assert_array_equal(f.position, before.position_m)
    assert f.rejected_gps == 1


def test_prediction_covariance_matches_numerical_error_state_jacobian():
    kwargs = dict(
        position_m=[1, 2, 3],
        velocity_mps=[0.1, -0.2, 0.3],
        quaternion_xyzw=Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_quat(),
        accel_bias_mps2=[0.01, -0.02, 0.03],
        gyro_bias_rad_s=[0.001, 0.002, -0.001],
        accel_std_mps2=0,
        gyro_std_rad_s=0,
        accel_bias_walk=0,
        gyro_bias_walk=0,
    )
    force, gyro, dt = [0.3, 0.2, 9.7], [0.2, -0.1, 0.3], 0.02
    reference = ekf(**kwargs)
    reference.predict(force, gyro, dt)
    J = np.zeros((15, 15))
    for column in range(15):
        errors = []
        for sign in (-1, 1):
            perturbed = ekf(**kwargs)
            delta = np.zeros(15)
            delta[column] = sign * 1e-6
            perturbed.position += delta[:3]
            perturbed.velocity += delta[3:6]
            perturbed.rotation = perturbed.rotation * Rotation.from_rotvec(delta[6:9])
            perturbed.accel_bias += delta[9:12]
            perturbed.gyro_bias += delta[12:15]
            perturbed.predict(force, gyro, dt)
            errors.append(
                np.r_[
                    perturbed.position - reference.position,
                    perturbed.velocity - reference.velocity,
                    (reference.rotation.inv() * perturbed.rotation).as_rotvec(),
                    perturbed.accel_bias - reference.accel_bias,
                    perturbed.gyro_bias - reference.gyro_bias,
                ]
            )
        J[:, column] = (errors[1] - errors[0]) / 2e-6
    np.testing.assert_allclose(reference.P, J @ (np.eye(15) * 0.1) @ J.T, atol=1e-8)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), 0.2])
def test_invalid_dt_does_not_change_state(dt):
    f = ekf()
    with pytest.raises(ValueError):
        f.predict([0, 0, 9.81], [0, 0, 0], dt)
    np.testing.assert_array_equal(f.position, 0)
