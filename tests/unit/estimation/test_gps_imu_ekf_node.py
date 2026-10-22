import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.core.measurements import EstimatedState, GpsReading, ImuReading
from robotlab.estimation.gps_imu_ekf import GpsImuEkf, GpsImuEkfNode
from robotlab.sensors.streams import Publisher
from robotlab.sensors.geodesy import gps_to_enu


def test_rotated_imu_output_fanout_and_explicit_gps_timing():
    reference = GpsReading(40, -3, 650)
    ekf = GpsImuEkf(
        position_m=[0, 0, 0],
        velocity_mps=[0, 0, 0],
        quaternion_xyzw=[0, 0, 0, 1],
        initial_covariance=np.eye(15),
    )
    node = GpsImuEkfNode(
        ekf,
        gps_reference=reference,
        gps_covariance_m2=np.eye(3),
        imu_quaternion_body_xyzw=Rotation.from_euler("x", 180, degrees=True).as_quat(),
    )
    downstream = node.estimates.subscribe(capacity=8)
    recorder = node.estimates.subscribe(capacity=8)
    imu = Publisher("imu", ImuReading, frame_id="drone/imu")
    gps = Publisher("gps", GpsReading, frame_id="wgs84")
    try:
        first = node.on_imu(
            imu.publish(ImuReading([0, 0, -9.81], [0, 0, 0]), sim_time_ns=0)
        )
        node.on_imu(
            imu.publish(ImuReading([0, 0, -9.81], [0, 0, 0]), sim_time_ns=10_000_000)
        )
        assert node.on_gps(gps.publish(reference, sim_time_ns=10_000_000))
        assert not node.on_gps(gps.publish(reference, sim_time_ns=20_000_000))
        assert node.rejected_gps_timing == 1
        for i in range(3):
            state = downstream.next(timeout_wall_s=0)
            other = recorder.next(timeout_wall_s=0)
            assert state is other and state.sequence == i + 1
            assert isinstance(state.payload, EstimatedState)
            np.testing.assert_allclose(state.payload.position_m, 0, atol=1e-12)
        with pytest.raises(ValueError):
            first.payload.covariance[0, 0] = 123
        node.close()
        with pytest.raises(RuntimeError, match="closed"):
            node.on_imu(
                imu.publish(
                    ImuReading([0, 0, -9.81], [0, 0, 0]), sim_time_ns=30_000_000
                )
            )
    finally:
        node.close()
        downstream.close()
        recorder.close()
        imu.close()
        gps.close()


def test_geodetic_reference_and_altitude_not_world_z():
    origin = GpsReading(40, -3, 650)
    np.testing.assert_allclose(gps_to_enu(origin, origin), 0)
    np.testing.assert_allclose(
        gps_to_enu(GpsReading(40, -3, 651), origin), [0, 0, 1], atol=1e-8
    )


def _node(**flags):
    ekf = GpsImuEkf(
        position_m=[0, 0, 0],
        velocity_mps=[0, 0, 0],
        quaternion_xyzw=[0, 0, 0, 1],
        initial_covariance=np.eye(15) * 25.0,  # 5 m position prior
    )
    reference = GpsReading(40, -3, 650)
    node = GpsImuEkfNode(
        ekf, gps_reference=reference, gps_covariance_m2=np.eye(3) * 0.25, **flags
    )
    imu = Publisher("imu", ImuReading, frame_id="drone/imu")
    gps = Publisher("gps", GpsReading, frame_id="wgs84")
    node.on_imu(imu.publish(ImuReading([0, 0, 9.81], [0, 0, 0]), sim_time_ns=0))
    return node, reference, imu, gps


def test_first_fix_initializes_position_and_its_covariance():
    from robotlab.core.measurements import GpsStatus

    node, reference, imu, gps = _node(
        initialize_from_gps=True, use_receiver_accuracy=True
    )
    try:
        # A fix 3 m up and 10 m east of the prior, with receiver accuracies.
        fix = GpsReading(
            40.0,
            -3.0 + 10 / 85_000,
            653.0,
            status=GpsStatus("3d", 12, 1.0, 1.5, 2.4, 0.1),
        )
        assert node.on_gps(gps.publish(fix, sim_time_ns=0))
        expected = gps_to_enu(fix, reference)
        np.testing.assert_allclose(node.ekf.position, expected, atol=1e-9)
        np.testing.assert_allclose(np.diag(node.ekf.P)[:3], [1.5**2, 1.5**2, 2.4**2])
        assert not node.ekf.P[:3, 3:].any()  # cross-covariances cleared
        assert node.position_initialized_ns == 0
        assert node.ekf.accepted_gps == 0  # an initialization, not a correction
        assert node.receiver_covariance_fixes == 1
        # Later fixes correct normally.
        node.on_imu(
            imu.publish(ImuReading([0, 0, 9.81], [0, 0, 0]), sim_time_ns=100_000_000)
        )
        assert node.on_gps(gps.publish(fix, sim_time_ns=100_000_000))
        assert node.ekf.accepted_gps == 1
    finally:
        node.close()
        imu.close()
        gps.close()


def test_receiver_accuracy_is_used_only_when_valid():
    from robotlab.core.measurements import GpsStatus

    node, reference, imu, gps = _node(use_receiver_accuracy=True)
    try:
        configured = node._fix_covariance(reference)  # no status: configured
        np.testing.assert_allclose(configured, np.eye(3) * 0.25)
        good = GpsReading(40, -3, 650, status=GpsStatus("3d", 12, 1.0, 2.0, 3.0, 0.1))
        np.testing.assert_allclose(node._fix_covariance(good), np.diag([4.0, 4.0, 9.0]))
        bad = GpsReading(40, -3, 650, status=GpsStatus("3d", 12, 1.0, 0.0, 3.0, 0.1))
        np.testing.assert_allclose(node._fix_covariance(bad), np.eye(3) * 0.25)
        assert node.receiver_covariance_fixes == 1
    finally:
        node.close()
        imu.close()
        gps.close()
    off, _, imu2, gps2 = _node()
    try:
        np.testing.assert_allclose(off._fix_covariance(good), np.eye(3) * 0.25)
    finally:
        off.close()
        imu2.close()
        gps2.close()
