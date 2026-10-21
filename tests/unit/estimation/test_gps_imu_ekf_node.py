import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronelab.core.measurements import EstimatedState, GpsReading, ImuReading
from dronelab.estimation.gps_imu_ekf import GpsImuEkf, GpsImuEkfNode
from dronelab.sensors.streams import Publisher
from dronelab.sensors.geodesy import gps_to_enu


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
