import numpy as np

from robotlab.core.frames import GRAVITY
from robotlab.simulation.state import initial_state
from robotlab.sensors.models import SensorConfig, Sensors


def test_imu_specific_force():
    config = SensorConfig(
        accel_std=0.0, gyro_std=0.0, accel_bias=np.zeros(3), gyro_bias=np.zeros(3)
    )
    sensors = Sensors(config=config)
    accel, gyro = sensors.imu(initial_state(), np.zeros(3))
    np.testing.assert_allclose(accel, -GRAVITY)
    np.testing.assert_allclose(gyro, 0.0)
    accel, _ = sensors.imu(initial_state(), GRAVITY)
    np.testing.assert_allclose(accel, 0.0)
