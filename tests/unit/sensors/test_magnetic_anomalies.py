"""Dipole anomalies and time-varying magnetometer interference."""

import math

import numpy as np
import pytest

from dronelab.sensors.magnetic_anomalies import MagneticAnomalies, scene_dipoles
from dronelab.sensors.magnetometer import MagnetometerErrorModel
from dronelab.simulation.gazebo.config import load_drone_config

MS = 1_000_000


def test_dipole_on_axis_and_equatorial_fields():
    dipole = MagneticAnomalies([dict(position_m=[0, 0, 0], moment_am2=[0, 0, 1000])])
    # On the axis: 2·1e-7·m/r³; in the equatorial plane: −1e-7·m/r³.
    np.testing.assert_allclose(dipole.field_world([0, 0, 5]), [0, 0, 2e-7 * 1000 / 125])
    np.testing.assert_allclose(
        dipole.field_world([5, 0, 0]), [0, 0, -1e-7 * 1000 / 125]
    )
    assert dipole.max_field_ut == pytest.approx(1.6)
    # Clamped inside the object.
    assert np.all(np.isfinite(dipole.field_world([0, 0, 0.01])))
    assert not MagneticAnomalies([]).enabled


def test_scene_dipoles_use_obstacle_centres():
    scene = dict(
        obstacles=[
            dict(
                name="car",
                position=[3, 1, 0.7],
                size=[4, 2, 1.4],
                magnetic_moment_am2=[0, 500, 0],
            ),
            dict(name="wall", position=[0, 0, 1], size=[1, 1, 1]),
        ]
    )
    assert scene_dipoles(scene) == [
        dict(position_m=[3, 1, 0.7], moment_am2=[0, 500, 0])
    ]


def errors(**overrides):
    settings = dict(
        load_drone_config().data["sensors"]["magnetometer"]["error_model"], enabled=True
    )
    settings.update(overrides)
    return MagnetometerErrorModel(settings)


def test_current_interference_and_drift():
    model = errors(current_interference_ut_per_a=[0.5, 0, 0])
    field = model.apply([2e-5, 0, 4e-5], 0.0, 20.0, 0)
    np.testing.assert_allclose(field, [2e-5 + 10e-6, 0, 4e-5])
    drifting = errors(drift_std_ut=[2.0, 2.0, 2.0], drift_correlation_s=10.0, seed=4)
    values = np.array(
        [
            drifting.apply([0, 0, 0], sim_time_ns=i * 100 * MS)[0] * 1e6
            for i in range(40000)
        ]
    )
    assert values.std() == pytest.approx(2.0, rel=0.15)
    assert np.corrcoef(values[:-100], values[100:])[0, 1] == pytest.approx(
        math.exp(-1), abs=0.1
    )
