"""Truth trajectories (robotlab.testing.trajectories): exact derivatives."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.testing.trajectories import (
    Harmonic,
    HarmonicPath,
    circling,
    figure_eight,
    flown,
    tumbling,
)

STEP_S = 1e-5


def assert_consistent(trajectory, times):
    """Velocity, acceleration, and body rates match central differences."""
    for t in times:
        before, now, after = (trajectory(t + d) for d in (-STEP_S, 0.0, STEP_S))
        np.testing.assert_allclose(
            (after.position_m - before.position_m) / (2 * STEP_S),
            now.velocity_mps,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            (after.velocity_mps - before.velocity_mps) / (2 * STEP_S),
            now.acceleration_mps2,
            atol=1e-6,
        )
        relative = Rotation.from_quat(
            before.quaternion_xyzw
        ).inv() * Rotation.from_quat(after.quaternion_xyzw)
        np.testing.assert_allclose(
            relative.as_rotvec() / (2 * STEP_S), now.body_rate_rad_s, atol=1e-6
        )


TIMES = np.linspace(0.3, 19.7, 25)


@pytest.mark.parametrize(
    "trajectory",
    [
        flown(figure_eight(6.0, 16.0, climb_m=1.0), yaw_along_path=True),
        flown(figure_eight(4.0, 7.0, climb_m=0.5), yaw_rate_rad_s=0.8),
        flown(HarmonicPath((0, 0, 2), (1.0, 0.5, 0.2)), yaw_rad=0.3),
        tumbling(),
        circling(5.0, 2.0),
    ],
)
def test_trajectories_are_self_consistent(trajectory):
    assert_consistent(trajectory, TIMES)


def test_flown_thrust_is_along_body_z():
    """A multirotor's accelerometer reads only thrust (no drag)."""
    trajectory = flown(figure_eight(4.0, 7.0, climb_m=0.5), yaw_rate_rad_s=0.8)
    tilts = []
    for t in TIMES:
        point = trajectory(t)
        body = Rotation.from_quat(point.quaternion_xyzw)
        force = body.inv().apply(point.acceleration_mps2 + [0.0, 0.0, 9.81])
        np.testing.assert_allclose(force[:2], 0.0, atol=1e-9)
        assert force[2] > 0
        tilts.append(math.degrees(math.acos(body.apply([0.0, 0.0, 1.0])[2])))
    assert max(tilts) > 30.0  # aggressive: the body really banks


def test_yaw_along_path_follows_the_velocity():
    trajectory = flown(figure_eight(6.0, 16.0), yaw_along_path=True)
    point = trajectory(3.0)
    nose = Rotation.from_quat(point.quaternion_xyzw).apply([1.0, 0.0, 0.0])
    heading = math.atan2(point.velocity_mps[1], point.velocity_mps[0])
    assert math.atan2(nose[1], nose[0]) == pytest.approx(heading, abs=1e-9)


def test_figure_eight_shape():
    path = figure_eight(6.0, 16.0, height_m=2.0)
    positions = np.array([path.derivatives(t)[0] for t in np.linspace(0, 16, 161)])
    assert positions[:, 0].max() == pytest.approx(6.0)
    assert positions[:, 1].max() == pytest.approx(3.0)
    np.testing.assert_allclose(positions[0], positions[-1], atol=1e-9)


def test_invalid_trajectories_raise():
    with pytest.raises(ValueError, match="axis"):
        HarmonicPath((0, 0, 0), harmonics=(Harmonic(3, 1.0, 1.0),))
    with pytest.raises(ValueError, match="horizontal motion"):
        flown(HarmonicPath((0, 0, 1)), yaw_along_path=True)(0.0)
    falling = HarmonicPath((0, 0, 0), harmonics=(Harmonic(2, 9.81, 1.0),))
    with pytest.raises(ValueError, match="free fall"):
        flown(falling)(math.pi / 2)  # acceleration -9.81 m/s² vertical
    with pytest.raises(ValueError, match="90°"):
        tumbling(pitch_amplitude_rad=math.pi / 2)
