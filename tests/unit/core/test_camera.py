"""Camera calibration and sensor extrinsics in the algorithm context."""

import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.core.algorithms import SensorDescription, extrinsics
from robotlab.core.camera import CameraCalibration
from robotlab.core.serialization import decode, encode
from robotlab.sensors.imaging.camera_effects import distortion_map
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.testing.scenarios import SCENARIOS, scenario_context, scenario_samples
from robotlab.testing.synthetic_scene import BOX_RGB, SceneBox, SyntheticScene

CAMERA = load_drone_config().data["sensors"]["camera"]


def test_intrinsics_from_the_default_camera():
    calibration = CameraCalibration.from_settings(CAMERA)
    focal = 160 / math.tan(1.047 / 2)
    np.testing.assert_allclose(
        calibration.matrix, [[focal, 0, 159.5], [0, focal, 119.5], [0, 0, 1]]
    )
    np.testing.assert_allclose(calibration.distortion, 0.0)
    # The optical axis hits the principal point.
    np.testing.assert_allclose(calibration.project([[0.0, 0.0, 2.0]]), [[159.5, 119.5]])


def test_projection_matches_the_rendered_distortion():
    """``project`` inverts the remap RobotLab applies to rendered frames."""
    lens = dict(CAMERA["lens"], k1=-0.2, k2=0.05, p1=0.001, p2=-0.002)
    calibration = CameraCalibration.from_settings(dict(CAMERA, lens=lens))
    rows, cols = distortion_map(320, 240, calibration.fx_px, lens)
    for point in ([0.4, -0.3, 1.0], [-0.5, 0.2, 1.0], [0.1, 0.1, 1.0]):
        u, v = np.rint(calibration.project([point])[0]).astype(int)
        # The output pixel samples the undistorted pinhole pixel of the point.
        pinhole = dataclasses.replace(calibration, k1=0, k2=0, p1=0, p2=0)
        expected = pinhole.project([point])[0]
        assert (
            abs(cols[v, u] - expected[0]) <= 1.5
            and abs(rows[v, u] - expected[1]) <= 1.5
        )


def test_extrinsics_between_sensors_and_the_optical_frame():
    camera = SensorDescription(
        "camera",
        "drone/camera",
        10.0,
        (0.3, 0.0, 0.1),
        tuple(Rotation.from_euler("y", 30, degrees=True).as_quat()),
        camera=CameraCalibration.from_settings(CAMERA),
    )
    imu = SensorDescription("imu", "drone/imu", 200.0, (0.0, 0.0, 0.05))
    rotation, translation = extrinsics(camera, imu, source_optical=True)
    # Optical +z (forward) is camera +x, pitched 30° down about body y.
    forward = rotation @ [0.0, 0.0, 1.0]
    np.testing.assert_allclose(
        forward, [math.cos(math.radians(30)), 0, -0.5], atol=1e-12
    )
    np.testing.assert_allclose(
        rotation @ [1.0, 0.0, 0.0], [0, -1, 0], atol=1e-12
    )  # right
    np.testing.assert_allclose(translation, [0.3, 0.0, 0.05])
    # The inverse transform: IMU axes into camera (sensor) axes.
    back, offset = extrinsics(imu, camera)
    np.testing.assert_allclose(back @ camera.to_body()[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(back @ [0.3, 0.0, 0.05] + offset, 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="no camera optical frame"):
        extrinsics(imu, camera, source_optical=True)


def test_projection_errors():
    calibration = CameraCalibration.from_settings(CAMERA)
    with pytest.raises(ValueError, match="in front"):
        calibration.project([[0.0, 0.0, -1.0]])
    fisheye = CameraCalibration.from_settings(
        dict(CAMERA, lens=dict(CAMERA["lens"], type="equidistant"))
    )
    with pytest.raises(ValueError, match="pinhole"):
        fisheye.project([[0.0, 0.0, 1.0]])


def test_context_carries_the_camera_and_serializes():
    context = scenario_context(SCENARIOS["hover"], ("imu", "camera"))
    camera = context.sensors["camera"]
    assert camera.camera == CameraCalibration.from_settings(CAMERA)
    assert context.sensors["imu"].camera is None
    restored = decode(encode(context.sensors))
    assert restored["camera"].camera == camera.camera


def test_projected_world_point_lands_on_the_synthetic_image():
    """End to end: truth pose + context extrinsics + intrinsics → the pixel
    where the synthetic camera rendered a known box."""
    box = SceneBox((6.0, 1.0, 0.5), (6.4, 1.4, 1.5))
    scenario = dataclasses.replace(
        SCENARIOS["hover"],
        scene=SyntheticScene((box,)),
        duration_s=0.0,
        settings={"sensors": {"camera": {"intensity": {"noise_stddev": 0.0}}}},
    )
    image = next(s for k, s in scenario_samples(scenario, ("camera",))).payload.image
    context = scenario_context(scenario, ("camera",))
    camera = context.sensors["camera"]
    truth = scenario.trajectory(0.0)
    body = Rotation.from_quat(truth.quaternion_xyzw)
    # Centre of the box face looking at the camera (x = 6.0).
    face_world = np.array([6.0, 1.2, 1.0])
    point_body = body.inv().apply(face_world - truth.position_m)
    to_optical, offset = extrinsics(
        SensorDescription("body", "drone/body", None), camera
    )
    from robotlab.core.camera import SENSOR_FROM_OPTICAL

    point_optical = SENSOR_FROM_OPTICAL.T @ (to_optical @ point_body + offset)
    u, v = np.rint(camera.camera.project([point_optical])[0]).astype(int)
    pixel = image[v, u].astype(float)
    # Lit box face: a shade of the first box colour, not sky or ground.
    ratio = pixel / BOX_RGB[0]
    assert np.allclose(ratio, ratio[0], atol=0.02) and 0.3 < ratio[0] <= 1.0
