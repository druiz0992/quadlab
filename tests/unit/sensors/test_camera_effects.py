import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dronelab.sensors.camera_effects import (
    CameraEffects,
    distortion_map,
    focal_length_px,
    image_motion_px_s,
    motion_blur,
    rolling_shutter,
)
from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.world import build_world


def vertical_line(h=40, w=60, col=30):
    image = np.zeros((h, w, 3), np.uint8)
    image[:, col] = 255
    return image


def test_focal_length_and_rotation_to_image_motion():
    assert focal_length_px(640, math.radians(90), "pinhole") == pytest.approx(320)
    assert focal_length_px(640, math.radians(180), "equidistant") == pytest.approx(
        640 / math.pi
    )
    # Yaw left (+z) moves the scene right; pitch about +y (nose down) moves it up.
    np.testing.assert_allclose(image_motion_px_s(100, [0, 0, 2]), [200, 0])
    np.testing.assert_allclose(image_motion_px_s(100, [0, 1, 0]), [0, -100])


def test_rolling_shutter_slants_vertical_edges_by_readout_motion():
    out = rolling_shutter(
        vertical_line(), [300, 0], readout_s=0.02
    )  # 6 px over the frame
    first = np.argmax(out[0, :, 0]), np.argmax(out[-1, :, 0])
    assert first == (27, 33)  # top row read 10 ms early, bottom 10 ms late
    np.testing.assert_array_equal(
        rolling_shutter(vertical_line(), [0, 0], 0.02), vertical_line()
    )


def test_motion_blur_spreads_energy_along_motion():
    out = motion_blur(vertical_line(), [8, 0])
    row = out[20, :, 0].astype(int)
    assert (row > 0).sum() == 9  # 9 samples over 8 px
    assert row.sum() == pytest.approx(255, abs=9)  # energy roughly preserved
    np.testing.assert_array_equal(
        motion_blur(vertical_line(), [0.2, 0]), vertical_line()
    )


def test_barrel_distortion_pulls_edges_in_and_keeps_centre():
    rows, cols = distortion_map(101, 81, 50.0, dict(k1=-0.3, k2=0, k3=0, p1=0, p2=0))
    assert (rows[40, 50], cols[40, 50]) == (40, 50)  # centre unchanged
    assert (
        cols[40, 100] > 100 - 1 or cols[40, 100] == 100
    )  # samples from further out (clipped)
    # Pincushion samples from closer to the centre at the edge.
    _, cols_p = distortion_map(101, 81, 50.0, dict(k1=0.3, k2=0, k3=0, p1=0, p2=0))
    assert cols_p[40, 100] < 100


def camera(**overrides):
    settings = load_drone_config().data["sensors"]["camera"]
    settings.update(enabled=True, **overrides)
    return settings


def test_effects_disabled_by_default_and_track_blur():
    assert not CameraEffects(camera()).enabled
    effects = CameraEffects(camera(exposure_time_s=0.01, width=60, height=40))
    assert effects.enabled
    effects.process(vertical_line(), [0, 0, 3.0])
    assert effects.max_blur_px == pytest.approx(
        focal_length_px(60, 1.047, "pinhole") * 3.0 * 0.01
    )


def test_wide_angle_lens_generates_wideanglecamera_sdf(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "sensors": {
                    "camera": {
                        "enabled": True,
                        "horizontal_fov_rad": 6.2832,
                        "lens": {"type": "equidistant"},
                    }
                }
            }
        )
    )
    config = load_drone_config(path)
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "w.sdf", config=config)
    ).getroot()
    sensor = root.find(".//sensor[@name='camera']")
    assert sensor.get("type") == "wideanglecamera"
    assert sensor.findtext(".//lens/type") == "equidistant"


@pytest.mark.parametrize(
    "cam",
    [
        {"lens": {"type": "cubemap"}},
        {"lens": {"type": 3}},
        {"horizontal_fov_rad": 3.5},  # pinhole limited to < pi
        {"lens": {"type": "equidistant"}, "horizontal_fov_rad": 7.0},
        {"exposure_time_s": -0.01},
        {"latency_s": -1},
    ],
)
def test_invalid_camera_settings(tmp_path, cam):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"sensors": {"camera": cam}}))
    with pytest.raises(ValueError):
        load_drone_config(path)
