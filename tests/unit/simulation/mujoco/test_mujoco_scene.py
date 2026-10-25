"""Scene loading: edits, validation, and saved MJCF (needs the mujoco extra)."""

from pathlib import Path

import numpy as np
import pytest

from robotlab.simulation.mujoco.config import BUILTIN_MODELS, load_biped_config
from robotlab.simulation.mujoco.scene import FLOOR_GEOM, build_scene
from robotlab.simulation.mujoco.state import imu_reading, read_truth

mujoco = pytest.importorskip("mujoco")
TEST_BIPED = BUILTIN_MODELS["test_biped"].read_text()


def settings(**biped: object) -> dict:
    data = load_biped_config().data
    data["biped"].update(biped)
    return data


def model_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "robot.xml"
    path.write_text(text)
    return path


def test_builtin_biped_indices_and_initial_state() -> None:
    scene = build_scene(settings(), BUILTIN_MODELS["test_biped"])
    assert scene.joint_names == (
        "left_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
        "right_hip_pitch",
        "right_knee",
        "right_ankle_pitch",
    )
    assert scene.step_ns == 2_000_000
    assert scene.overrides == {}
    data = scene.new_data()
    state = read_truth(scene, data, 0)
    np.testing.assert_allclose(state.position_m, [0, 0, 0.167])
    np.testing.assert_allclose(state.quaternion_xyzw, [0, 0, 0, 1])
    np.testing.assert_array_equal(scene.initial_ctrl, np.zeros(6))


def test_overrides_are_reported_not_silent(tmp_path: Path) -> None:
    text = TEST_BIPED.replace(
        'timestep="0.002" gravity="0 0 -9.81"', 'timestep="0.005" gravity="0 0 -9.8"'
    )
    scene = build_scene(settings(), model_file(tmp_path, text))
    assert scene.overrides == {
        "model_timestep_s": 0.005,
        "model_gravity_mps2": [0.0, 0.0, -9.8],
    }
    assert scene.model.opt.timestep == 0.002
    np.testing.assert_allclose(scene.model.opt.gravity, [0, 0, -9.81])


def test_floor_added_only_when_missing(tmp_path: Path) -> None:
    text = TEST_BIPED.replace(
        '<geom name="floor" type="plane" size="0 0 0.05" rgba="0.3 0.35 0.4 1"/>', ""
    )
    scene = build_scene(settings(), model_file(tmp_path, text))
    assert scene.overrides["floor_added"] is True
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM) >= 0
    without = build_scene(settings(add_floor=False), model_file(tmp_path, text))
    assert "floor_added" not in without.overrides


def test_imu_reads_specific_force_at_rest() -> None:
    scene = build_scene(settings(), BUILTIN_MODELS["test_biped"])
    data = scene.new_data()
    for _ in range(500):  # settle onto the floor (1 s)
        mujoco.mj_step(scene.model, data)
    mujoco.mj_forward(scene.model, data)
    force, rate = imu_reading(scene, data)
    np.testing.assert_allclose(force, [0, 0, 9.81], atol=0.05)
    np.testing.assert_allclose(rate, [0, 0, 0], atol=1e-3)


@pytest.mark.parametrize(
    "edit, layer, message",
    [
        (
            ('<freejoint name="trunk_freejoint"/>', ""),
            {"drop_keyframe": True},
            "exactly one free joint",
        ),
        (('<site name="imu" pos="0 0 0.01"/>', ""), {}, "no site 'imu'"),
        ((None, None), {"initial_keyframe": "WALK"}, "no keyframe 'WALK'"),
        (
            (
                '<position name="left_knee" joint="left_knee"/>',
                '<motor name="left_knee" joint="left_knee"/>',
            ),
            {},
            "only position actuators",
        ),
        (("<mujoco", "<mujoco broken"), {}, "Cannot load MJCF"),
    ],
)
def test_invalid_models_raise(
    tmp_path: Path, edit: tuple, layer: dict, message: str
) -> None:
    old, new = edit
    text = TEST_BIPED if old is None else TEST_BIPED.replace(old, new)
    if layer.pop("drop_keyframe", False):  # its qpos no longer fits
        start, end = text.index("<keyframe>"), text.index("</keyframe>")
        text = text[:start] + text[end + len("</keyframe>") :]
    with pytest.raises(ValueError, match=message):
        build_scene(settings(**layer), model_file(tmp_path, text))


def test_saved_scene_loads_from_another_directory(tmp_path: Path) -> None:
    scene = build_scene(settings(), BUILTIN_MODELS["test_biped"])
    saved = tmp_path / "elsewhere" / "scene.xml"
    saved.parent.mkdir()
    scene.write_xml(saved)
    model = mujoco.MjModel.from_xml_path(str(saved))
    assert model.nu == 6 and model.nsensor == 2


def test_model_source_records_git_checkout() -> None:
    scene = build_scene(settings(), BUILTIN_MODELS["test_biped"])
    assert scene.source.path.endswith("test_biped.xml")
    # The built-in model lives in this repository; outside git both are None.
    assert (scene.source.commit is None) == (scene.source.dirty is None)


def test_properties_and_imu_mount(tmp_path: Path) -> None:
    text = TEST_BIPED.replace(
        '<site name="imu" pos="0 0 0.01"/>',
        '<site name="imu" pos="0.02 0 0.01" euler="0 0 1.5707963"/>',
    ).replace(
        '<position name="left_knee" joint="left_knee"/>',
        '<position name="left_knee" joint="left_knee" ctrllimited="false"/>',
    )
    scene = build_scene(settings(), model_file(tmp_path, text))
    robot = scene.properties(50.0)
    assert robot.joint_names == scene.joint_names and robot.mass_kg == pytest.approx(
        0.64
    )
    np.testing.assert_allclose(robot.position_gain, 6.0)
    # No control range on the knee actuator: the joint range bounds it.
    assert robot.target_lower_rad[1] == -1.5 and robot.target_upper_rad[1] == 1.5
    position, quaternion = scene.imu_mount()
    np.testing.assert_allclose(position, [0.02, 0, 0.01], atol=1e-9)
    s = np.sqrt(0.5)
    np.testing.assert_allclose(quaternion, [0, 0, s, s], atol=1e-6)
