"""Load a biped MJCF model into a validated, ready-to-step MuJoCo scene.

``build_scene`` reads ``biped.model_path`` with ``mujoco.MjSpec`` and
edits it before compiling:

- the physics step becomes ``biped.step_s`` and gravity (0, 0, -9.81) m/s²
  (AGENTS.md §4); the model's own values are reported in
  ``BipedScene.overrides`` rather than replaced silently;
- an accelerometer and a gyro are added on ``biped.imu_site`` (noise-free;
  RobotLab's IMU error model perturbs them, sensors/imu_errors.py);
- a z = 0 ground plane is added when ``biped.add_floor`` is set and the
  world body has no plane.

Requirements (ValueError otherwise): exactly one free joint (the trunk),
the IMU site, the initial keyframe, and only position actuators, each
driving one hinge or slide joint, so a command is a joint target (rad or
m). MuJoCo's quaternions are (w, x, y, z); everything leaving this package
is converted to RobotLab's (x, y, z, w).
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.legged import LeggedProperties

GRAVITY_MPS2 = (0.0, 0.0, -9.81)
ACCEL_SENSOR = "robotlab_imu_accelerometer"
GYRO_SENSOR = "robotlab_imu_gyro"
FLOOR_GEOM = "robotlab_floor"


@dataclass(frozen=True)
class ModelSource:
    """Where the MJCF came from: its path and, when the file is in a git
    checkout, that checkout's commit and whether it had local changes."""

    path: str
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True)
class BipedScene:
    """A compiled MuJoCo model and the indices RobotLab reads it through.

    ``joint_names`` are the actuated joints in actuator (command) order;
    ``qpos_index``/``dof_index`` locate them in ``qpos``/``qvel``.
    ``free_qpos``/``free_dof`` are the trunk free joint's first addresses
    and ``trunk_body`` its body; ``imu_site`` is the IMU site's index.
    ``initial_ctrl`` holds the keyframe's actuator targets.
    """

    model: Any
    spec: Any
    name: str
    joint_names: tuple[str, ...]
    qpos_index: NDArray[np.int64]
    dof_index: NDArray[np.int64]
    free_qpos: int
    free_dof: int
    trunk_body: int
    imu_site: int
    keyframe_id: int
    initial_ctrl: NDArray[np.float64]
    ctrl_range: NDArray[np.float64]
    accel_address: int
    gyro_address: int
    source: ModelSource
    overrides: dict[str, Any]

    @property
    def step_ns(self) -> int:
        """Physics step (integer nanoseconds)."""
        return round(float(self.model.opt.timestep) * 1e9)

    def new_data(self) -> Any:
        """Fresh ``MjData`` reset to the initial keyframe (qpos, qvel, ctrl)."""
        import mujoco

        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.keyframe_id)
        mujoco.mj_forward(self.model, data)
        return data

    def properties(self, control_rate_hz: float) -> LeggedProperties:
        """Nominal ``LeggedProperties`` for the algorithm context: command
        bounds from the actuators' control ranges (the joint range when an
        actuator has none), servo gains, keyframe targets, total mass."""
        model = self.model
        lower, upper = self.ctrl_range[:, 0].copy(), self.ctrl_range[:, 1].copy()
        joints = model.actuator_trnid[:, 0]
        for a in range(model.nu):
            if not model.actuator_ctrllimited[a]:
                joint = joints[a]
                limited = model.jnt_limited[joint]
                lower[a] = model.jnt_range[joint, 0] if limited else -np.inf
                upper[a] = model.jnt_range[joint, 1] if limited else np.inf
        return LeggedProperties(
            name=self.name,
            joint_names=self.joint_names,
            target_lower_rad=lower,
            target_upper_rad=upper,
            position_gain=np.array(model.actuator_gainprm[:, 0], dtype=float),
            initial_target_rad=self.initial_ctrl,
            mass_kg=float(np.sum(model.body_mass)),
            control_rate_hz=control_rate_hz,
        )

    def imu_mount(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """IMU site pose in trunk-body axes at the initial keyframe:
        (position m, quaternion xyzw rotating sensor axes into body axes)."""
        import mujoco

        data = self.new_data()
        trunk_r = data.xmat[self.trunk_body].reshape(3, 3)
        site_r = data.site_xmat[self.imu_site].reshape(3, 3)
        offset = trunk_r.T @ (
            data.site_xpos[self.imu_site] - data.xpos[self.trunk_body]
        )
        wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(wxyz, (trunk_r.T @ site_r).flatten())
        w, x, y, z = wxyz
        return tuple(float(v) for v in offset), (float(x), float(y), float(z), float(w))

    def write_xml(self, path: Path) -> None:
        """Save the edited model as MJCF (asset paths made absolute so the
        file loads from any directory)."""
        path.write_text(self.spec.to_xml())


def build_scene(settings: dict[str, Any], model_path: Path) -> BipedScene:
    """Compile the validated ``settings`` (simulation/mujoco/config.py) and
    the MJCF at ``model_path`` into a ``BipedScene`` (see module docs)."""
    import mujoco

    try:
        spec = mujoco.MjSpec.from_file(str(model_path))
    except Exception as error:  # MuJoCo raises ValueError or mujoco.FatalError
        raise ValueError(f"Cannot load MJCF {model_path}: {error}") from error
    _absolute_asset_dirs(spec, model_path)
    biped = settings["biped"]
    overrides = _override_physics(spec, biped["step_s"])
    if biped["add_floor"] and not _has_floor(spec, mujoco):
        spec.worldbody.add_geom(
            name=FLOOR_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05]
        )
        overrides["floor_added"] = True
    _add_imu_sensors(spec, biped["imu_site"], mujoco)
    try:
        model = spec.compile()
    except Exception as error:
        raise ValueError(f"Cannot compile MJCF {model_path}: {error}") from error
    return _index(
        model, spec, biped["initial_keyframe"], biped["imu_site"], model_path, overrides
    )


def _absolute_asset_dirs(spec: Any, model_path: Path) -> None:
    """Resolve mesh and texture directories against the model file, so the
    compiled and saved scene do not depend on the working directory."""
    base = model_path.resolve().parent
    for attribute in ("meshdir", "texturedir"):
        value = getattr(spec, attribute) or ""
        setattr(spec, attribute, str((base / value).resolve()))


def _override_physics(spec: Any, step_s: float) -> dict[str, Any]:
    """Set the step and gravity; return the model's own values that changed."""
    overrides: dict[str, Any] = {}
    model_step = float(spec.option.timestep)
    if model_step != step_s:
        overrides["model_timestep_s"] = model_step
    model_gravity = [float(g) for g in spec.option.gravity]
    if model_gravity != list(GRAVITY_MPS2):
        overrides["model_gravity_mps2"] = model_gravity
    spec.option.timestep = step_s
    spec.option.gravity = list(GRAVITY_MPS2)
    return overrides


def _has_floor(spec: Any, mujoco: Any) -> bool:
    return any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms)


def _add_imu_sensors(spec: Any, site: str, mujoco: Any) -> None:
    if spec.site(site) is None:
        raise ValueError(f"biped.imu_site: the model has no site {site!r}")
    for name, kind in (
        (ACCEL_SENSOR, mujoco.mjtSensor.mjSENS_ACCELEROMETER),
        (GYRO_SENSOR, mujoco.mjtSensor.mjSENS_GYRO),
    ):
        spec.add_sensor(
            name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site
        )


def _index(
    model: Any,
    spec: Any,
    keyframe: str,
    imu_site: str,
    path: Path,
    overrides: dict[str, Any],
) -> BipedScene:
    """Validate the compiled model (see module docs) and build the scene."""
    import mujoco

    free_joint = _free_joint(model, mujoco)
    joints = _actuated_joints(model, mujoco)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise ValueError(
            f"biped.initial_keyframe: the model has no keyframe {keyframe!r}"
        )
    sensor = lambda name: int(  # noqa: E731
        model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)]
    )
    return BipedScene(
        model=model,
        spec=spec,
        name=str(spec.modelname),
        joint_names=tuple(model.joint(j).name for j in joints),
        qpos_index=np.array([model.jnt_qposadr[j] for j in joints], dtype=np.int64),
        dof_index=np.array([model.jnt_dofadr[j] for j in joints], dtype=np.int64),
        free_qpos=int(model.jnt_qposadr[free_joint]),
        free_dof=int(model.jnt_dofadr[free_joint]),
        trunk_body=int(model.jnt_bodyid[free_joint]),
        imu_site=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site)),
        keyframe_id=int(key_id),
        initial_ctrl=np.array(model.key_ctrl[key_id], dtype=float),
        ctrl_range=np.array(model.actuator_ctrlrange, dtype=float),
        accel_address=sensor(ACCEL_SENSOR),
        gyro_address=sensor(GYRO_SENSOR),
        source=model_source(path),
        overrides=overrides,
    )


def _free_joint(model: Any, mujoco: Any) -> int:
    free = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free) != 1:
        raise ValueError(
            f"A biped model needs exactly one free joint (the trunk), found {len(free)}"
        )
    return free[0]


def _actuated_joints(model: Any, mujoco: Any) -> list[int]:
    """Joint index of each actuator, in actuator order; raises ValueError
    for anything but a position actuator on one hinge or slide joint."""
    if model.nu == 0:
        raise ValueError("A biped model needs at least one actuator")
    joints = []
    for a in range(model.nu):
        name = model.actuator(a).name or f"#{a}"
        kp = model.actuator_gainprm[a, 0]
        is_position = (
            model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            and kp > 0
            and np.isclose(model.actuator_biasprm[a, 1], -kp)
        )
        joint = int(model.actuator_trnid[a, 0])
        # int(): NumPy integers are not found in a tuple of MuJoCo enums.
        if not is_position or int(model.jnt_type[joint]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(
                f"Actuator {name}: only position actuators on hinge or slide "
                "joints are supported (commands are joint targets)"
            )
        joints.append(joint)
    return joints


def model_source(path: Path) -> ModelSource:
    """``ModelSource`` for ``path``; commit and dirty are None outside git or
    when git is unavailable."""
    directory = path.resolve().parent

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), *arguments],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", ".") if commit else None
    return ModelSource(
        str(path.resolve()), commit, None if status is None else bool(status)
    )
