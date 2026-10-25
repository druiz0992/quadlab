"""Strict, layered JSON settings for MuJoCo biped runs.

Layers merge exactly like drone configs (``extends`` parents first, objects
key by key, unknown keys rejected; see simulation/gazebo/config.py, whose
generic helpers are reused here). The schema is the biped's own: the
multirotor sections do not apply to a body that MuJoCo simulates. Every
value must have its default's type (booleans, strings, finite numbers,
lists of finite numbers of the same length); ranges are checked per key.

Timing: every rate (control, recording, each sensor) must give a period
that is a whole number of physics steps, so tasks run on an exact
simulation-time grid (integer nanoseconds) and never drift. A rate that
does not fit raises ValueError instead of being rounded.

``biped.model_path`` names an MJCF file: ``builtin:test_biped`` (RobotLab's
own test biped, simulation/mujoco/assets/) or a path, ``~`` expanded and
relative to the working directory. External models are not copied into
this repository (the Microduck meshes are CC BY-NC-SA; see
docs/9_bipeds.md).
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ..gazebo.config import (
    DEFAULTS as DRONE_DEFAULTS,
    _expand_layers,
    merge,
    number,
    stream_timing,
)
from ..wheeled.config import MUJOCO_KINDS
from ...control.legged import CONTROL_STATE_SOURCES

BUILTIN_PREFIX = "builtin:"
ASSETS_DIR = Path(__file__).parent / "assets"
BUILTIN_MODELS = {"test_biped": ASSETS_DIR / "test_biped.xml"}

# Sensor streams a biped publishes on the sensor bus (sensors/bus/__init__.py).
BIPED_STREAMS = ("imu", "joint_encoders")

BIPED_DEFAULTS: dict[str, Any] = {
    "robot": {"kind": "biped"},
    "biped": {
        "model_path": BUILTIN_PREFIX + "test_biped",
        # Keyframe (qpos and ctrl) the run starts from; its ctrl targets
        # are held while no controller is configured.
        "initial_keyframe": "STAND",
        # MJCF site whose frame is the IMU's sensor axes.
        "imu_site": "imu",
        # Add a z = 0 ground plane when the model's world body has none.
        "add_floor": True,
        # Physics step (s); replaces the model's own timestep (reported).
        "step_s": 0.002,
        "control_rate_hz": 50.0,
        # telemetry.jsonl rate; the initial and final states are always kept.
        "record_rate_hz": 50.0,
        # --keyboard walking (control/legged_keyboard.py): command maxima and
        # ramp (fraction of the maximum per second of key hold).
        "keyboard": {
            "max_forward_mps": 0.3,
            "max_left_mps": 0.2,
            "max_yaw_rate_rad_s": 1.5,
            "ramp_per_s": 2.0,
        },
    },
    "sensors": {
        "imu": {
            "enabled": True,
            "rate_hz": 250.0,
            # Same Allan-variance-style model as the drone IMU
            # (sensors/imu/errors.py); ideal by default.
            "error_model": deepcopy(DRONE_DEFAULTS["sensors"]["imu"]["error_model"]),
        },
        # Servo position encoders (sensors/joint_encoders.py); the defaults
        # follow the Dynamixel XL330's 4096 counts per revolution and
        # 0.229 rpm velocity unit.
        "joint_encoders": {
            "enabled": True,
            "rate_hz": 50.0,
            "seed": 0,
            "counts_per_revolution": 4096,
            "velocity_resolution_rad_s": 0.229 * 2 * 3.141592653589793 / 60,
            "position_noise_std_rad": 0.0,
        },
    },
    "sensor_timing": {
        "seed": 0,
        "reorder_window_s": 0.0,
        "imu": stream_timing(),
        "joint_encoders": stream_timing(),
    },
    # Joint-level controller plug-in (control/legged.py): empty holds the
    # start keyframe (built-in ``hold``). ``control_state_source`` ``truth``
    # also hands the controller the trunk's true pose, ``sensors`` does not.
    "algorithms": {
        "controller": "",
        "controller_options": {},
        "control_state_source": "truth",
    },
    "health": {
        # Fallen: trunk tilt above this, or trunk height below this fraction
        # of its initial height (truth, evaluation only).
        "fall_tilt_deg": 60.0,
        "fall_height_fraction": 0.5,
    },
}

ConfigPaths = str | Path | Sequence[str | Path] | None


@dataclass(frozen=True)
class BipedConfig:
    """Validated biped settings and where they came from.

    ``data`` follows ``BIPED_DEFAULTS``; ``model_path`` is the resolved
    MJCF file; ``sources`` lists the layer files in merge order.
    """

    data: dict[str, Any]
    model_path: Path
    sources: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        """Robot family (``biped``)."""
        return str(self.data["robot"]["kind"])

    @property
    def content_hash(self) -> str:
        """SHA-256 of the merged settings (summary.json provenance)."""
        return hashlib.sha256(
            json.dumps(self.data, sort_keys=True).encode()
        ).hexdigest()


def load_biped_config(paths: ConfigPaths = None) -> BipedConfig:
    """Built-in biped defaults overridden by zero or more layered JSON files.

    Raises ValueError for unknown keys, wrong types, rates that are not a
    whole number of physics steps, a non-biped ``robot.kind``, or a
    missing model file.
    """
    layers = [layer for path in _as_paths(paths) for layer in _expand_layers(path)]
    data = deepcopy(BIPED_DEFAULTS)
    for source, layer in layers:
        data = merge(data, layer, f"{source}: config")
    check_types(data, BIPED_DEFAULTS)
    validate_biped(data)
    return BipedConfig(
        data,
        resolve_model_path(data["biped"]["model_path"]),
        tuple(source for source, _ in layers),
    )


def check_types(value: Any, template: Any, path: str = "config") -> None:
    """Raise ValueError unless ``value`` has ``template``'s structure and
    types (see module docs). Keys were already checked by ``merge``."""
    if isinstance(template, dict):
        for key, default in template.items():
            check_types(value[key], default, f"{path}.{key}")
    elif isinstance(template, bool):
        if type(value) is not bool:
            raise ValueError(f"{path} must be boolean")
    elif isinstance(template, str):
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
    elif isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            raise ValueError(f"{path} must be a list of {len(template)} numbers")
        for item in value:
            number(item, path)
    else:
        number(value, path)


def validate_biped(data: dict[str, Any]) -> None:
    """Value checks beyond types (see module docs); raises ValueError."""
    if data["robot"]["kind"] not in MUJOCO_KINDS:
        raise ValueError(
            f"robot.kind must be one of {', '.join(MUJOCO_KINDS)} for a MuJoCo run"
        )
    biped = data["biped"]
    number(biped["step_s"], "biped.step_s", positive=True)
    step_ns = round(biped["step_s"] * 1e9)
    if step_ns < 1:
        raise ValueError("biped.step_s must be at least 1 ns")
    for key in ("control_rate_hz", "record_rate_hz"):
        period_steps(biped[key], step_ns, f"biped.{key}")
    for kind in BIPED_STREAMS:
        period_steps(data["sensors"][kind]["rate_hz"], step_ns, f"sensors.{kind}")
    for key, value in biped["keyboard"].items():
        number(value, f"biped.keyboard.{key}", positive=True)
    encoders = data["sensors"]["joint_encoders"]
    count = encoders["counts_per_revolution"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            "sensors.joint_encoders.counts_per_revolution must be a positive integer"
        )
    for key in ("velocity_resolution_rad_s", "position_noise_std_rad"):
        number(encoders[key], f"sensors.joint_encoders.{key}", nonnegative=True)
    for name, seed in (
        ("sensors.imu.error_model.seed", data["sensors"]["imu"]["error_model"]["seed"]),
        ("sensors.joint_encoders.seed", encoders["seed"]),
        ("sensor_timing.seed", data["sensor_timing"]["seed"]),
    ):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for kind in BIPED_STREAMS:
        for key, value in data["sensor_timing"][kind].items():
            if key != "timestamp_offset_s":
                number(value, f"sensor_timing.{kind}.{key}", nonnegative=True)
    number(
        data["sensor_timing"]["reorder_window_s"],
        "sensor_timing.reorder_window_s",
        nonnegative=True,
    )
    imu_errors = data["sensors"]["imu"]["error_model"]
    for key, values in imu_errors.items():
        if isinstance(values, list) and any(v < 0 for v in values):
            raise ValueError(f"sensors.imu.error_model.{key} must be nonnegative")
    source = data["algorithms"]["control_state_source"]
    if source not in CONTROL_STATE_SOURCES:
        raise ValueError(
            "algorithms.control_state_source must be one of "
            + ", ".join(CONTROL_STATE_SOURCES)
        )
    health = data["health"]
    number(health["fall_tilt_deg"], "health.fall_tilt_deg", positive=True)
    if not 0 < health["fall_height_fraction"] < 1:
        raise ValueError("health.fall_height_fraction must be in (0, 1)")


def period_steps(rate_hz: float, step_ns: int, name: str) -> int:
    """Physics steps per period of ``rate_hz``; raises ValueError unless the
    period is a whole number (>= 1) of ``step_ns`` steps."""
    number(rate_hz, f"{name} rate_hz", positive=True)
    period_ns = 1e9 / rate_hz
    steps = round(period_ns / step_ns)
    if steps < 1 or abs(steps * step_ns - period_ns) > 1e-6 * period_ns:
        raise ValueError(
            f"{name}: {rate_hz} Hz is not a whole number of "
            f"{step_ns * 1e-9:g} s physics steps"
        )
    return steps


def resolve_model_path(spec: str) -> Path:
    """MJCF file for ``biped.model_path`` (see module docs); raises
    ValueError when a built-in name is unknown or the file is missing."""
    if spec.startswith(BUILTIN_PREFIX):
        name = spec[len(BUILTIN_PREFIX) :]
        if name not in BUILTIN_MODELS:
            raise ValueError(
                f"biped.model_path: unknown built-in {name!r} "
                f"(choose {', '.join(BUILTIN_MODELS)})"
            )
        return BUILTIN_MODELS[name]
    path = Path(spec).expanduser()
    if not path.is_file():
        raise ValueError(
            f"biped.model_path: {path} not found. For the Microduck, clone "
            "https://github.com/pollen-robotics/microduck_rl beside this "
            "repository or point biped.model_path at its scene.xml."
        )
    return path


def _as_paths(paths: ConfigPaths) -> list[Path]:
    """Accept no file, one path, or an ordered sequence of paths."""
    if not paths:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]
