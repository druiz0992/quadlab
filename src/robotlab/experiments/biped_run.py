"""Compose a biped mission in MuJoCo (``--robot biped``).

Bipeds are simulated only in MuJoCo (docs/9_bipeds.md): the settings come
from simulation/mujoco/config.py, not the drone config. Shared with the
Gazebo robots: the sensor bus, the IMU error model, the health-verdict
shape, and the output conventions (``summary.json``, ``telemetry.jsonl``,
``commands.jsonl``). Options that only mean something for Gazebo raise
ValueError before MuJoCo is imported.

``--generate-only`` writes the edited scene as ``scene.xml`` (an MJCF file
that loads on its own) and does not step it.
"""

import math
from pathlib import Path
from typing import Any

from ..evaluation.biped_health import BipedHealth
from ..sensors.bus import SensorBus
from ..sensors.imu_errors import ImuErrorModel
from ..sensors.joint_encoders import JointEncoderModel
from ..simulation.mujoco.config import BIPED_STREAMS, BipedConfig, load_biped_config
from .biped_loop import BipedLoop, BipedParts

# Gazebo-only options and the value that means "not used".
_GAZEBO_ONLY = (
    ("world", None, "--world"),
    ("edit", False, "--edit"),
    ("ekf", False, "--ekf"),
    ("render_sensors", False, "--render-sensors"),
    ("rerun", False, "--rerun"),
    ("keyboard", False, "--keyboard"),
)


def validate_biped_run(args: Any, has_command_source: bool) -> None:
    """Reject options a biped run does not support (ValueError)."""
    from .gazebo import GazeboConfig

    for attribute, unused, option in _GAZEBO_ONLY:
        if getattr(args, attribute) != unused:
            raise ValueError(f"biped: {option} is not supported for MuJoCo bipeds")
    if args.control_mode != "velocity":
        raise ValueError("biped: --control-mode applies to multirotors only")
    if args.scene != GazeboConfig.scene:
        raise ValueError("biped: --scene is a Gazebo scene; set biped.model_path")
    if tuple(args.target) != tuple(GazeboConfig.target):
        raise ValueError("biped: --target applies to multirotor missions only")
    if has_command_source:
        raise ValueError("biped: programmatic command sources are multirotor-only")
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Duration must be positive")


def run_biped(args: Any, has_command_source: bool = False) -> dict[str, Any] | None:
    """Generate or run one biped mission (see module docs); returns the
    summary, or None for ``--generate-only``. Raises ValueError for invalid
    options or settings and RuntimeError when the run fails."""
    validate_biped_run(args, has_command_source)
    config = load_biped_config(args.drone)
    from ..simulation.mujoco.scene import build_scene

    try:
        import mujoco  # noqa: F401
    except ImportError as error:
        raise ValueError(
            f"{error}. Install the mujoco extra: pip install -e .[mujoco]"
        ) from error
    scene = build_scene(config.data, config.model_path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.generate_only:
        path = output / "scene.xml"
        scene.write_xml(path)
        print(path)
        return None
    loop = BipedLoop(_parts(config, scene, output, args.duration))
    try:
        summary = loop.run()
    except BaseException:
        if loop.summary is not None:
            print(f"Results: {output}")
            _print_health(loop.summary["health"])
        raise
    timing = summary["timing"]
    print(f"Results: {output}")
    print(
        f"Simulated {timing['sim_time_s']:.2f} s in {timing['wall_time_s']:.2f} s "
        f"({timing['realtime_factor']:.0f}x real time)"
    )
    _print_health(summary["health"])
    return summary


def _parts(
    config: BipedConfig, scene: Any, output: Path, duration_s: float
) -> BipedParts:
    settings = config.data
    sensors = settings["sensors"]
    kinds = [kind for kind in BIPED_STREAMS if sensors[kind]["enabled"]]
    return BipedParts(
        config=config,
        scene=scene,
        bus=SensorBus(settings["sensor_timing"], kinds),
        imu_errors=ImuErrorModel(
            sensors["imu"]["error_model"], sensors["imu"]["rate_hz"]
        ),
        encoders=JointEncoderModel(sensors["joint_encoders"], scene.joint_names),
        health=BipedHealth(settings["health"]),
        output=output,
        duration_s=duration_s,
    )


def _print_health(health: dict[str, Any]) -> None:
    from .gazebo import _print_health as print_health

    print_health(health)
