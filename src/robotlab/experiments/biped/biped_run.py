"""Compose a biped mission in MuJoCo (``--robot biped``).

Bipeds are simulated only in MuJoCo (docs/9_bipeds.md): the settings come
from simulation/mujoco/config.py, not the drone config. The controller is
``algorithms.controller`` (a ``LeggedController`` class or a built-in
alias, ``BUILTIN_CONTROLLERS``), else ``hold`` (the start keyframe). Shared
with the
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

from ...control.legged import LeggedController
from ...core.algorithms import AlgorithmContext, SensorDescription
from ...core.measurements import GpsReading
from ...core.vehicle import DroneProperties
from ...evaluation.biped_health import BipedHealth
from ...sensors.bus import SensorBus
from ...sensors.imu_errors import ImuErrorModel
from ...sensors.joint_encoders import JointEncoderModel
from ...simulation.mujoco.config import BIPED_STREAMS, BipedConfig, load_biped_config
from ...sensors.bus import STREAMS
from ...simulation.gazebo.world import WORLD_ELEVATION_M
from ..harness.algorithms import load_class
from .biped_loop import BipedLoop, BipedParts

BUILTIN_CONTROLLERS = {
    "hold": "robotlab.control.legged_hold:HoldPose",
    "template": "robotlab.control.legged_template:TemplateLeggedController",
    "onnx_policy": "robotlab.control.onnx_policy:OnnxPolicyController",
}
# The generated Gazebo worlds' origin (simulation/gazebo/world.py), so GPS
# references agree across simulators; bipeds have no GPS yet.
GPS_REFERENCE = GpsReading(40.4168, -3.7038, WORLD_ELEVATION_M)

# Gazebo-only options and the value that means "not used".
_GAZEBO_ONLY = (
    ("world", None, "--world"),
    ("edit", False, "--edit"),
    ("render_sensors", False, "--render-sensors"),
)


def validate_biped_run(args: Any, has_command_source: bool) -> None:
    """Reject options a biped run does not support (ValueError)."""
    from ..run import GazeboConfig

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
    from ...simulation.mujoco.scene import build_scene

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
        _open_frontends(args, config, scene, loop, output)
        summary = loop.run()
    except BaseException:
        if loop.summary is not None:
            print(f"Results: {output}")
            _print_health(loop.summary["health"])
        raise
    finally:
        for frontend in (loop.parts.operator, loop.parts.recorder):
            if frontend is not None:
                frontend.close()
    timing = summary["timing"]
    print(f"Results: {output}")
    print(
        f"Simulated {timing['sim_time_s']:.2f} s in {timing['wall_time_s']:.2f} s "
        f"({timing['realtime_factor']:.0f}x real time)"
    )
    _print_health(summary["health"])
    return summary


def _open_frontends(
    args: Any, config: BipedConfig, scene: Any, loop: BipedLoop, output: Path
) -> None:
    """Keyboard window (``--keyboard``) and the live Rerun viewer, recorded
    to ``biped.rrd`` (``--keyboard`` or ``--rerun``, as for Gazebo runs)."""
    if not (args.keyboard or args.rerun):
        return
    try:
        from ...visualization.biped_rerun import BipedRerunRecorder
    except ImportError as error:
        raise ValueError(f"{error}. Install the gazebo extra (Rerun).") from error
    loop.parts.recorder = BipedRerunRecorder(
        scene.model, output, headless=False, title=f"{scene.name} (MuJoCo)"
    )
    if args.keyboard:
        from ...control.legged_keyboard import WalkInput
        from .biped_keyboard import open_keyboard_operator

        loop.parts.operator = open_keyboard_operator(
            WalkInput.from_settings(config.data["biped"]["keyboard"])
        )


def biped_context(settings: dict[str, Any], scene: Any) -> AlgorithmContext:
    """Context for biped plug-ins: ``legged`` properties and the enabled
    sensors' frames, rates, and mounts (the IMU at its site)."""
    properties = scene.properties(settings["biped"]["control_rate_hz"])
    sensors = settings["sensors"]
    descriptions = {}
    for kind in BIPED_STREAMS:
        if not sensors[kind]["enabled"]:
            continue
        position, quaternion = (
            scene.imu_mount() if kind == "imu" else ((0.0,) * 3, (0.0, 0.0, 0.0, 1.0))
        )
        descriptions[kind] = SensorDescription(
            kind, STREAMS[kind][1], sensors[kind]["rate_hz"], position, quaternion
        )
    return AlgorithmContext(
        GPS_REFERENCE, DroneProperties(), descriptions, legged=properties
    )


def build_legged_controller(
    settings: dict[str, Any], context: AlgorithmContext
) -> tuple[str, Any]:
    """``(spec, controller)`` for ``algorithms.controller`` (empty: ``hold``);
    raises ValueError when it cannot be built or is not a LeggedController."""
    algorithms = settings["algorithms"]
    spec = algorithms["controller"] or "hold"
    controller = load_class(spec, BUILTIN_CONTROLLERS)(
        context, algorithms["controller_options"]
    )
    if not isinstance(controller, LeggedController):
        raise ValueError(f"{spec} does not implement LeggedController (update)")
    return spec, controller


def _parts(
    config: BipedConfig, scene: Any, output: Path, duration_s: float
) -> BipedParts:
    settings = config.data
    sensors = settings["sensors"]
    kinds = [kind for kind in BIPED_STREAMS if sensors[kind]["enabled"]]
    context = biped_context(settings, scene)
    spec, controller = build_legged_controller(settings, context)
    assert context.legged is not None
    return BipedParts(
        config=config,
        scene=scene,
        properties=context.legged,
        controller=controller,
        controller_spec=spec,
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
    from ..common.gazebo_helpers import print_health

    print_health(health)
