"""Compose a wheeled-robot mission (rover, car, tractor) in Gazebo.

Builds every part before Gazebo starts, then runs ``WheeledMissionLoop``.
Shared with multirotor missions: world generation, Gazebo process and
lockstep stepping, message ingress and ordering, the sensor pipeline and
estimator plug-ins, evaluation, and the health verdict. Specific to
wheeled robots: the Python wheel model (``simulation/wheeled/``), wheel
encoders, the wheeled controller contract (``control/wheeled.py``), the
driving keyboard, and the Rerun layout.

The controller is ``algorithms.controller`` when set (a
``WheeledController`` class or a built-in alias, ``BUILTIN_CONTROLLERS``),
else the keyboard driver with ``--keyboard``, else none (the vehicle stays
parked with zero commands). Option conflicts raise ValueError before Gazebo
is checked for or any window opens.
"""

import os
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ...control.wheeled import WheeledController
from ...control.wheeled_keyboard import DriverInput, KeyboardDriver
from ...core.algorithms import AlgorithmContext
from ...core.track import Track
from ...core.wheeled_vehicle import WheeledProperties
from ...evaluation.laps import LapTimer
from ...evaluation.health import HealthThresholds, RunHealth
from ...sensors.wheel_encoders import WheelEncoderModel
from ...simulation.gazebo.config import DroneConfig
from ...simulation.gazebo.ingress import (
    OPPONENT_KIND_PREFIX,
    LockstepOrderer,
    MessageIngress,
    gz_sim_command,
    subscriptions,
)
from ...simulation.gazebo.wheeled_vehicle import (
    OPPONENT_COMMAND_TOPIC,
    OPPONENT_ODOMETRY_TOPIC,
    TRAILER_JOINT_TOPIC,
    opponent_name,
)
from .opponents import (
    KinematicOpponent,
    Opponent,
    build_opponent_controller,
    opponent_options,
)
from ...simulation.gazebo.odometry import OdometryVelocityFix
from ...simulation.gazebo.plugins import lockstep_barrier
from ...simulation.gazebo.world import world_elevation_m
from ...simulation.powertrain import Powertrain
from ...simulation.wheeled.ground import GroundModel
from ...simulation.wheeled.model import WheeledVehicleModel
from ..harness.algorithm_debug import AlgorithmDebugLog
from ..harness.algorithms import build_algorithm_stack, load_class
from ...simulation.gazebo.world import reference_from_world
from ..common.mission import MissionRunner
from ..harness.run_evaluation import RunEvaluation
from ..common.run_setup import prepare_world
from ..common.sensor_pipeline import SensorPipeline, TruthSnapshot
from .wheeled_mission import WheeledMissionLoop, WheeledParts

BUILTIN_CONTROLLERS = {
    "keyboard": "robotlab.control.wheeled_keyboard:KeyboardDriver",
    "example": "robotlab.control.examples.speed_yaw_wheeled_controller:SpeedYawWheeledController",
    "track_follower": "robotlab.control.track_follower:TrackFollower",
}


def validate_wheeled_run(
    args: Any, config: DroneConfig, has_command_source: bool
) -> None:
    """Reject options a wheeled robot does not support (ValueError)."""
    settings = config.data
    kind = config.kind
    if args.world:
        raise ValueError(
            f"{kind}: needs a generated world (the wheel model reads the scene); omit --world"
        )
    if args.control_mode != "velocity":
        raise ValueError(f"{kind}: --control-mode applies to multirotors only")
    if has_command_source:
        raise ValueError(f"{kind}: programmatic command sources are multirotor-only")
    if args.keyboard and args.edit:
        raise ValueError("--keyboard requires mission mode; omit --edit")
    if settings["randomization"]["enabled"]:
        raise ValueError(
            f"{kind}: randomization is the multirotor section; "
            "use ground_vehicle.randomization"
        )
    if settings["mission"]["type"] != "none":
        raise ValueError(
            f"{kind}: mission references are multirotor-only; "
            "use ground_vehicle.route for a reference path"
        )
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Duration must be positive")


def build_wheeled_controller(
    settings: dict[str, Any], context: AlgorithmContext, keyboard: bool
) -> Any | None:
    """The run's ``WheeledController`` (see module docs), or None."""
    algorithms = settings["algorithms"]
    spec = algorithms["controller"]
    if not spec and not keyboard:
        return None
    if not spec:
        options = {
            "max_speed_mps": settings["ground_vehicle"]["keyboard"]["max_speed_mps"]
        }
        return KeyboardDriver(context, options)
    controller = load_class(spec, BUILTIN_CONTROLLERS)(
        context, algorithms["controller_options"]
    )
    if not isinstance(controller, WheeledController):
        raise ValueError(f"{spec} does not implement WheeledController (update)")
    return controller


def run_wheeled(
    args: Any, config: DroneConfig, has_command_source: bool = False
) -> Any:
    """Generate, edit, or drive one wheeled-robot mission (see module docs)."""
    validate_wheeled_run(args, config, has_command_source)
    prepared = prepare_world(args, config, rotor_mode=False)
    if args.generate_only:
        print(prepared.world)
        return None
    from ..common.gazebo_helpers import edit_world, require_gz

    if args.edit:
        if args.headless:
            raise ValueError("--edit needs a desktop window; omit --headless")
        require_gz()
        edit_world(prepared.world)
        return None
    require_gz()
    try:
        from gz.transport13 import Node

        from ...simulation.gazebo.body_drag import BurstImpulseWrench
        from ...visualization.wheeled_rerun import WheeledRerunRecorder
    except ImportError as error:
        raise ValueError(
            f"{error}. Install .[gazebo] and source the Gazebo/ROS environment."
        )
    window = None
    try:
        if args.keyboard:
            from ...visualization.driving_keyboard import DrivingKeyboardWindow

            assert config.wheeled is not None
            window = DrivingKeyboardWindow(
                f"{config.kind} driving", config.wheeled.steering_layout
            )
        return _drive(
            args,
            config,
            prepared,
            window,
            Node,
            BurstImpulseWrench,
            WheeledRerunRecorder,
        )
    finally:
        if window is not None:
            window.close()


def _drive(args, config, prepared, window, Node, Wrench, Recorder):
    from ..common.gazebo_helpers import print_health, scheduled_periods_ns, step_ns
    from ...simulation.gazebo.stepping import TrackedNode, steps_per_period

    # Gazebo and the wheel model simulate the true (possibly randomized)
    # vehicle; algorithms keep the nominal one (simulation/randomization.py).
    settings, nominal = config.data, config.wheeled
    vehicle = prepared.true_config.wheeled
    if nominal is None or vehicle is None:
        raise ValueError("wheeled run without wheeled properties")
    output, scene, terrain, world = (
        prepared.output,
        prepared.scene,
        prepared.terrain,
        prepared.world,
    )
    os.environ["GZ_PARTITION"] = "robotlab_" + uuid.uuid4().hex
    os.environ.setdefault("GZ_IP", "127.0.0.1")
    recorder = Recorder(
        world,
        output,
        args.headless and not (args.rerun or args.keyboard),
        vehicle,
        scene,
    )
    node = TrackedNode(Node())
    ingress = MessageIngress()
    truth = TruthSnapshot()
    pipeline = _sensor_pipeline(
        settings, config, scene, terrain, world, recorder, truth
    )
    for warning in pipeline.calibration_warnings:
        print(f"Warning: {warning}")
    debug = AlgorithmDebugLog(
        output / "algorithm-debug.jsonl", rerun=recorder.log_algorithm_debug
    )
    stack = build_algorithm_stack(
        settings,
        config.properties,
        reference_from_world(world),
        pipeline.bus_kinds(),
        rotor_mode=False,
        log=debug,
        wheeled=nominal,
        track=reference_track(settings, scene),
    )
    controller = build_wheeled_controller(settings, stack.context, args.keyboard)
    ground = GroundModel.from_scene(scene, terrain)
    steps = steps_per_period(
        1 / settings["control"]["odometry_rate_hz"], settings["simulation"]["step_s"]
    )
    ego_wrench = Wrench(node, f"{vehicle.model_name}::base_link", steps)
    opponents = _opponents(
        settings, vehicle, stack.context, ground, Wrench, node, ego_wrench.publisher
    )
    names = frozenset(o.name for o in opponents)
    implement, lift_publisher = _implement(settings, ground, node)
    trailer = None
    if settings["ground_vehicle"]["trailer"]["enabled"]:
        from .trailer import Trailer

        trailer = Trailer(
            settings["ground_vehicle"]["trailer"],
            vehicle,
            ground,
            Wrench(node, f"{vehicle.model_name}::trailer", steps, ego_wrench.publisher),
        )
    keyboard = settings["ground_vehicle"]["keyboard"]
    parts = WheeledParts(
        args=args,
        config=config,
        output=output,
        command=gz_sim_command(
            world,
            paused=True,
            headless=args.headless,
            render_sensors=config.render_sensors,
        ),
        node=node,
        ingress=ingress,
        orderer=LockstepOrderer(
            ingress,
            {
                **scheduled_periods_ns(settings, pipeline),
                # Hitch joint states: every physics step.
                **(
                    {"trailer": float(step_ns(settings))} if trailer is not None else {}
                ),
                # Wait for each opponent's odometry at its rate.
                **{
                    OPPONENT_KIND_PREFIX + name: _opponent_period_ns(settings)
                    for name in names
                },
            },
            step_ns(settings),
            publisher_kinds=frozenset(OPPONENT_KIND_PREFIX + name for name in names),
        ),
        barrier=lockstep_barrier(settings),
        recorder=recorder,
        pipeline=pipeline,
        stack=stack,
        controller=controller,
        model=WheeledVehicleModel(
            vehicle, ground, vehicles=1 + sum(o.dynamic for o in opponents)
        ),
        wrench=ego_wrench,
        encoders=WheelEncoderModel(
            settings["sensors"]["wheel_encoders"], vehicle.wheel_count
        ),
        health=RunHealth(
            HealthThresholds.from_settings(settings["health"]),
            ground_height_m=lambda x, y: float(ground.height(np.array(x), np.array(y))),
            ground_half_m=scene.get("ground_size_m", 100.0) / 2,
            weight_n=None,
            own_models=frozenset({vehicle.model_name}),
            wheeled=True,
            vehicles=names,
        ),
        evaluation=RunEvaluation(
            settings,
            vehicle.mass_kg * 9.81,
            MissionRunner(settings["mission"]),
            recorder.log_vector,
            pipeline.imu_bias.current,
        ),
        truth=truth,
        odometry_fix=OdometryVelocityFix(),
        debug=debug,
        keyboard=window,
        laps=LapTimer(stack.context.track) if stack.context.track else None,
        opponents=tuple(opponents),
        implement=implement,
        lift_publisher=lift_publisher,
        trailer=trailer,
        randomization_draws=prepared.randomization_draws,
        driver_input=(
            DriverInput(keyboard["throttle_rate_per_s"], keyboard["steer_rate_per_s"])
            if window is not None
            else None
        ),
    )
    # Poses only move Rerun obstacles; without moving obstacles they cost
    # ~0.2 ms per period for nothing.
    topics = [
        topic
        for topic in subscriptions(settings["sensors"], False)
        if topic[2] != "poses" or scene.get("moving_obstacles")
    ]
    ingress.subscribe(
        node,
        topics
        + [
            (
                "odometry_pb2.Odometry",
                OPPONENT_ODOMETRY_TOPIC.format(name=o.name),
                OPPONENT_KIND_PREFIX + o.name,
            )
            for o in opponents
        ]
        + (
            [("model_pb2.Model", TRAILER_JOINT_TOPIC, "trailer")]
            if trailer is not None
            else []
        ),
    )
    loop = WheeledMissionLoop(parts)
    try:
        summary = loop.run()
    except BaseException:
        if loop.summary is not None:
            print(f"Results: {output}")
            print_health(loop.summary["health"])
        raise
    print(f"Results: {output}")
    print(
        f"Messages: {summary['messages']}; max speed {summary['max_speed_mps']:.2f} m/s"
    )
    print_health(summary["health"])
    return summary


def reference_track(settings: dict[str, Any], scene: dict[str, Any]) -> Track | None:
    """``ground_vehicle.route`` as a track when it has waypoints, else the
    scene's track (or None)."""
    route = settings["ground_vehicle"]["route"]
    if route["waypoints_m"]:
        return Track.from_route(route["waypoints_m"], route["width_m"])
    return Track.from_scene(scene)


def _implement(settings, ground, node):
    """The hitch implement model and its lift publisher, or (None, None)."""
    section = settings["ground_vehicle"]["implement"]
    if not section["enabled"]:
        return None, None
    from gz.msgs10.double_pb2 import Double

    from ...simulation.gazebo.wheeled_vehicle import IMPLEMENT_LIFT_TOPIC
    from ...simulation.wheeled.implement import ImplementModel

    return ImplementModel(section, ground), node.advertise(IMPLEMENT_LIFT_TOPIC, Double)


def _opponents(settings, vehicle, context, ground, Wrench, node, publisher):
    """Runtime parts of every configured opponent (experiments/wheeled/opponents.py);
    their wrenches share the ego's wrench ``publisher``."""
    from ...simulation.gazebo.stepping import steps_per_period

    section = settings["ground_vehicle"]["opponents"]
    if section["mode"] == "kinematic":
        return _kinematic_opponents(settings, vehicle, context, ground, node)
    control = settings["control"]
    period_s = 1 / control["odometry_rate_hz"]
    steps = steps_per_period(period_s, settings["simulation"]["step_s"])
    opponents: list[Any] = []
    for index in range(section["count"]):
        name = opponent_name(vehicle, index + 1)
        opponents.append(
            Opponent(
                name=name,
                index=index + 1,
                controller=build_opponent_controller(
                    section["controller"],
                    BUILTIN_CONTROLLERS,
                    context,
                    opponent_options(settings, index),
                ),
                wrench=Wrench(node, f"{name}::base_link", steps, publisher),
                laps=LapTimer(context.track) if context.track else None,
                control_period_ns=round(1e9 / control["control_rate_hz"]),
                odometry_fix=OdometryVelocityFix(),
            )
        )
    return opponents


def _opponent_period_ns(settings: dict) -> float:
    """Odometry period of the opponents (ns): every ego period when
    dynamic, the update period when kinematic."""
    section = settings["ground_vehicle"]["opponents"]
    if section["mode"] == "kinematic":
        return 1e9 / section["update_rate_hz"]
    return 1e9 / settings["control"]["odometry_rate_hz"]


def _kinematic_opponents(
    settings: dict,
    vehicle: WheeledProperties,
    context: AlgorithmContext,
    ground: GroundModel,
    node: Any,
) -> list[Any]:
    """Kinematic opponents (experiments/wheeled/opponents.py), each with its own
    twist publisher to Gazebo's VelocityControl."""
    from ...control.track_follower import KinematicTrackDriver

    section = settings["ground_vehicle"]["opponents"]
    hubs = vehicle.hub_positions_m()[:, :2]
    opponents: list[Any] = []
    for index in range(section["count"]):
        name = opponent_name(vehicle, index + 1)
        opponents.append(
            KinematicOpponent(
                name=name,
                driver=KinematicTrackDriver(context, opponent_options(settings, index)),
                publish=_twist_publisher(
                    node, OPPONENT_COMMAND_TOPIC.format(name=name)
                ),
                height_m=ground.height,
                hub_xy_m=hubs,
                laps=LapTimer(context.track) if context.track else None,
                odometry_fix=OdometryVelocityFix(),
            )
        )
    return opponents


def _twist_publisher(node: Any, topic: str) -> Callable[[dict], None]:
    """``publish(fields)`` sending ``dict(linear=(x, y, z), angular=(x, y,
    z))`` as a gz Twist on ``topic``; raises RuntimeError if it fails."""
    from gz.msgs10.twist_pb2 import Twist

    publisher = node.advertise(topic, Twist)

    def publish(fields: dict) -> None:
        message = Twist()
        for part in ("linear", "angular"):
            target = getattr(message, part)
            target.x, target.y, target.z = fields[part]
        if not publisher.publish(message):
            raise RuntimeError(f"Unable to publish {topic}")

    return publish


def _sensor_pipeline(settings, config, scene, terrain, world, recorder, truth):
    """The shared sensor pipeline; multirotor-only models stay disabled
    (validated in simulation/wheeled/config.py)."""
    powertrain = Powertrain(
        config.properties,
        settings["battery"],
        settings["aerodynamics"]["ground_effect"],
        settings["motors"]["faults"],
        aerodynamics=settings["aerodynamics"],
        terrain=terrain,
        ambient_c=settings["environment"]["ambient_temperature_c"],
        forward_figure_of_merit=settings["motors"]["forward_figure_of_merit"],
    )
    return SensorPipeline(
        settings,
        scene=scene,
        terrain=terrain,
        elevation_m=world_elevation_m(world),
        propeller_radius_m=config.properties.propeller_diameter / 2,
        recorder=recorder,
        powertrain=powertrain,
        current_wind=lambda: np.zeros(3),
        truth=truth,
        true_imu_mount=Rotation.from_euler(
            "xyz", settings["sensors"]["imu"]["mount"]["rotation_deg"], degrees=True
        ),
    )
