"""Compose Gazebo transport, waypoint control, and recording for one drone mission."""

from dataclasses import dataclass
import os
import shutil
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from .takeoff import TakeoffSequence
from .failsafe import BatteryFailsafe
from ..harness.live_ekf import reference_from_world
from ..common.mission import MissionRunner
from ..harness.run_evaluation import RunEvaluation
from ..common.run_setup import prepare_world, validate_run
from ..common.sensor_pipeline import SensorPipeline, TruthSnapshot
from .vehicle_forces import ExternalForces
from .keyboard_flight import keyboard_effects
from .mission_loop import MissionLoop, MissionParts
from ...simulation.gazebo.config import sample_timing_warnings
from ...simulation.gazebo.process import GazeboProcess
from ...simulation.gazebo.stepping import TrackedNode
from ...simulation.gazebo.plugins import lockstep_barrier
from ...simulation.gazebo.ingress import (
    LockstepOrderer,
    MessageIngress,
    gz_sim_command,
    subscriptions,
)
from ...evaluation.health import HealthThresholds, RunHealth
from ..harness.algorithms import build_algorithm_stack
from ..harness.algorithm_debug import AlgorithmDebugLog
from ..harness.replay_log import ReplayRecorder
from ...core.measurements import GpsReading
from ...simulation.gazebo.odometry import OdometryVelocityFix
from ...simulation.moving_obstacles import MovingObstacleCommands
from ...simulation.gazebo.payload import PayloadControl, Winch
from ...simulation.powertrain import Powertrain
from ...simulation.esc import EscModel
from ...simulation.gazebo.environment import gazebo_environment
from ...simulation.gazebo.world import WORLD_ELEVATION_M, world_elevation_m
from ..common.gazebo_helpers import (
    edit_world,
    print_health,
    require_gz,
    scheduled_periods_ns,
    step_ns,
)


def _ground_height(settings, terrain):
    """Ground height (m) under the configured spawn x, y."""
    x, y, _ = settings["simulation"]["spawn_position_m"]
    return 0.0 if terrain is None else float(terrain.height(x, y))


def fly(args, config, open_keyboard, command_source):
    """Generate, edit, or fly one drone mission with the loaded ``config``
    (dispatched from run.py); returns ``summary.json``, or None for
    ``--generate-only`` and ``--edit``."""
    from ...simulation.gazebo.actuators import RotorThrustControl

    settings = config.data

    sensors = settings["sensors"]
    mode = validate_run(args, settings, command_source is not None)
    rotor_mode, lockstep = mode.rotor_mode, mode.lockstep
    barrier = lockstep_barrier(settings)
    prepared = prepare_world(args, config, rotor_mode)
    true_config, randomization_draws = (
        prepared.true_config,
        prepared.randomization_draws,
    )
    output, scene, terrain, world = (
        prepared.output,
        prepared.scene,
        prepared.terrain,
        prepared.world,
    )
    if args.generate_only:
        print(world)
        return
    if args.edit:
        if args.headless:
            raise ValueError("--edit needs a desktop window; omit --headless")
        require_gz()
        edit_world(world)
        return

    # Reject mode conflicts before checking for Gazebo or opening windows
    # (the keyboard window, and the recorder's Rerun viewer).
    powertrain = Powertrain(
        config.properties,
        settings["battery"],
        settings["aerodynamics"]["ground_effect"],
        settings["motors"]["faults"],
        overridable=bool(args.keyboard),
        aerodynamics=settings["aerodynamics"],
        terrain=terrain,
        ambient_c=settings["environment"]["ambient_temperature_c"],
        forward_figure_of_merit=settings["motors"]["forward_figure_of_merit"],
    )
    if powertrain.enabled and not rotor_mode:
        raise ValueError(
            "battery, ground effect, motor faults, translational lift, and vortex "
            "ring state require rotor-thrust or keyboard mode"
        )
    if settings["fixed_wing"]["enabled"] and not rotor_mode:
        raise ValueError("fixed_wing needs rotor-thrust mode")
    require_gz()
    try:
        from ...visualization.gazebo_rerun import GazeboRerunRecorder
        from gz.transport13 import Node
        from gz.msgs10.twist_pb2 import Twist
    except ImportError as error:
        raise ValueError(
            f"{error}. Install .[gazebo] and source the Gazebo/ROS environment."
        )
    keyboard = open_keyboard() if open_keyboard is not None else None
    if keyboard is not None:
        keyboard.pilot_rates = settings["pilot"]

    live_ekf = None
    estimate_reader = None
    if args.ekf:
        from ..harness.live_ekf import LiveEkf

        live_ekf = LiveEkf(settings, reference_from_world(world))
        estimate_reader = live_ekf.estimates.subscribe(capacity=4096)

    # Keep discovery private to this run, without interfering with another simulator.
    os.environ["GZ_PARTITION"] = "robotlab_" + uuid.uuid4().hex
    os.environ.setdefault("GZ_IP", "127.0.0.1")
    recorder = GazeboRerunRecorder(
        world,
        output,
        None if rotor_mode else args.target,
        args.headless and not (args.rerun or args.keyboard),
        properties=config.properties,
        body_size=settings["vehicle"]["body_size_m"],
        optical_flow_mount=(
            sensors["optical_flow"]["mount"]
            if sensors["optical_flow"]["enabled"]
            else None
        ),
    )
    ingress = MessageIngress()
    # Automatic take-off after a ground start (rotor-thrust and keyboard modes).
    takeoff = (
        TakeoffSequence(
            # Take-off altitude is above the take-off point (terrain height).
            _ground_height(settings, terrain) + settings["pilot"]["takeoff_altitude_m"],
            settings["pilot"]["takeoff_climb_mps"],
        )
        if rotor_mode and settings["simulation"]["start_on_ground"]
        else None
    )
    mission = MissionRunner(settings["mission"])
    # Latest truth for measurement synthesis and fault models, never EKF input.
    truth = TruthSnapshot()

    # Tracks advertised command topics for the lockstep start-up barrier.
    node = TrackedNode(Node())
    from ...simulation.gazebo.wind import WindCommands, dynamic_wind

    # Keyboard flight can switch configured wind on and off, which needs the
    # Python wind publisher even for static wind.
    wind_control = (
        WindCommands(node, settings["wind"])
        if dynamic_wind(settings["wind"])
        or (args.keyboard and settings["wind"]["enabled"])
        else None
    )
    esc_settings = settings["motors"]["esc"]
    esc = EscModel(
        esc_settings,
        (config.properties.max_rotor_thrust / config.properties.thrust_coefficient)
        ** 0.5,
        clock_ns=lambda: (
            None if truth.sim_time_s is None else round(truth.sim_time_s * 1e9)
        ),
    )
    actuator = (
        RotorThrustControl(
            node,
            config.properties,
            powertrain if powertrain.enabled else None,
            esc=esc,
        )
        if rotor_mode
        else None
    )
    odometry_fix = OdometryVelocityFix()
    payload = (
        PayloadControl(node, settings["payload"])
        if settings["payload"]["enabled"]
        else None
    )
    tether = settings["payload"]["tether"]
    winch = (
        Winch(node, tether)
        if settings["payload"]["enabled"] and tether["enabled"]
        else None
    )

    # Keyboard flight can spawn moving objects (key 7), so it always has one.
    moving = (
        MovingObstacleCommands(node, scene.get("moving_obstacles", []))
        if scene.get("moving_obstacles") or keyboard is not None
        else None
    )
    body_force = None
    if ExternalForces.needed(settings):
        from ...simulation.gazebo.body_drag import BodyForce

        body_force = BodyForce(node)
    forces = ExternalForces(settings, body_force, recorder)

    def current_wind():
        """Wind the vehicle sees, for airspeed: latest target, else configured."""
        if wind_control is not None:
            return np.asarray(wind_control.velocity_enu_mps)
        wind = settings["wind"]
        return np.asarray(wind["velocity_enu_mps"] if wind["enabled"] else [0.0] * 3)

    pipeline = SensorPipeline(
        settings,
        scene=scene,
        terrain=terrain,
        # Barometer datum: the world's sea-level elevation.
        elevation_m=world_elevation_m(world),
        propeller_radius_m=config.properties.propeller_diameter / 2,
        recorder=recorder,
        powertrain=powertrain,
        current_wind=current_wind,
        truth=truth,
        true_imu_mount=Rotation.from_euler(
            "xyz",
            true_config.data["sensors"]["imu"]["mount"]["rotation_deg"],
            degrees=True,
        ),
    )
    for warning in pipeline.calibration_warnings + sample_timing_warnings(settings):
        print(f"Warning: {warning}")
    failsafe_settings = settings["failsafe"]["low_battery"]
    failsafe = (
        BatteryFailsafe(
            failsafe_settings,
            settings["simulation"]["spawn_position_m"][:2],
            settings["battery"]["cells"],
        )
        if failsafe_settings["enabled"]
        else None
    )

    effects = (
        keyboard_effects(
            keyboard,
            settings,
            node=node,
            recorder=recorder,
            moving=moving,
            payload=payload,
            powertrain=powertrain,
            wind_control=wind_control,
            pipeline=pipeline,
            truth=truth,
            sim_time_s=lambda: truth.sim_time_s or 0.0,
        )
        if keyboard is not None
        else None
    )
    debug = AlgorithmDebugLog(
        output / "algorithm-debug.jsonl", rerun=recorder.log_algorithm_debug
    )
    record_mode = settings["algorithms"]["record_bus"]
    stack = build_algorithm_stack(
        settings,
        config.properties,
        reference_from_world(world) if not args.world else GpsReading(0, 0, 0),
        pipeline.bus_kinds(),
        rotor_mode=rotor_mode,
        log=debug,
        need_bus=record_mode != "off",
    )
    replay = (
        ReplayRecorder(
            output / "replay.jsonl",
            record_mode,
            settings,
            stack.context,
            true_config.properties.mass * 9.81,
        )
        if record_mode != "off"
        else None
    )

    true_biases = pipeline.imu_bias.current

    evaluation = RunEvaluation(
        settings,
        true_config.properties.mass * 9.81,
        mission,
        recorder.log_vector,
        true_biases,
        # Controllers that fly their own route score against it instead.
        tracking_reference=getattr(stack.controller, "tracking_reference", None),
    )
    # Flying past the ground plane's edge means nothing collides below.
    ground_half_m = scene.get("ground_size_m", 100.0) / 2
    # Truth-based verdict (summary.json.health); never feeds the flight.
    health = RunHealth(
        HealthThresholds.from_settings(settings["health"]),
        ground_height_m=lambda x, y: (
            0.0 if terrain is None else float(terrain.height(x, y))
        ),
        ground_half_m=ground_half_m,
        weight_n=true_config.properties.mass * 9.81,
    )
    publisher = None if rotor_mode else node.advertise("/drone/cmd_vel", Twist)
    ingress.subscribe(node, subscriptions(sensors, pipeline.esc_telemetry.enabled))
    command = gz_sim_command(
        world,
        paused=lockstep,
        headless=args.headless,
        render_sensors=config.render_sensors,
    )
    orderer = (
        LockstepOrderer(
            ingress,
            scheduled_periods_ns(settings, pipeline),
            step_ns(settings),
        )
        if lockstep
        else None
    )
    loop = MissionLoop(
        MissionParts(
            args=args,
            config=config,
            true_config=true_config,
            randomization_draws=randomization_draws,
            output=output,
            rotor_mode=rotor_mode,
            lockstep=lockstep,
            barrier=barrier,
            keyboard=keyboard,
            command_source=command_source,
            recorder=recorder,
            node=node,
            ingress=ingress,
            orderer=orderer,
            replay=replay,
            debug=debug,
            command=command,
            wind_control=wind_control,
            current_wind=current_wind,
            powertrain=powertrain,
            esc=esc,
            actuator=actuator,
            publisher=publisher,
            odometry_fix=odometry_fix,
            payload=payload,
            winch=winch,
            moving=moving,
            forces=forces,
            pipeline=pipeline,
            failsafe=failsafe,
            effects=effects,
            stack=stack,
            live_ekf=live_ekf,
            estimate_reader=estimate_reader,
            evaluation=evaluation,
            health=health,
            mission=mission,
            takeoff=takeoff,
            truth=truth,
            terrain=terrain,
            ground_half_m=ground_half_m,
        )
    )
    try:
        summary = loop.run()
    except BaseException:
        # summary.json is written even when the run fails: point at it.
        if loop.summary is not None:
            print(f"Results: {output}")
            print_health(loop.summary["health"])
        raise
    counts, completed = loop.counts, loop.completed
    expected = {"odometry"} | {
        name
        for name in (
            "imu",
            "gps",
            "baro",
            "magnetometer",
            "rangefinder",
            "optical_flow",
            "airspeed",
            "camera",
            "lidar",
        )
        if sensors[name]["enabled"]
    }
    missing = expected - counts.keys()
    if completed and missing:
        raise RuntimeError(
            f'Flight ran but sensors had no data: {sorted(missing)}. See {output / "gazebo.log"}'
        )
    print(f"Results: {output}")
    print(f"Messages: {dict(counts)}; contact samples: {len(loop.collisions)}")
    print_health(summary["health"])
