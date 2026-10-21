"""Compose Gazebo transport, waypoint control, and recording for one mission."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import subprocess
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from ..control.waypoint import waypoint_velocity
from .rotor_loop import RotorControlLoop, TruthState
from .takeoff import TakeoffSequence
from .failsafe import BatteryFailsafe
from .live_ekf import reference_from_world
from .mission import MissionRunner
from .run_evaluation import RunEvaluation
from .algorithms import (
    AlgorithmRunner,
    EstimateStateSource,
    StaleEstimate,
    build_context,
    build_controller,
    build_estimator,
)
from ..core.measurements import CameraFrame, GpsReading, LidarPoints
from ..sensors.bus import STREAMS, SensorBus
from .effect_toggles import ACTIONS, EFFECTS, EffectToggles
from ..simulation.aerodynamics import (
    hover_induced_velocity,
    quadratic_body_drag_world,
)
from ..simulation.gazebo.odometry import OdometryVelocityFix
from ..sensors.camera_effects import CameraEffects
from ..sensors.lidar_effects import LidarEffects
from ..sensors.gazebo import rgb_image
from ..control.timing import DelayLine
from ..simulation.moving_obstacles import MovingObstacleCommands
from ..simulation.gazebo.entities import EntityService
from ..simulation.gazebo.payload import PayloadControl, Winch
from .flying_object import FlyingObjectSpawner
from ..simulation.powertrain import Powertrain
from ..simulation.randomization import randomize_config
from ..simulation.fixed_wing import FixedWingAero, ServoBank
from ..sensors.airspeed import AirspeedSensor
from ..simulation.esc import EscModel
from ..sensors.esc_telemetry import EscTelemetryModel
from ..simulation.gazebo.vehicle import ROTOR_STATE_TOPIC, ROTOR_VELOCITY_SLOWDOWN
from ..simulation.gazebo.environment import gazebo_environment
from ..simulation.gazebo.messages import timestamp, vector
from ..simulation.gazebo.messages import timestamp_ns, with_timestamp_ns
from ..sensors.barometer import BaroErrorModel
from ..sensors.dropout import GpsDropout
from ..sensors.gps_errors import GpsErrorModel
from ..sensors.imu_errors import ImuErrorModel
from ..sensors.imu_faults import ImuFaults
from ..sensors.imu_temperature import ImuTemperatureModel
from ..sensors.gps_receiver import (
    GpsHeading,
    GpsStatusModel,
    GpsVelocityErrors,
    RtkErrorModel,
)
from ..sensors.imu_imperfections import ImuImperfections, ImuSaturation, ImuTiming
from ..sensors.magnetometer import MagnetometerErrorModel
from ..estimation.magnetometer_calibration import MagnetometerCalibration
from ..sensors.magnetic_anomalies import MagneticAnomalies, scene_dipoles
from ..sensors.optical_flow import OpticalFlowModel
from ..sensors.battery_monitor import BatteryMonitor
from ..sensors.multipath import GpsMultipath
from ..sensors.vibration import AccelerometerVibration
from ..sensors.gazebo import (
    baro_datum_message,
    baro_errors_message,
    gps_errors_message,
    gps_multipath_message,
    imu_errors_message,
    magnetometer_anomaly_message,
    magnetometer_errors_message,
    vibrate_imu_message,
    bus_payload,
    gps_rtk_message,
    gps_velocity_message,
)
from ..simulation.gazebo.world import (
    build_world,
    load_scene,
    moving_obstacle_sdf,
    scene_terrain,
    world_elevation_m,
)


@dataclass(frozen=True)
class GazeboConfig:
    scene: str = "worlds/training.json"
    world: str | None = None
    # One path or an ordered sequence of layered drone config files.
    drone: str | tuple[str, ...] | list[str] | None = None
    output: str = "outputs/gazebo"
    duration: float = 20.0
    target: tuple[float, float, float] = (3.0, 0.0, 2.0)
    headless: bool = False
    rerun: bool = False
    render_sensors: bool = False
    generate_only: bool = False
    edit: bool = False
    keyboard: bool = False
    control_mode: str = "velocity"
    ekf: bool = False


def run(args: GazeboConfig, command_source=None):
    """Run a mission; optional command_source(sim_time) drives rotor-mode experiments."""
    if args.keyboard and args.edit:
        raise ValueError("--keyboard requires mission mode; omit --edit")
    if args.keyboard and command_source is not None:
        raise ValueError("Choose keyboard or a programmatic command source")
    if command_source is not None and args.control_mode != "rotor-thrust":
        raise ValueError("A command source requires rotor-thrust mode")
    keyboard = None
    try:
        # Generate-only deliberately never opens a keyboard window.
        if args.keyboard and not args.generate_only:
            from ..visualization.keyboard import KeyboardWindow

            keyboard = KeyboardWindow()
        return _run(args, keyboard, command_source)
    finally:
        if keyboard is not None:
            keyboard.close()


def _log_sensor_status(
    recorder, kind, message, t, *, gps_dropped, multipath, imu_faults
):
    """Sensor health indicators for Rerun's Sensor status tab.

    1 = healthy/delivered, 0 = dropped/invalid; fault values count active
    faults. Uses only the mission's own fault-model state.
    """
    if kind == "gps":
        recorder.log_status("gps_delivered", t, 0.0 if gps_dropped else 1.0)
        recorder.log_status("gps_multipath", t, float(multipath.active))
    elif kind == "imu":
        stamp = timestamp_ns(message)
        recorder.log_status("imu_faults_active", t, imu_faults.active_count(stamp))
    elif kind == "rangefinder":
        from ..sensors.gazebo import rangefinder_distance

        valid = np.isfinite(rangefinder_distance(message))
        recorder.log_status("rangefinder_valid", t, float(valid))


def _submit_to_bus(
    bus, kind, message, gps_dropped, camera_effects, gps_status, calibrated_field_ut
):
    """Submit a processed transported measurement to the sensor bus.

    Lidar (with effects), delayed camera frames, and synthetic sensors are
    submitted where they are produced. Dropped GPS fixes are never submitted.
    """
    if kind == "gps" and gps_dropped:
        return
    if kind == "camera" and not camera_effects.enabled:
        bus.submit("camera", CameraFrame(rgb_image(message)), timestamp_ns(message))
        return
    if kind == "lidar":
        return
    payload = bus_payload(kind, message)
    if kind == "gps":
        payload = replace(payload, status=gps_status)
    if kind == "magnetometer" and calibrated_field_ut is not None:
        payload = replace(payload, calibrated_field_ut=calibrated_field_ut)
    if payload is not None:
        bus.submit(kind, payload, timestamp_ns(message))


def _control_state(estimate_source, runner, now_ns, elapsed_s, max_age_s, truth):
    """Controller state from truth or the estimator (see experiments/algorithms.py).

    Returns None during start-up before the first estimate; raises
    StaleEstimate if there is still none after ``max_age_s``.
    """
    if estimate_source is None:
        return truth
    state = estimate_source.state(now_ns, runner.latest_estimate)
    if state is None and elapsed_s > max_age_s:
        raise StaleEstimate("No state estimate available for control")
    return state


def _calibrate_magnetometer(calibration, message, battery_reading, thrust_n, recorder):
    """Driver-side correction with the configured calibration (µT), or None.

    Uses only what an autopilot knows: the measured battery current (battery
    monitor; 0 without one) and its own total thrust command.
    """
    if calibration is None:
        return None
    raw = np.asarray(vector(message.field_tesla)) * 1e6
    current = battery_reading.current_a if battery_reading is not None else 0.0
    corrected = calibration.apply(raw, current, thrust_n)
    recorder.log_vector("magnetometer/calibrated_ut", corrected, ["x", "y", "z"])
    recorder.log_vector(
        "magnetometer/calibrated_level_heading_deg",
        [float(np.degrees(np.arctan2(corrected[0], corrected[1])))],
        ["value"],
    )
    return corrected


def _anomaly_in_sensor(anomalies, position, attitude, mount):
    """Anomaly field at the magnetometer, in sensor axes (truth synthesis)."""
    mount_rotation = Rotation.from_euler("xyz", mount["rotation_deg"], degrees=True)
    sensor_position = np.asarray(position) + attitude.apply(mount["position_m"])
    world_to_sensor = (attitude * mount_rotation).inv()
    return world_to_sensor.apply(anomalies.field_world(sensor_position))


def _fixed_wing_wrench(
    fixed_wing, servos, recorder, sim_time_ns, velocity, wind, quaternion, rates
):
    """Wing force and moment (world axes) with current servo positions."""
    if not fixed_wing.enabled:
        return np.zeros(3), np.zeros(3)
    deflections = servos.update(sim_time_ns)
    force, moment = fixed_wing.wrench_world(
        velocity, wind, quaternion, rates, deflections
    )
    state = fixed_wing.last
    recorder.log_vector(
        "aero/fixed_wing",
        [state.airspeed_mps, np.degrees(state.alpha_rad), np.degrees(state.beta_rad)],
        ["airspeed_mps", "alpha_deg", "beta_deg"],
    )
    recorder.log_vector(
        "aero/surfaces_deg",
        [np.degrees(v) for v in deflections.values()],
        list(deflections),
    )
    return force, moment


def _rotor_velocities(message):
    """Rotor joint velocities (rad/s, joint frame) in rotor index order."""
    by_name = {joint.name: joint.axis1.velocity for joint in message.joint}
    return [by_name[f"rotor_{i}_joint"] for i in range(len(by_name))]


def _ground_height(settings, terrain):
    """Ground height (m) under the configured spawn x, y."""
    x, y, _ = settings["simulation"]["spawn_position_m"]
    return 0.0 if terrain is None else float(terrain.height(x, y))


def _lockstep_stepper(node, settings):
    """Stepper advancing the paused world one odometry period per burst."""
    from types import SimpleNamespace

    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.world_control_pb2 import WorldControl

    from ..simulation.gazebo.stepping import LockstepStepper, steps_per_period

    period_s = 1 / settings["control"]["odometry_rate_hz"]
    return LockstepStepper(
        node.request,
        SimpleNamespace(WorldControl=WorldControl, Boolean=Boolean),
        steps=steps_per_period(period_s, settings["simulation"]["step_s"]),
        period_ns=round(period_s * 1e9),
    )


def _run(args, keyboard, command_source):
    from ..control.manual import FlightCommand, ManualController
    from ..control.drone import RotorThrustControl as ThrustCommands
    from ..simulation.gazebo.actuators import RotorThrustControl

    from ..simulation.gazebo.config import load_drone_config

    if args.drone and args.world:
        raise ValueError("--drone/--config requires a generated world; omit --world")
    config = load_drone_config(args.drone, render_sensors=args.render_sensors)
    settings = config.data

    if args.ekf:
        from .live_ekf import validate_live_settings

        if args.world or args.edit:
            raise ValueError(
                "--ekf requires a generated mission; omit --world and --edit"
            )
        validate_live_settings(settings)

    sensors, timing = settings["sensors"], settings["control"]

    if keyboard is not None:
        keyboard.pilot_rates = settings["pilot"]

    rotor_mode = args.keyboard or args.control_mode == "rotor-thrust"
    if args.control_mode not in {"velocity", "rotor-thrust"}:
        raise ValueError("Unknown control mode")

    lockstep = settings["simulation"]["lockstep"]
    if lockstep:
        if args.keyboard:
            raise ValueError("--keyboard needs real time; disable simulation.lockstep")
        if args.world or args.edit:
            raise ValueError(
                "Lockstep requires a generated mission; omit --world/--edit"
            )
        if rotor_mode and not timing["publish_on_send"]:
            raise ValueError(
                "Lockstep in rotor-thrust mode requires control.publish_on_send"
            )

    algorithms = settings["algorithms"]
    if algorithms["estimator"] and args.ekf:
        raise ValueError("Choose --ekf or algorithms.estimator, not both")
    if algorithms["control_state_source"] == "estimate" and not rotor_mode:
        raise ValueError("Estimate-driven control needs rotor-thrust mode")
    if (algorithms["estimator"] or algorithms["controller"]) and args.world:
        raise ValueError("Algorithm plug-ins require a generated world; omit --world")

    if settings["mission"]["type"] != "none":
        if not rotor_mode or args.keyboard or command_source is not None:
            raise ValueError(
                "A mission reference needs rotor-thrust mode without keyboard "
                "or a programmatic command source"
            )

    if rotor_mode and args.world:
        raise ValueError(
            "Rotor-thrust mode requires a generated scene; omit --world to prevent controller conflicts"
        )

    if (
        not np.isfinite(args.duration)
        or args.duration <= 0
        or not np.isfinite(args.target).all()
    ):
        raise ValueError("Duration must be positive and target must be finite")

    # Gazebo simulates the true (possibly randomized) vehicle; everything
    # algorithm-side keeps the nominal configuration.
    true_config, randomization_draws = randomize_config(config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.scene)
    terrain = scene_terrain(scene)
    world = (
        Path(args.world).resolve()
        if args.world
        else build_world(
            scene,
            output / "world.sdf",
            args.render_sensors,
            "rotor-thrust" if rotor_mode else "velocity",
            true_config,
        )
    )
    if not args.world:
        resolved = dict(settings)
        resolved["derived"] = dict(
            total_mass_kg=config.properties.mass,
            weight_n=config.properties.mass * 9.81,
            max_rotor_thrust_n=config.properties.max_rotor_thrust,
            config_sha256=config.content_hash,
        )
        if randomization_draws:
            true = true_config.data
            resolved["randomized_truth"] = dict(
                draws=randomization_draws,
                vehicle=true["vehicle"],
                motors={
                    k: true["motors"][k]
                    for k in (
                        "rotor_thrust_scale",
                        "rotor_moment_scale",
                        "time_constant_up_s",
                        "time_constant_down_s",
                    )
                },
                imu_mount=true["sensors"]["imu"]["mount"],
                total_mass_kg=true_config.properties.mass,
            )
        resolved["sources"] = list(config.sources)
        resolved["provenance"] = config.provenance
        (output / "drone-resolved.json").write_text(
            json.dumps(resolved, indent=2) + "\n"
        )
    if args.generate_only:
        print(world)
        return
    if shutil.which("gz") is None:
        raise ValueError(
            "gz not found. Install Gazebo Harmonic and source /opt/ros/jazzy/setup.bash if using ROS Jazzy."
        )
    if args.edit:
        if args.headless:
            raise ValueError("--edit needs a desktop window; omit --headless")
        subprocess.run(["gz", "sim", str(world)], check=True, env=gazebo_environment())
        return

    try:
        from ..visualization.gazebo_rerun import GazeboRerunRecorder
        from gz.transport13 import Node
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.twist_pb2 import Twist
        from gz.msgs10.imu_pb2 import IMU
        from gz.msgs10.navsat_pb2 import NavSat
        from gz.msgs10.contacts_pb2 import Contacts
        from gz.msgs10.fluid_pressure_pb2 import FluidPressure
        from gz.msgs10.magnetometer_pb2 import Magnetometer
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.laserscan_pb2 import LaserScan
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.model_pb2 import Model
    except ImportError as error:
        raise ValueError(
            f"{error}. Install .[gazebo] and source the Gazebo/ROS environment."
        )

    live_ekf = None
    estimate_reader = None
    if args.ekf:
        from .live_ekf import LiveEkf

        live_ekf = LiveEkf(settings, reference_from_world(world))
        estimate_reader = live_ekf.estimates.subscribe(capacity=4096)

    # Keep discovery private to this run, without interfering with another simulator.
    os.environ["GZ_PARTITION"] = "dronelab_" + uuid.uuid4().hex
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
    queue = Queue(maxsize=4000)
    dropped = Counter()
    gps_dropout = GpsDropout(**settings["sensors"]["gps"]["dropout"])
    vibration = AccelerometerVibration(settings["sensors"]["imu"]["vibration"])
    imu_errors = ImuErrorModel(
        sensors["imu"]["error_model"], rate_hz=sensors["imu"]["rate_hz"]
    )
    gps_errors = GpsErrorModel(sensors["gps"]["error_model"])
    gps_multipath = GpsMultipath(sensors["gps"]["multipath"])
    gps_velocity = GpsVelocityErrors(sensors["gps"]["velocity_error"])
    gps_rtk = RtkErrorModel(sensors["gps"]["rtk"])
    gps_status = GpsStatusModel(sensors["gps"]["receiver"])
    gps_heading = GpsHeading(sensors["gps"]["heading"])
    imu_faults = ImuFaults(sensors["imu"]["faults"])
    imu_imperfections = ImuImperfections(sensors["imu"]["imperfections"])
    imu_saturation = ImuSaturation(sensors["imu"]["imperfections"])
    imu_timing = ImuTiming(sensors["imu"]["imperfections"], sensors["imu"]["rate_hz"])
    imu_temperature = ImuTemperatureModel(
        sensors["imu"]["temperature"], settings["environment"]["ambient_temperature_c"]
    )
    baro_errors = BaroErrorModel(sensors["baro"]["error_model"])
    magnetometer_errors = MagnetometerErrorModel(sensors["magnetometer"]["error_model"])
    mag_settings = sensors["magnetometer"]["calibration"]
    mag_calibration = (
        MagnetometerCalibration.from_config(mag_settings)
        if mag_settings["enabled"]
        else None
    )
    calibrated_mag = None
    anomalies = MagneticAnomalies(
        list(sensors["magnetometer"]["dipoles"]) + scene_dipoles(scene)
    )
    optical_flow = OpticalFlowModel(sensors["optical_flow"], terrain)
    battery_monitor = BatteryMonitor(
        sensors["battery_monitor"], settings["battery"]["capacity_mah"]
    )
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
    # Latest total commanded thrust, for magnetometer motor interference.
    total_thrust_n = 0.0
    # Barometer datum: the world's sea-level elevation (see baro_datum_message).
    elevation_m = world_elevation_m(world)
    # Latest true position: fault-model state for GPS zones, never EKF input.
    truth_position = None

    def receive(kind):
        def callback(message):
            try:
                queue.put_nowait((kind, message, time.monotonic()))
            except Full:
                dropped[kind] += 1

        return callback

    node = Node()
    from ..simulation.gazebo.wind import WindCommands, dynamic_wind

    # Keyboard flight can switch configured wind on and off, which needs the
    # Python wind publisher even for static wind.
    wind_control = (
        WindCommands(node, settings["wind"])
        if dynamic_wind(settings["wind"])
        or (args.keyboard and settings["wind"]["enabled"])
        else None
    )
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
    esc_settings = settings["motors"]["esc"]
    esc = EscModel(
        esc_settings,
        (config.properties.max_rotor_thrust / config.properties.thrust_coefficient)
        ** 0.5,
        clock_ns=lambda: None if last_time is None else round(last_time * 1e9),
    )
    esc_telemetry = EscTelemetryModel(
        esc_settings["telemetry"], ROTOR_VELOCITY_SLOWDOWN
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

    def _payload_toggle(control, on, stamp):
        """Keyboard payload switch: off drops, on tries a pickup."""
        if not on:
            return control.drop(stamp)
        return control.pickup(stamp, truth_position, latest_attitude.as_quat())

    camera_effects = CameraEffects(sensors["camera"])
    camera_delay = DelayLine(round(sensors["camera"]["latency_s"] * 1e9))
    camera_mount = Rotation.from_euler(
        "xyz", sensors["camera"]["mount"]["rotation_deg"], degrees=True
    )
    latest_body_rates = np.zeros(3)
    latest_velocity_world = np.zeros(3)
    latest_attitude = Rotation.identity()
    lidar_effects = LidarEffects(sensors["lidar"], settings["environment"]["weather"])
    lidar_offset = np.asarray(sensors["lidar"]["mount"]["position_m"], float)
    lidar_mount = Rotation.from_euler(
        "xyz", sensors["lidar"]["mount"]["rotation_deg"], degrees=True
    )

    def _lidar_with_effects(message):
        """Scan with return physics, noise, dropouts, and motion distortion (body axes)."""
        # Sensor velocity includes the mount lever arm: v + ω × r.
        velocity_body = latest_attitude.inv().apply(latest_velocity_world) + np.cross(
            latest_body_rates, lidar_offset
        )
        points = lidar_effects.process_scan(
            message,
            lidar_mount.inv().apply(velocity_body),
            lidar_mount.inv().apply(latest_body_rates),
        )
        return lidar_mount.apply(points) + lidar_offset if len(points) else points

    # Keyboard flight can spawn moving objects (key 7), so it always has one.
    moving = (
        MovingObstacleCommands(node, scene.get("moving_obstacles", []))
        if scene.get("moving_obstacles") or keyboard is not None
        else None
    )
    aero = settings["aerodynamics"]
    body_drag = None
    max_drag_n = 0.0
    fixed_wing = FixedWingAero(settings["fixed_wing"], aero["air_density_kg_m3"])
    servos = ServoBank(settings["fixed_wing"]["servos"])
    airspeed_sensor = AirspeedSensor(sensors["airspeed"], aero["air_density_kg_m3"])
    if fixed_wing.enabled and not rotor_mode:
        raise ValueError("fixed_wing needs rotor-thrust mode")
    if aero["body_drag"]["enabled"] or fixed_wing.enabled:
        from ..simulation.gazebo.body_drag import BodyForce

        body_drag = BodyForce(node)

    def current_wind():
        """Wind the vehicle sees, for airspeed: latest target, else configured."""
        if wind_control is not None:
            return np.asarray(wind_control.velocity_enu_mps)
        wind = settings["wind"]
        return np.asarray(wind["velocity_enu_mps"] if wind["enabled"] else [0.0] * 3)

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
    last_battery_reading = None

    def _failsafe_step(time_s, position, quaternion, velocity_world):
        """Run the failsafe on measured battery values; apply landing/disarm."""
        if failsafe is None or control_loop is None or control_loop.disarmed:
            return None
        if last_battery_reading is not None:  # battery monitor (measured)
            remaining = last_battery_reading.remaining_pct
            voltage = last_battery_reading.voltage_v
        else:  # ideal measurement from the battery model
            state = powertrain.battery_state()
            remaining, voltage = 100.0 * state.soc, state.terminal_v
        ground = 0.0 if terrain is None else float(terrain.height(*position[:2]))
        clearance = position[2] - ground - settings["vehicle"]["body_size_m"][2] / 2
        yaw = Rotation.from_quat(quaternion).as_euler("xyz")[2]
        before = len(failsafe.events)
        command = failsafe.update(
            time_s, remaining, voltage, position, yaw, velocity_world[2], clearance
        )
        for event in failsafe.events[before:]:
            recorder.log_failsafe(event)
        if failsafe.state == "landing" and not landing_requested["done"]:
            controller.allow_landing()
            landing_requested["done"] = True
        if failsafe.state == "landed":
            control_loop.disarm()
        return command

    effects = None
    if keyboard is not None:
        flying_object = FlyingObjectSpawner(
            settings["keyboard_effects"],
            entities=EntityService(node),
            moving=moving,
            recorder=recorder,
            sdf_text=moving_obstacle_sdf,
            drone_state=lambda: (
                truth_position if truth_position is not None else np.zeros(3),
                last_time or 0.0,
            ),
        )
        effects = EffectToggles(
            settings["keyboard_effects"],
            flying_object=flying_object,
            payload=(
                (lambda on, stamp: _payload_toggle(payload, on, stamp))
                if payload is not None
                else None
            ),
            powertrain=powertrain,
            wind=wind_control,
            gps_dropout=gps_dropout,
            gps_multipath=gps_multipath,
            imu_faults=imu_faults,
            imu_temperature=imu_temperature,
            ambient_c=settings["environment"]["ambient_temperature_c"],
        )
        keyboard.configure_actions(
            [
                (a.key, a.name, a.label, effects.action_available(a.name))
                for a in ACTIONS
            ]
        )
        keyboard.configure_effects(
            [(e.key, e.name, e.label, effects.available(e.name)) for e in EFFECTS],
            effects.state,
        )
    thrust_control = None
    thrust_handle = None
    control_loop = None
    stepper = None
    landing_requested = {"done": False}
    # Algorithms see configuration-derived context only, never truth.
    bus_kinds = tuple(
        kind
        for kind in STREAMS
        if (kind in sensors and sensors[kind]["enabled"])
        or (kind == "esc" and esc_telemetry.enabled)
        or (kind == "gps_heading" and gps_heading.enabled)
    )
    context = build_context(
        settings,
        config.properties,
        reference_from_world(world) if not args.world else GpsReading(0, 0, 0),
        bus_kinds,
    )
    estimator = build_estimator(settings, context)
    bus = SensorBus(settings["sensor_timing"], bus_kinds) if estimator else None
    runner = AlgorithmRunner(bus, estimator) if bus is not None else None
    estimate_source = (
        EstimateStateSource(
            context.sensors["imu"].mount_quaternion_xyzw,
            algorithms["max_estimate_age_s"],
        )
        if algorithms["control_state_source"] == "estimate"
        else None
    )
    controller = build_controller(settings, context) if rotor_mode else None
    # True mount (randomization may misalign it), for evaluation only.
    imu_body = Rotation.from_euler(
        "xyz", true_config.data["sensors"]["imu"]["mount"]["rotation_deg"], degrees=True
    )
    native_bias = [
        np.asarray(sensors["imu"][field]["bias"], float)
        for field in ("acceleration_mps2", "angular_rate_rad_s")
    ]

    def true_biases():
        """Simulator IMU biases in body axes, for evaluation only."""
        (a1, g1), (a2, g2), (a3, g3) = (
            imu_errors.current_biases(),
            imu_temperature.current_biases(),
            imu_imperfections.current_biases(),
        )
        return (
            imu_body.apply(native_bias[0] + a1 + a2 + a3),
            imu_body.apply(native_bias[1] + g1 + g2 + g3),
        )

    evaluation = RunEvaluation(
        settings,
        true_config.properties.mass * 9.81,
        mission,
        recorder.log_vector,
        true_biases,
    )
    bus_now_ns = 0
    # Flying past the ground plane's edge means nothing collides below.
    ground_half_m = scene.get("ground_size_m", 100.0) / 2
    left_ground_s = None
    publisher = None if rotor_mode else node.advertise("/drone/cmd_vel", Twist)
    subscriptions = [
        (Odometry, "/dronelab/odometry", "odometry"),
        (Pose_V, "/world/dronelab/pose/info", "poses"),
    ]
    if esc_telemetry.enabled:
        subscriptions.append((Model, ROTOR_STATE_TOPIC, "esc"))
    for name, cls, topic, kind in (
        ("imu", IMU, "/dronelab/imu", "imu"),
        ("gps", NavSat, "/dronelab/gps", "gps"),
        ("baro", FluidPressure, "/dronelab/baro", "baro"),
        ("magnetometer", Magnetometer, "/dronelab/magnetometer", "magnetometer"),
        ("rangefinder", LaserScan, "/dronelab/rangefinder", "rangefinder"),
        ("contacts", Contacts, "/dronelab/contacts", "contacts"),
        ("camera", Image, "/dronelab/camera", "camera"),
        ("lidar", LaserScan, "/dronelab/lidar", "lidar"),
    ):
        if sensors[name]["enabled"]:
            subscriptions.append((cls, topic, kind))
    for cls, topic, kind in subscriptions:
        if not node.subscribe(cls, topic, receive(kind)):
            raise ValueError(f"Unable to subscribe to {topic}")

    # Lockstep starts paused; Python advances the world (see stepping.py).
    command = ["gz", "sim", "-v", "3"] if lockstep else ["gz", "sim", "-r", "-v", "3"]
    if args.headless:
        command.append("-s")
        if config.render_sensors:
            command.append("--headless-rendering")
    command.append(str(world))
    counts, positions = Counter(), []
    first_time = last_time = None
    collisions = []
    last_received = time.monotonic()
    started = last_received
    completed = False
    last_command_time = -float("inf")
    previous_control_time = None
    exit_reason = "duration"
    with (
        (
            (output / "estimates.jsonl").open("w")
            if args.ekf or estimator is not None
            else nullcontext(None)
        ) as estimate_log,
        (output / "gazebo.log").open("w") as log,
        (output / "telemetry.jsonl").open("w") as telemetry,
        (output / "commands.jsonl").open("w") as command_log,
        (output / "command-events.jsonl").open("w") as command_events,
        (
            (output / "effect-toggles.jsonl").open("w")
            if keyboard is not None
            else nullcontext(None)
        ) as effect_log,
        (
            (output / "wind-commands.jsonl").open("w")
            if wind_control
            else nullcontext(None)
        ) as wind_log,
    ):
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=gazebo_environment()
        )
        try:
            if rotor_mode:
                thrust_control = ThrustCommands(
                    actuator.send,
                    properties=config.properties,
                    publish_rate_hz=timing["publish_rate_hz"],
                    max_lifetime_sim_s=timing["command_lifetime_sim_s"],
                    # Gazebo cannot advance while Python stalls in lockstep.
                    watchdog_wall_s=None if lockstep else timing["watchdog_wall_s"],
                    publish_on_send=timing["publish_on_send"],
                )
                thrust_handle = thrust_control.acquire()
                control_loop = RotorControlLoop(
                    controller,
                    thrust_control,
                    thrust_handle,
                    timing,
                    # Half a physics step absorbs timestamp quantization.
                    tolerance_ns=round(settings["simulation"]["step_s"] * 0.5e9),
                )
            if lockstep:
                stepper = _lockstep_stepper(node, settings)
                stepper.start()

            def write_estimate(sample):
                """Record one estimate (estimates.jsonl, Rerun, metrics)."""
                state = sample.payload
                estimate_log.write(
                    json.dumps(
                        dict(
                            sim_time_ns=sample.sim_time_ns,
                            sequence=sample.sequence,
                            stream_id=sample.stream_id,
                            epoch=sample.epoch,
                            frame_id=sample.frame_id,
                            position_m=state.position_m.tolist(),
                            velocity_mps=state.velocity_mps.tolist(),
                            quaternion_xyzw=state.quaternion_xyzw.tolist(),
                            accel_bias_mps2=state.accel_bias_mps2.tolist(),
                            gyro_bias_rad_s=state.gyro_bias_rad_s.tolist(),
                            covariance=state.covariance.tolist(),
                            parents=sample.parents,
                        )
                    )
                    + "\n"
                )
                recorder.log_estimate(sample)
                evaluation.on_estimate(sample)

            def drain_estimates():
                if estimate_reader is None:
                    return
                from ..sensors.streams import ReadTimeout

                while True:
                    try:
                        sample = estimate_reader.next(timeout_wall_s=0)
                    except ReadTimeout:
                        break
                    write_estimate(sample)

            while True:
                if live_ekf is not None:
                    live_ekf.check_health()
                    if dropped["imu"] or dropped["gps"]:
                        raise RuntimeError(
                            "Sensor ingress loss during live EKF mission"
                        )
                if thrust_control is not None:
                    thrust_control.check_health()
                    for event in thrust_control.drain_events():
                        command_events.write(json.dumps(event) + "\n")
                        recorder.log_actuator_event(event)
                pilot_command = (
                    keyboard.poll() if keyboard is not None else FlightCommand()
                )
                if effects is not None:
                    for name in keyboard.drain_actions():
                        event = effects.perform(name, round((last_time or 0) * 1e9))
                        if event is not None:
                            effect_log.write(json.dumps(event) + "\n")
                            recorder.log_effect(event)
                    for name, on in keyboard.drain_toggles():
                        event = effects.set(name, on, round((last_time or 0) * 1e9))
                        if event is not None:
                            effect_log.write(json.dumps(event) + "\n")
                            recorder.log_effect(event)
                if keyboard is not None and keyboard.closed:
                    exit_reason = "keyboard_closed"
                    break
                if (
                    rotor_mode
                    and not lockstep
                    and previous_control_time is not None
                    and time.monotonic() - last_received > timing["watchdog_wall_s"]
                ):
                    raise RuntimeError(
                        "Odometry watchdog expired; ending rotor-control mission"
                    )
                if process.poll() is not None:
                    raise RuntimeError(
                        f'Gazebo exited ({process.returncode}); see {output / "gazebo.log"}'
                    )
                try:
                    kind, message, received_at = queue.get(
                        timeout=0.01 if keyboard is not None else 0.2
                    )
                except Empty:
                    if stepper is not None:
                        stepper.check_stalled()
                    if time.monotonic() - last_received > 30:
                        raise RuntimeError(
                            f'No telemetry for 30 seconds; see {output / "gazebo.log"}'
                        )
                    continue
                counts[kind] += 1
                # Sensor-error and fault models run before recording and the
                # EKF. IMU: errors, vibration/clipping, then digital faults.
                # GPS: errors and multipath see every fix, before dropout, so
                # dropouts do not shift their random sequences.
                if kind == "imu":
                    stamp = timestamp_ns(message)
                    if imu_timing.drop(stamp):
                        continue  # sample lost: never recorded or delivered
                    message = imu_errors_message(message, imu_imperfections, stamp)
                    message = imu_errors_message(message, imu_errors, stamp)
                    message = imu_errors_message(message, imu_temperature, stamp)
                    message = vibrate_imu_message(message, vibration, stamp)
                    message = imu_errors_message(message, imu_saturation, stamp)
                    message = imu_errors_message(message, imu_faults, stamp)
                    if imu_timing.enabled:
                        message = with_timestamp_ns(message, imu_timing.stamp(stamp))
                elif kind == "magnetometer":
                    if anomalies.enabled and truth_position is not None:
                        message = magnetometer_anomaly_message(
                            message,
                            _anomaly_in_sensor(
                                anomalies,
                                truth_position,
                                latest_attitude,
                                sensors["magnetometer"]["mount"],
                            ),
                        )
                    battery = powertrain.battery_state() if powertrain.enabled else None
                    message = magnetometer_errors_message(
                        message,
                        magnetometer_errors,
                        total_thrust_n,
                        battery.current_a if battery is not None else 0.0,
                    )
                    calibrated_mag = _calibrate_magnetometer(
                        mag_calibration,
                        message,
                        last_battery_reading,
                        total_thrust_n,
                        recorder,
                    )
                elif kind == "baro":
                    message = baro_datum_message(message, elevation_m)
                    # Downwash from the latest total commanded thrust (momentum
                    # theory) and horizontal airspeed act on the static port.
                    horizontal_air = latest_velocity_world - current_wind()
                    message = baro_errors_message(
                        message,
                        baro_errors,
                        timestamp_ns(message),
                        induced_velocity_mps=hover_induced_velocity(
                            total_thrust_n / 4,
                            config.properties.propeller_diameter / 2,
                            settings["aerodynamics"]["air_density_kg_m3"],
                        ),
                        airspeed_mps=float(np.hypot(*horizontal_air[:2])),
                    )
                elif kind == "gps":
                    stamp = timestamp_ns(message)
                    message = gps_errors_message(message, gps_errors, stamp)
                    message = gps_rtk_message(
                        message, gps_rtk, stamp, gps_multipath.active
                    )
                    message = gps_multipath_message(
                        message, gps_multipath, stamp, truth_position
                    )
                    message = gps_velocity_message(message, gps_velocity, stamp)
                    status = gps_status.update(gps_multipath.active, gps_rtk)
                    recorder.log_vector(
                        "gps/status",
                        [status.satellites, status.hdop, status.horizontal_accuracy_m],
                        ["satellites", "hdop", "horizontal_accuracy_m"],
                    )
                t = timestamp(message)
                if kind == "odometry" and wind_control is not None:
                    wind_event = wind_control.update(timestamp_ns(message))
                    if wind_event is not None:
                        wind_log.write(json.dumps(wind_event) + "\n")
                gps_dropped = kind == "gps" and gps_dropout.drop(
                    timestamp_ns(message), truth_position
                )
                if kind == "gps" and not gps_dropped and gps_heading.enabled:
                    heading = gps_heading.sample(
                        float(latest_attitude.as_euler("ZYX")[0])
                    )
                    recorder.log_vector(
                        "gps/heading_rad", [heading.heading_rad], ["value"]
                    )
                    if bus is not None:
                        bus.submit("gps_heading", heading, timestamp_ns(message))
                if kind == "camera" and camera_effects.enabled:
                    # Effects need the camera rotation rate at capture.
                    image = camera_effects.process(
                        rgb_image(message), camera_mount.inv().apply(latest_body_rates)
                    )
                    camera_delay.push(timestamp_ns(message), image)
                elif kind == "lidar" and lidar_effects.enabled:
                    points = _lidar_with_effects(message)
                    recorder.log_lidar_points(
                        points, t, lidar_effects.last_reflectivity
                    )
                    if bus is not None:
                        bus.submit(
                            "lidar",
                            LidarPoints(points, lidar_effects.last_reflectivity),
                            timestamp_ns(message),
                        )
                elif kind == "esc":
                    report = esc_telemetry.sample(
                        timestamp_ns(message), _rotor_velocities(message)
                    )
                    if report is not None:
                        recorder.log_vector(
                            "esc/rotor_speed_rad_s",
                            report.rotor_speed_rad_s,
                            [
                                f"rotor_{i}"
                                for i in range(len(report.rotor_speed_rad_s))
                            ],
                        )
                        if bus is not None:
                            bus.submit("esc", report, timestamp_ns(message))
                elif not gps_dropped:
                    recorder.log(kind, message, t)
                if camera_effects.enabled:
                    # Deliver frames whose latency has elapsed, at delivery time.
                    newest = max(round(t * 1e9), round((last_time or 0) * 1e9))
                    for ready_ns, frame in camera_delay.pop_ready(newest):
                        recorder.log_camera_image(frame, ready_ns * 1e-9)
                        if bus is not None:
                            # Stamped at capture, delivered at release.
                            bus.submit(
                                "camera",
                                CameraFrame(frame),
                                ready_ns - camera_delay.delay_ns,
                            )
                if kind == "imu" and imu_temperature.enabled:
                    recorder.log_vector(
                        "imu/temperature_c", [imu_temperature.temperature_c], ["value"]
                    )
                _log_sensor_status(
                    recorder,
                    kind,
                    message,
                    t,
                    gps_dropped=gps_dropped,
                    multipath=gps_multipath,
                    imu_faults=imu_faults,
                )
                if bus is not None:
                    _submit_to_bus(
                        bus,
                        kind,
                        message,
                        gps_dropped,
                        camera_effects,
                        gps_status.status,
                        calibrated_mag if kind == "magnetometer" else None,
                    )
                    if kind not in ("poses", "contacts"):
                        bus_now_ns = max(bus_now_ns, timestamp_ns(message))
                    delivered, new_estimates, innovations = runner.advance(bus_now_ns)
                    for delivered_kind, sample in delivered:
                        if estimate_source is not None:
                            estimate_source.on_delivered(delivered_kind, sample)
                    for sample in new_estimates:
                        write_estimate(sample)
                    evaluation.on_innovations(innovations)
                if live_ekf is not None:
                    live_ekf.ingest(
                        kind, message, received_at=received_at, gps_dropped=gps_dropped
                    )
                    drain_estimates()
                if kind == "odometry":
                    last_received = time.monotonic()
                    if first_time is None:
                        first_time = t
                    last_time = t
                    p = vector(message.pose.position)
                    truth_position = p
                    q = message.pose.orientation
                    quaternion = np.array([q.x, q.y, q.z, q.w])
                    # Gazebo publishes linear and angular odometry twist in body axes.
                    velocity_world = Rotation.from_quat(quaternion).apply(
                        vector(message.twist.linear)
                    )
                    if winch is not None:
                        rate = keyboard.winch_rate() if keyboard is not None else 0
                        cable_m = winch.update(timestamp_ns(message), rate)
                        recorder.log_vector(
                            "payload/cable_length_m", [cable_m], labels=["value"]
                        )
                    if payload is not None:
                        payload_event = payload.update(timestamp_ns(message))
                        if payload_event is not None and effects is not None:
                            effects.state["payload"] = False
                    if moving is not None:
                        # Path time is absolute simulation time (start at waypoint 0).
                        moving.update(t)
                    # Gazebo reports bogus start-up velocities (see odometry.py).
                    velocity_world, body_rates = odometry_fix.correct(
                        timestamp_ns(message),
                        p,
                        quaternion,
                        velocity_world,
                        vector(message.twist.angular),
                    )
                    # Latest motion, for camera and lidar effects.
                    latest_body_rates = body_rates
                    latest_velocity_world = velocity_world
                    latest_attitude = Rotation.from_quat(quaternion)
                    airspeed = velocity_world - current_wind()
                    air_body = latest_attitude.inv().apply(airspeed)
                    if body_drag is not None:
                        drag = (
                            quadratic_body_drag_world(
                                velocity_world,
                                current_wind(),
                                quaternion,
                                aero["body_drag"]["drag_area_m2"],
                                aero["air_density_kg_m3"],
                            )
                            if aero["body_drag"]["enabled"]
                            else np.zeros(3)
                        )
                        wing_force, wing_moment = _fixed_wing_wrench(
                            fixed_wing,
                            servos,
                            recorder,
                            timestamp_ns(message),
                            velocity_world,
                            current_wind(),
                            quaternion,
                            body_rates,
                        )
                        body_drag.set(drag + wing_force, wing_moment)
                        max_drag_n = max(max_drag_n, float(np.linalg.norm(drag)))
                        recorder.log_vector("aero/body_drag_n", drag)
                    flow_reading = optical_flow.sample(
                        timestamp_ns(message), p, velocity_world, quaternion, body_rates
                    )
                    pitot = airspeed_sensor.sample(timestamp_ns(message), air_body)
                    if pitot is not None:
                        counts["airspeed"] += 1
                        recorder.log_vector(
                            "airspeed/measured",
                            [
                                pitot.indicated_airspeed_mps,
                                pitot.differential_pressure_pa,
                            ],
                            ["indicated_mps", "differential_pa"],
                        )
                        if bus is not None:
                            bus.submit("airspeed", pitot, pitot.sim_time_ns)
                    if flow_reading is not None:
                        counts["optical_flow"] += 1
                        recorder.log_optical_flow(flow_reading)
                        if bus is not None:
                            bus.submit(
                                "optical_flow", flow_reading, flow_reading.sim_time_ns
                            )
                    evaluation.on_truth(
                        timestamp_ns(message),
                        p,
                        velocity_world,
                        quaternion,
                        controlled=rotor_mode,
                    )
                    positions.append(p)
                    if left_ground_s is None and np.max(np.abs(p[:2])) > ground_half_m:
                        left_ground_s = t - first_time
                        print(
                            f"Warning: vehicle left the {2 * ground_half_m:.0f} m ground "
                            f"area at {left_ground_s:.1f} s; nothing below it collides"
                        )
                    telemetry.write(
                        json.dumps(
                            dict(
                                time=t,
                                position=p.tolist(),
                                quaternion=quaternion.tolist(),
                            )
                        )
                        + "\n"
                    )
                    if rotor_mode:
                        if not lockstep and time.monotonic() - received_at > 0.25:
                            raise RuntimeError("Stale odometry in rotor-control queue")
                        dt = (
                            1 / timing["odometry_rate_hz"]
                            if previous_control_time is None
                            else t - previous_control_time
                        )
                        if dt <= 0 or dt > 0.1:
                            raise RuntimeError(
                                "Invalid odometry interval in rotor-control mode"
                            )
                        previous_control_time = t
                        if powertrain.enabled:
                            powertrain.update(
                                timestamp_ns(message), p, quaternion, airspeed
                            )
                            recorder.log_vector(
                                "aero/rotor_thrust_factor",
                                powertrain.rotor_thrust_factors(),
                                labels=[
                                    f"rotor_{i}"
                                    for i in range(len(config.properties.rotors))
                                ],
                            )
                            recorder.log_powertrain(
                                powertrain.battery_state(),
                                powertrain.ground_effect_factors(),
                                powertrain.available_thrust_fraction(),
                            )
                            reading = battery_monitor.sample(
                                timestamp_ns(message), powertrain.battery_state()
                            )
                            if reading is not None:
                                counts["battery_monitor"] += 1
                                recorder.log_battery_monitor(reading)
                                last_battery_reading = reading
                                if bus is not None:
                                    bus.submit(
                                        "battery_monitor", reading, reading.sim_time_ns
                                    )
                        state = _control_state(
                            estimate_source,
                            runner,
                            timestamp_ns(message),
                            t - first_time,
                            algorithms["max_estimate_age_s"],
                            TruthState(
                                timestamp_ns(message),
                                p,
                                velocity_world,
                                quaternion,
                                body_rates,
                                airspeed_mps=float(air_body[0]),
                            ),
                        )
                        if state is None:
                            # Start-up: no estimate yet; no command this tick.
                            if stepper is not None:
                                stepper.on_odometry(timestamp_ns(message))
                            continue
                        failsafe_command = _failsafe_step(
                            t - first_time,
                            state.position_m,
                            state.quaternion_xyzw,
                            state.velocity_world_mps,
                        )

                        # Take-off ends on the controller's altitude setpoint
                        # (the vehicle's height before the first update).
                        setpoint = controller.setpoint_m
                        setpoint_z = (
                            state.position_m[2] if setpoint is None else setpoint[2]
                        )

                        def pilot(
                            tick_time_s,
                            held=pilot_command,
                            altitude=setpoint_z,
                            override=failsafe_command,
                            position=state.position_m,
                        ):
                            if override is not None:
                                return override  # failsafe takes precedence
                            elapsed = tick_time_s - first_time
                            if mission.enabled:
                                if takeoff is not None and not takeoff.complete:
                                    return takeoff.command(
                                        altitude, elapsed, FlightCommand()
                                    )
                                return mission.command(elapsed, position)
                            command = (
                                held
                                if command_source is None
                                else command_source(tick_time_s - first_time)
                            )
                            if takeoff is None:
                                return command
                            return takeoff.command(
                                altitude, tick_time_s - first_time, command
                            )

                        for record in control_loop.on_state(state, pilot):
                            command_log.write(json.dumps(record) + "\n")
                            total_thrust_n = sum(record["accepted_thrust_n"])
                            evaluation.on_command(record)
                        surface_commands = getattr(controller, "surface_commands", None)
                        if surface_commands is not None and fixed_wing.enabled:
                            servos.set_command(surface_commands())
                        if keyboard is not None:
                            keyboard.show_position(p)
                    elif t - last_command_time >= 0.04:
                        twist = Twist()
                        velocity = waypoint_velocity(p, quaternion, args.target)
                        twist.linear.x, twist.linear.y, twist.linear.z = map(
                            float, velocity
                        )
                        publisher.publish(twist)
                        last_command_time = t
                    if t - first_time >= args.duration:
                        if live_ekf is not None and live_ekf.node.ekf.accepted_gps == 0:
                            raise RuntimeError(
                                "Live EKF has no accepted GPS correction; inspect summary"
                            )
                        completed = True
                        break
                    if stepper is not None:
                        # Everything for this period is done: advance one burst.
                        stepper.on_odometry(timestamp_ns(message))
                elif kind == "poses" and (moving is not None or payload is not None):
                    for pose in message.pose:
                        if payload is not None:
                            payload.on_pose(pose.name, vector(pose.position))
                        if moving is None:
                            continue
                        q = pose.orientation
                        moving.on_pose(
                            pose.name,
                            vector(pose.position),
                            np.array([q.x, q.y, q.z, q.w]),
                        )
                elif kind == "contacts":
                    for contact in message.contact:
                        collisions.append(
                            dict(
                                time=t,
                                a=contact.collision1.name,
                                b=contact.collision2.name,
                            )
                        )
        except KeyboardInterrupt:
            exit_reason = "interrupted"
            print("Interrupted; saving received data.")
        finally:
            # Publication failure must not prevent owned-process cleanup.
            try:
                if thrust_control is not None:
                    thrust_control.close()
                elif actuator is not None:
                    actuator.stop()
                else:
                    publisher.publish(Twist())
            except Exception:
                pass
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if thrust_control is not None:
                for event in thrust_control.drain_events():
                    command_events.write(json.dumps(event) + "\n")
                    recorder.log_actuator_event(event)
            if live_ekf is not None:
                live_ekf.close()
                estimate_reader.close()
            if bus is not None:
                bus.close()
            metrics = evaluation.summary()
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            summary = dict(
                completed=completed,
                ekf=live_ekf.summary() if live_ekf is not None else None,
                exit_reason=(
                    exit_reason if completed or exit_reason != "duration" else "error"
                ),
                control_mode="rotor-thrust" if rotor_mode else "velocity",
                state_source=(
                    "simulator ground truth"
                    if estimate_source is None
                    else f"estimate ({algorithms['estimator']})"
                ),
                algorithms=dict(
                    estimator=algorithms["estimator"] or None,
                    controller=algorithms["controller"] or "manual",
                    estimator_summary=(
                        estimator.summary() if estimator is not None else None
                    ),
                    max_estimate_age_ms=(
                        estimate_source.max_age_ms if estimate_source else None
                    ),
                    controller_summary=(
                        controller.summary()
                        if controller is not None and hasattr(controller, "summary")
                        else None
                    ),
                ),
                sensor_bus=bus.summary() if bus is not None else None,
                randomization=randomization_draws or None,
                esc=(
                    dict(command=esc.summary(), telemetry=esc_telemetry.summary())
                    if esc.enabled or esc_telemetry.enabled
                    else None
                ),
                metrics=metrics,
                drone_config_sha256=config.content_hash if not args.world else None,
                saturated_commands=(
                    control_loop.stats["saturated_ticks"] if control_loop else 0
                ),
                control_timing=control_loop.summary() if control_loop else None,
                dropped_command_events=(
                    thrust_control.dropped_events if thrust_control else 0
                ),
                thrust_publish_rate_hz=(
                    timing["publish_rate_hz"] if rotor_mode else None
                ),
                command_lifetime_sim_s=(
                    timing["command_lifetime_sim_s"] if rotor_mode else None
                ),
                command_watchdog_wall_s=(
                    timing["watchdog_wall_s"] if rotor_mode else None
                ),
                messages=dict(counts),
                dropped=dict(dropped),
                gps_dropout=gps_dropout.summary(),
                accelerometer_vibration=vibration.summary(),
                imu_errors=imu_errors.summary(),
                gps_errors=gps_errors.summary(),
                gps_multipath=gps_multipath.summary(),
                imu_faults=imu_faults.summary(),
                baro_errors=baro_errors.summary(),
                magnetometer_errors=magnetometer_errors.summary(),
                magnetic_anomalies=(
                    dict(max_field_ut=anomalies.max_field_ut)
                    if anomalies.enabled
                    else None
                ),
                optical_flow=optical_flow.summary(),
                battery_monitor=battery_monitor.summary(),
                powertrain=powertrain.summary() if powertrain.enabled else None,
                effect_toggles=effects.summary() if effects is not None else None,
                odometry_startup_corrections=odometry_fix.corrected_samples,
                lidar_effects=(
                    lidar_effects.summary() if lidar_effects.enabled else None
                ),
                payload=payload.summary() if payload is not None else None,
                mission=mission.summary() if mission.enabled else None,
                imu_temperature=imu_temperature.summary(),
                airspeed=airspeed_sensor.summary() if airspeed_sensor.enabled else None,
                gps_receiver=dict(
                    last_status=(
                        None
                        if gps_status.status is None
                        else gps_status.status.__dict__
                    ),
                    rtk_fixed_fixes=gps_rtk.fixed_fixes if gps_rtk.enabled else None,
                    heading_samples=gps_heading.samples,
                ),
                imu_imperfections=dict(
                    enabled=imu_imperfections.enabled,
                    saturated_samples=imu_saturation.saturated_samples,
                    dropped_samples=imu_timing.dropped,
                ),
                failsafe=failsafe.summary() if failsafe is not None else None,
                camera_effects=(
                    dict(max_blur_px=camera_effects.max_blur_px)
                    if camera_effects.enabled
                    else None
                ),
                moving_obstacles=(
                    dict(max_path_error_m=moving.max_path_error_m)
                    if moving is not None
                    else None
                ),
                body_drag=(
                    dict(max_force_n=max_drag_n) if body_drag is not None else None
                ),
                takeoff=(
                    dict(
                        complete=takeoff.complete,
                        complete_time_s=takeoff.complete_time_s,
                    )
                    if takeoff is not None
                    else None
                ),
                sim_start=first_time,
                sim_end=last_time,
                collisions=collisions,
                target=None if rotor_mode else args.target,
                wall_seconds=time.monotonic() - started,
                lockstep=stepper.summary() if stepper is not None else None,
                final_position=positions[-1].tolist() if positions else None,
                left_ground_area_s=left_ground_s,
            )
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            recorder.close()
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
    print(f"Messages: {dict(counts)}; contact samples: {len(collisions)}")
