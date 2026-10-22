"""Self-contained quadcopter SDF model, motors, controller, and sensor mounts."""

import math

import numpy as np

from .sdf import element, plugin, box
from .config import load_drone_config, mount_pose, rotor_scale, wind_effects_enabled


def _axis_rpy(axis):
    """Fixed-axis roll, pitch, yaw rotating +z onto the unit ``axis``.

    SDF poses are R = Rz(yaw)·Ry(pitch)·Rx(roll); with roll 0,
    Rz(ψ)·Ry(θ)·ẑ = (sinθ cosψ, sinθ sinψ, cosθ), so θ = acos(a_z) and
    ψ = atan2(a_y, a_x).
    """
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    pitch = math.acos(max(-1.0, min(1.0, z)))
    yaw = math.atan2(y, x) if pitch > 1e-12 else 0.0
    return 0.0, pitch, yaw


# Gazebo's motor model spins rotor joints at speed / slowdown (visual only).
ROTOR_VELOCITY_SLOWDOWN = 10
ROTOR_STATE_TOPIC = "/robotlab/rotor_state"


def build_drone(
    world, render_sensors=False, control_mode="velocity", config=None, terrain=None
):
    if control_mode not in {"velocity", "rotor-thrust"}:
        raise ValueError("Unknown control mode")
    config = config or load_drone_config(render_sensors=render_sensors)
    properties = config.properties
    settings = config.data
    sensors, motors = settings["sensors"], settings["motors"]
    drone = element(world, "model", name="drone")
    element(
        drone,
        "pose",
        " ".join(map(str, spawn_position(settings, terrain))) + " 0 0 0",
    )
    base = element(drone, "link", name="base_link")
    if wind_effects_enabled(settings["wind"]):
        element(base, "enable_wind", "true")
    inertial = element(base, "inertial")
    com = settings["vehicle"]["center_of_mass_offset_m"]
    if any(com):
        # Mass centre away from the body origin (odometry stays at the origin).
        element(inertial, "pose", " ".join(map(str, com)) + " 0 0 0")
    element(inertial, "mass", properties.body_mass)
    inertia = element(inertial, "inertia")
    for key, value in dict(zip(("ixx", "iyy", "izz"), properties.body_inertia)).items():
        element(inertia, key, value)
    box(base, "body", settings["vehicle"]["body_size_m"], ".15 .3 .7 1")
    # Heading marker: a red stripe on top of the body's front half, scaled to
    # the body so it never sits in front of a forward-mounted camera.
    bx, by, bz = settings["vehicle"]["body_size_m"]
    box(
        base,
        "nose",
        [bx / 2, by / 8, 0.01],
        ".9 .1 .1 1",
        f"{bx / 4} 0 {bz / 2 + 0.005} 0 0 0",
        collision=False,
    )
    wing = settings["fixed_wing"]
    if wing["enabled"]:
        # Visual only: the aerodynamics are computed in Python, and contacts
        # use the fuselage box.
        box(
            base,
            "wing",
            [wing["chord_m"], wing["wingspan_m"], 0.03],
            ".85 .85 .9 1",
            collision=False,
        )
        tail_x = -0.45 * settings["vehicle"]["body_size_m"][0]
        box(
            base,
            "tail",
            [0.6 * wing["chord_m"], 0.3 * wing["wingspan_m"], 0.02],
            ".85 .85 .9 1",
            f"{tail_x} 0 0.05 0 0 0",
            collision=False,
        )
        box(
            base,
            "fin",
            [0.6 * wing["chord_m"], 0.02, 0.3],
            ".85 .85 .9 1",
            f"{tail_x} 0 0.2 0 0 0",
            collision=False,
        )
    contact = element(base, "sensor", name="contact", type="contact")
    element(contact, "always_on", "true")
    element(contact, "update_rate", sensors["contacts"]["rate_hz"])
    contact_settings = element(contact, "contact")
    element(contact_settings, "collision", "body")
    element(contact_settings, "topic", "/robotlab/contacts")
    imu = element(base, "sensor", name="imu", type="imu")
    element(imu, "always_on", "true")
    element(imu, "update_rate", sensors["imu"]["rate_hz"])
    element(imu, "topic", "/robotlab/imu")
    imu_settings = element(imu, "imu")
    for section, field in (
        ("linear_acceleration", "acceleration_mps2"),
        ("angular_velocity", "angular_rate_rad_s"),
    ):
        component = element(imu_settings, section)
        error = sensors["imu"][field]
        for i, axis in enumerate("xyz"):
            add_noise(
                element(component, axis), error["bias"][i], error["noise_stddev"][i]
            )
    gps = element(base, "sensor", name="gps", type="navsat")
    element(gps, "always_on", "true")
    element(gps, "update_rate", sensors["gps"]["rate_hz"])
    gps_settings = element(gps, "navsat")
    for section, fields in (
        ("position_sensing", ("horizontal_position_m", "vertical_position_m")),
        ("velocity_sensing", ("horizontal_velocity_mps", "vertical_velocity_mps")),
    ):
        component = element(gps_settings, section)
        for axis, field in zip(("horizontal", "vertical"), fields):
            error = sensors["gps"][field]
            # gz-sensors8 applies horizontal position noise directly to degrees,
            # despite SDF documenting metres. Convert to angular units. A shared
            # lat/lon model makes longitude metre errors scale by cos(latitude).
            scale = (
                math.degrees(1 / 6378137.0) if field == "horizontal_position_m" else 1.0
            )
            add_noise(
                element(component, axis),
                error["bias"] * scale,
                error["noise_stddev"] * scale,
            )
    element(gps, "topic", "/robotlab/gps")
    add_barometer(base, sensors["baro"])

    if sensors["camera"]["enabled"]:
        lens = sensors["camera"]["lens"]["type"]
        kind = "camera" if lens == "pinhole" else "wideanglecamera"
        camera = element(base, "sensor", name="camera", type=kind)
        element(camera, "topic", "/robotlab/camera")
        element(camera, "update_rate", sensors["camera"]["rate_hz"])
        element(camera, "always_on", "true")
        camera = element(camera, "camera")
        element(camera, "horizontal_fov", sensors["camera"]["horizontal_fov_rad"])
        image = element(camera, "image")
        element(image, "width", sensors["camera"]["width"])
        element(image, "height", sensors["camera"]["height"])
        element(image, "format", "R8G8B8")
        clip = element(camera, "clip")
        element(clip, "near", sensors["camera"]["near_m"])
        element(clip, "far", sensors["camera"]["far_m"])
        error = sensors["camera"]["intensity"]
        add_noise(camera, error["bias"], error["noise_stddev"])
        if lens != "pinhole":
            # Native Gazebo fisheye; verified to render headless with ogre2.
            optics = element(camera, "lens")
            element(optics, "type", lens)
            element(optics, "scale_to_hfov", "true")
            element(optics, "cutoff_angle", sensors["camera"]["horizontal_fov_rad"] / 2)
            element(optics, "env_texture_size", 512)
    if sensors["lidar"]["enabled"]:
        lidar = sensors["lidar"]
        vertical = None
        if lidar["vertical_samples"] > 1:
            vertical = (
                lidar["vertical_samples"],
                math.radians(lidar["vertical_min_angle_deg"]),
                math.radians(lidar["vertical_max_angle_deg"]),
            )
        add_ray_sensor(
            base,
            "lidar",
            lidar,
            horizontal=(lidar["samples"], -3.14159265, 3.14159265),
            vertical=vertical,
        )
    if sensors["rangefinder"]["enabled"]:
        # One ray along the sensor's +x axis; the mount points it down.
        add_ray_sensor(
            base, "rangefinder", sensors["rangefinder"], horizontal=(1, 0, 0)
        )
    if sensors["magnetometer"]["enabled"]:
        add_magnetometer(base, sensors["magnetometer"])

    for sensor in list(base.findall("sensor")):
        name = sensor.get("name")
        if name in MOUNTED_SENSORS:
            element(sensor, "pose", " ".join(map(str, mount_pose(sensors[name]))))
        if not sensors[
            "contacts" if sensor.get("name") == "contact" else sensor.get("name")
        ]["enabled"]:
            base.remove(sensor)

    rotors = None
    if control_mode == "velocity":
        gains = settings["velocity_controller"]
        control = plugin(drone, "multicopter-control", "MulticopterVelocityControl")
        for key, value in dict(
            robotNamespace="drone",
            commandSubTopic="cmd_vel",
            enableSubTopic="enable",
            comLinkName="base_link",
            velocityGain=" ".join(map(str, gains["velocity_gain"])),
            attitudeGain=" ".join(map(str, gains["attitude_gain"])),
            angularRateGain=" ".join(map(str, gains["angular_rate_gain"])),
            maximumLinearAcceleration=" ".join(
                map(str, gains["max_acceleration_mps2"])
            ),
        ).items():
            element(control, key, value)
        rotors = element(control, "rotorConfiguration")
    if rotors is not None and len(properties.lift_rotors) != len(properties.rotors):
        raise ValueError("Velocity mode needs every rotor vertical; use rotor-thrust")
    thrust_scale = rotor_scale(motors, "rotor_thrust_scale", len(properties.rotors))
    moment_scale = rotor_scale(motors, "rotor_moment_scale", len(properties.rotors))
    for index, ((x, y, direction), position, axis) in enumerate(
        zip(properties.rotors, properties.positions, properties.axes)
    ):
        name = f"rotor_{index}"
        link = element(drone, "link", name=name)
        # The motor model pushes along the rotor link's +z (checked in Gazebo),
        # so the link is rotated to point +z along the thrust axis.
        roll, pitch, yaw = _axis_rpy(axis)
        element(link, "pose", " ".join(map(str, (*position, roll, pitch, yaw))))
        inertial = element(link, "inertial")
        element(inertial, "mass", properties.rotor_mass)
        inertia = element(inertial, "inertia")
        for axis in ["ixx", "iyy", "izz"]:
            element(inertia, axis, properties.rotor_inertia)
        diameter = properties.propeller_diameter
        box(
            link,
            "blade",
            [diameter, diameter / 10, 0.01],
            ".1 .1 .1 1",
            collision=False,
        )
        joint = element(drone, "joint", name=name + "_joint", type="revolute")
        element(joint, "parent", "base_link")
        element(joint, "child", name)
        element(element(joint, "axis"), "xyz", "0 0 1")
        motor = plugin(drone, "multicopter-motor-model", "MulticopterMotorModel")
        values = dict(
            robotNamespace="drone",
            jointName=name + "_joint",
            linkName=name,
            turningDirection="ccw" if direction == 1 else "cw",
            timeConstantUp=motors["time_constant_up_s"],
            timeConstantDown=motors["time_constant_down_s"],
            maxRotVelocity=(properties.max_rotor_thrust / properties.thrust_coefficient)
            ** 0.5,
            # True per-rotor constants (manufacturing spread, randomization).
            motorConstant=properties.thrust_coefficient * thrust_scale[index],
            momentConstant=properties.moment_ratio * moment_scale[index],
            commandSubTopic="command/motor_speed",
            actuator_number=index,
            rotorDragCoefficient=motors["rotor_drag_coefficient"],
            rollingMomentCoefficient=motors["rolling_moment_coefficient"],
            rotorVelocitySlowdownSim=ROTOR_VELOCITY_SLOWDOWN,
            motorType="velocity",
        )
        for key, value in values.items():
            element(motor, key, value)
        if rotors is not None:
            rotor = element(rotors, "rotor")
            for key, value in dict(
                jointName=name + "_joint",
                forceConstant=properties.thrust_coefficient,
                momentConstant=properties.moment_ratio,
                direction=direction,
            ).items():
                element(rotor, key, value)
    if settings["payload"]["enabled"]:
        tethered = settings["payload"]["tether"]["enabled"]
        if tethered:
            add_tether(drone, settings["payload"]["tether"])
        joint = plugin(drone, "detachable-joint", "DetachableJoint")
        for key, value in dict(
            parent_link="hook" if tethered else "base_link",
            child_model="payload",
            child_link="link",
            detach_topic="/robotlab/payload/detach",
            attach_topic="/robotlab/payload/attach",
        ).items():
            element(joint, key, value)
    if motors["esc"]["telemetry"]["enabled"]:
        # Rotor joint velocities (actual speed / slowdown) for ESC telemetry.
        states = plugin(drone, "joint-state-publisher", "JointStatePublisher")
        element(states, "topic", ROTOR_STATE_TOPIC)
        for index in range(len(properties.rotors)):
            element(states, "joint_name", f"rotor_{index}_joint")
    odometry = plugin(drone, "odometry-publisher", "OdometryPublisher")
    element(odometry, "dimensions", 3)
    element(odometry, "odom_publish_frequency", settings["control"]["odometry_rate_hz"])
    element(odometry, "odom_topic", "/robotlab/odometry")
    return drone


# SDF sensors that take a configured body-relative mount pose.
MOUNTED_SENSORS = {
    "imu",
    "gps",
    "camera",
    "lidar",
    "baro",
    "rangefinder",
    "magnetometer",
}


def add_ray_sensor(base, name, settings, *, horizontal, vertical=None):
    """Gazebo gpu_lidar publishing LaserScan on /robotlab/<name>.

    horizontal and vertical are (samples, min_angle_rad, max_angle_rad);
    vertical None gives a single scan plane. Requires the rendering system.
    """
    sensor = element(base, "sensor", name=name, type="gpu_lidar")
    element(sensor, "topic", f"/robotlab/{name}")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", settings["rate_hz"])
    lidar = element(sensor, "lidar")
    scan = element(lidar, "scan")
    for axis, values in (("horizontal", horizontal), ("vertical", vertical)):
        if values is None:
            continue
        samples, low, high = values
        block = element(scan, axis)
        for key, value in dict(
            samples=samples, resolution=1, min_angle=low, max_angle=high
        ).items():
            element(block, key, value)
    ranges = element(lidar, "range")
    for key, value in dict(
        min=settings["min_range_m"], max=settings["max_range_m"], resolution=0.01
    ).items():
        element(ranges, key, value)
    error = settings["range_m"]
    add_noise(lidar, error["bias"], error["noise_stddev"])


def add_magnetometer(base, settings):
    """Gazebo magnetometer on /robotlab/magnetometer; native noise µT -> tesla.

    It reports the world <magnetic_field> rotated into the sensor axes.
    """
    sensor = element(base, "sensor", name="magnetometer", type="magnetometer")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", settings["rate_hz"])
    element(sensor, "topic", "/robotlab/magnetometer")
    magnetometer = element(sensor, "magnetometer")
    error = settings["field_ut"]
    for i, axis in enumerate("xyz"):
        add_noise(
            element(magnetometer, axis),
            error["bias"][i] * 1e-6,
            error["noise_stddev"][i] * 1e-6,
        )


def add_barometer(base, settings):
    """Gazebo air-pressure sensor; pressure in Pa from the standard atmosphere.

    Gazebo reports pressure for the sensor's world z treated as height above
    sea level. The installed gz-sensors 8 ignores <reference_altitude>, so it
    is omitted; the mission shifts pressure to the world elevation in Python
    (sensors/gazebo.py, baro_datum_message). Opening this SDF directly gives
    the sea-level datum.
    """
    baro = element(base, "sensor", name="baro", type="air_pressure")
    element(baro, "always_on", "true")
    element(baro, "update_rate", settings["rate_hz"])
    element(baro, "topic", "/robotlab/baro")
    air = element(baro, "air_pressure")
    error = settings["pressure_pa"]
    add_noise(element(air, "pressure"), error["bias"], error["noise_stddev"])


def spawn_position(settings, terrain=None):
    """Configured spawn, or resting on the ground (body box bottom on terrain).

    A 2 mm clearance avoids starting interpenetrated with the ground. With
    terrain, the ground height at the spawn x, y is used.
    """
    x, y, z = settings["simulation"]["spawn_position_m"]
    if settings["simulation"]["start_on_ground"]:
        ground = 0.0 if terrain is None else float(terrain.height(x, y))
        z = ground + settings["vehicle"]["body_size_m"][2] / 2 + 0.002
        payload = settings["payload"]
        if payload["enabled"]:
            # Stand on the payload when it hangs below the body.
            z = max(
                z, ground + payload["size_m"][2] / 2 - payload["offset_m"][2] + 0.002
            )
    return [x, y, z]


HOOK_RADIUS_M = 0.03


def hook_offset(payload):
    """Hook position relative to the body origin at start (cable straight down)."""
    tether = payload["tether"]
    if not tether["enabled"]:
        return np.asarray(payload["offset_m"], float)
    return np.asarray(tether["anchor_m"], float) - [
        0.0,
        0.0,
        tether["initial_length_m"],
    ]


def payload_offset(payload):
    """Payload centre relative to the body origin at start."""
    if not payload["tether"]["enabled"]:
        return np.asarray(payload["offset_m"], float)
    below = HOOK_RADIUS_M + payload["size_m"][2] / 2
    return hook_offset(payload) - [0.0, 0.0, below]


def add_tether(drone, tether):
    """Cable as a variable-length spherical pendulum ending in a hook.

    A ball joint at the winch anchor lets the cable swing; a prismatic joint
    along it is the winch (joint position = length − initial length), driven
    by JointPositionController on /model/drone/joint/winch/0/cmd_pos. The
    cable is rigid (cannot go slack). Effort limits are required for velocity
    commands to respect the joint limits (Gazebo error otherwise).
    """
    anchor = np.asarray(tether["anchor_m"], float)
    cable = element(drone, "link", name="tether")
    element(cable, "pose", " ".join(map(str, anchor)) + " 0 0 0")
    small_inertial(cable, 0.01)
    ball = element(drone, "joint", name="tether_ball", type="ball")
    element(ball, "parent", "base_link")
    element(ball, "child", "tether")
    damping = element(element(ball, "axis"), "dynamics")
    element(damping, "damping", tether["swing_damping"])
    hook = element(drone, "link", name="hook")
    element(
        hook,
        "pose",
        " ".join(map(str, anchor - [0, 0, tether["initial_length_m"]])) + " 0 0 0",
    )
    small_inertial(hook, 0.02)
    visual = element(hook, "visual", name="hook_visual")
    element(element(element(visual, "geometry"), "sphere"), "radius", HOOK_RADIUS_M)
    winch = element(drone, "joint", name="winch", type="prismatic")
    element(winch, "parent", "tether")
    element(winch, "child", "hook")
    axis = element(winch, "axis")
    element(axis, "xyz", "0 0 -1")
    limit = element(axis, "limit")
    element(limit, "lower", 0.1 - tether["initial_length_m"])
    element(limit, "upper", tether["max_length_m"] - tether["initial_length_m"])
    element(limit, "effort", 1e6)
    controller = plugin(drone, "joint-position-controller", "JointPositionController")
    for key, value in dict(
        joint_name="winch",
        use_velocity_commands="true",
        p_gain=5.0,
        cmd_max=tether["winch_speed_mps"],
        cmd_min=-tether["winch_speed_mps"],
    ).items():
        element(controller, key, value)


def small_inertial(link, mass_kg):
    inertial = element(link, "inertial")
    element(inertial, "mass", mass_kg)
    inertia = element(inertial, "inertia")
    for key in ("ixx", "iyy", "izz"):
        element(inertia, key, 1e-5)


def add_payload(world, settings, terrain=None):
    """Payload model under the hook of the spawned drone (attached at start)."""
    payload = settings["payload"]
    x, y, z = np.asarray(spawn_position(settings, terrain)) + payload_offset(payload)
    model = element(world, "model", name="payload")
    element(model, "pose", f"{x} {y} {z} 0 0 0")
    link = element(model, "link", name="link")
    inertial = element(link, "inertial")
    element(inertial, "mass", payload["mass_kg"])
    sx, sy, sz = payload["size_m"]
    inertia = element(inertial, "inertia")
    for key, value in (
        ("ixx", sy * sy + sz * sz),
        ("iyy", sx * sx + sz * sz),
        ("izz", sx * sx + sy * sy),
    ):
        element(inertia, key, payload["mass_kg"] * value / 12)
    box(link, "box", payload["size_m"], ".95 .75 .1 1")


def add_noise(parent, bias, stddev):
    """SDF Gaussian mean implements a signed deterministic bias."""
    if bias == 0 and stddev == 0:
        return
    if parent.tag in {"camera", "lidar"}:
        model = element(parent, "noise")
        element(model, "type", "gaussian")
    else:
        model = element(parent, "noise", type="gaussian")
    element(model, "mean", bias)
    element(model, "stddev", stddev)
