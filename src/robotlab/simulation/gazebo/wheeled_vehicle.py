"""SDF for a ray-cast wheeled vehicle (rover, car, tractor).

One rigid link, ``base_link``, carries the whole vehicle mass and inertia
at its origin (the centre of mass; odometry and Python wrenches refer to
it). It has

- the body collision box ``body`` (watched by the contact sensor, so a
  rollover or bottoming out is reported like a drone crash);
- per wheel a collision cylinder ``wheel_<name>`` at the hub position with
  the suspension fully compressed: it touches the ground only when the
  suspension bottoms out (a bump stop), and it stops the vehicle at walls;
- wheel visuals at static ride height (Gazebo does not spin or steer them;
  Rerun draws the live wheel state);
- the configured sensors (``vehicle.add_body_sensors``) and the
  OdometryPublisher on ``/robotlab/odometry``.

Wheel forces are not Gazebo contacts: Python applies them through
ApplyLinkWrench (see simulation/wheeled/model.py).
"""

import math
import xml.etree.ElementTree as ET

from ...core.wheeled_vehicle import WheeledProperties
from ..wheeled.spawn import SpawnPose
from .sdf import box, element, plugin
from .vehicle import add_body_sensors

KIND_COLORS = {
    "rover": ".85 .55 .1 1",
    "car": ".8 .1 .1 1",
    "tractor": ".15 .55 .25 1",
}
TYRE_COLOR = ".08 .08 .08 1"
# Opponent body colours, in order (the ego keeps its kind colour).
OPPONENT_COLORS = (
    ".1 .3 .85 1",
    ".95 .75 .1 1",
    ".55 .2 .7 1",
    ".1 .65 .65 1",
    ".9 .45 .1 1",
    ".35 .35 .35 1",
    ".95 .95 .95 1",
)
OPPONENT_ODOMETRY_TOPIC = "/robotlab/opponents/{name}/odometry"
# Body-frame velocity commands of a kinematic opponent (VelocityControl).
OPPONENT_COMMAND_TOPIC = "/robotlab/opponents/{name}/cmd_vel"
IMPLEMENT_LIFT_TOPIC = "/robotlab/implement/lift"
IMPLEMENT_COLOR = ".75 .2 .1 1"
# Implement lift servo: joint velocity = gain × position error (1/s), capped.
LIFT_SERVO_GAIN_PER_S = 50.0


TRAILER_JOINT_TOPIC = "/robotlab/trailer/joint_state"
TRAILER_COLOR = ".55 .4 .2 1"


def add_trailer(
    model: ET.Element, trailer: dict, properties: WheeledProperties
) -> None:
    """Towed trailer: links ``hitch_yaw`` and ``trailer`` on revolute yaw
    (body z) and pitch joints at the hitch; joint states on
    ``TRAILER_JOINT_TOPIC`` every step. Its wheel forces are Python's (one
    instantaneous wrench on ``<model>::trailer``); the trailer box and wheel
    bump stops collide in Gazebo."""
    hx, hy, hz = trailer["hitch_position_m"]
    hitch = element(model, "link", name="hitch_yaw")
    element(hitch, "pose", _pose((hx, hy, hz, 0, 0, 0)))
    inertial = element(hitch, "inertial")
    element(inertial, "mass", 1.0)
    inertia = element(inertial, "inertia")
    for key in ("ixx", "iyy", "izz"):
        element(inertia, key, 0.01)
    link = element(model, "link", name="trailer")
    centre = (hx - trailer["drawbar_m"], hy, hz - trailer["cg_below_hitch_m"])
    element(link, "pose", _pose((*centre, 0, 0, 0)))
    inertial = element(link, "inertial")
    element(inertial, "mass", trailer["mass_kg"])
    inertia = element(inertial, "inertia")
    for key, value in zip(("ixx", "iyy", "izz"), trailer["inertia_kg_m2"]):
        element(inertia, key, value)
    box(link, "trailer_body", trailer["size_m"], TRAILER_COLOR)
    drawbar = trailer["drawbar_m"] - trailer["size_m"][0] / 2
    box(
        link,
        "drawbar",
        [drawbar, 0.12, 0.12],
        ".2 .2 .2 1",
        _pose(
            (
                trailer["size_m"][0] / 2 + drawbar / 2,
                0,
                trailer["cg_below_hitch_m"],
                0,
                0,
                0,
            )
        ),
        collision=False,
    )
    radius, width = trailer["wheel_radius_m"], properties.wheel_width_m
    for wheel in trailer["wheels"]:
        x, y, z = wheel["position_m"]
        _cylinder(
            link,
            f"trailer_wheel_{wheel['name']}",
            "collision",
            radius,
            width,
            _pose((x, y, z + properties.suspension.travel_m, 1.5707963, 0, 0)),
        )
        visual = _cylinder(
            link,
            f"trailer_wheel_{wheel['name']}_visual",
            "visual",
            radius,
            width,
            _pose((x, y, z + properties.ride_drop_m, 1.5707963, 0, 0)),
        )
        material = element(visual, "material")
        element(material, "ambient", TYRE_COLOR)
        element(material, "diffuse", TYRE_COLOR)
    for name, parent, child, axis, limit, pose in (
        (
            "trailer_yaw",
            "base_link",
            "hitch_yaw",
            "0 0 1",
            trailer["yaw_limit_deg"],
            (0, 0, 0),
        ),
        (
            "trailer_pitch",
            "hitch_yaw",
            "trailer",
            "0 1 0",
            trailer["pitch_limit_deg"],
            (trailer["drawbar_m"], 0, trailer["cg_below_hitch_m"]),
        ),
    ):
        joint = element(model, "joint", name=name, type="revolute")
        element(joint, "pose", _pose((*pose, 0, 0, 0)))
        element(joint, "parent", parent)
        element(joint, "child", child)
        axis_element = element(joint, "axis")
        element(axis_element, "xyz", axis)
        limits = element(axis_element, "limit")
        element(limits, "lower", -math.radians(limit))
        element(limits, "upper", math.radians(limit))
    states = plugin(model, "joint-state-publisher", "JointStatePublisher")
    element(states, "topic", TRAILER_JOINT_TOPIC)
    for name in ("trailer_yaw", "trailer_pitch"):
        element(states, "joint_name", name)


def add_implement(model: ET.Element, implement: dict) -> None:
    """Rear hitch implement: a visual-only link (soil forces are Python's)
    on a vertical prismatic joint whose position follows the lift published
    on ``IMPLEMENT_LIFT_TOPIC`` (metres, ≤ 0 below raised)."""
    size = implement["size_m"]
    link = element(model, "link", name="implement")
    element(link, "pose", _pose((*implement["hitch_position_m"], 0, 0, 0)))
    inertial = element(link, "inertial")
    mass = implement["mass_kg"]
    element(inertial, "mass", mass)
    inertia = element(inertial, "inertia")
    x, y, z = size
    for key, value in (
        ("ixx", y * y + z * z),
        ("iyy", x * x + z * z),
        ("izz", x * x + y * y),
    ):
        element(inertia, key, mass * value / 12)
    # Frame beam on top and tines reaching down to the tool point.
    box(
        link,
        "frame",
        [x * 0.3, y, z * 0.2],
        IMPLEMENT_COLOR,
        _pose((0, 0, z * 0.4, 0, 0, 0)),
        collision=False,
    )
    for side in (-0.35, 0.0, 0.35):
        box(
            link,
            f"tine_{side:+.2f}",
            [0.06, 0.06, z * 0.8],
            ".2 .2 .2 1",
            _pose((0, side * y, -z * 0.1, 0, 0, 0)),
            collision=False,
        )
    joint = element(model, "joint", name="implement_lift", type="prismatic")
    element(joint, "parent", "base_link")
    element(joint, "child", "implement")
    axis = element(joint, "axis")
    element(axis, "xyz", "0 0 1")
    limit = element(axis, "limit")
    element(limit, "lower", -implement["lift_travel_m"] - 0.05)
    element(limit, "upper", 0.05)
    servo = plugin(model, "joint-position-controller", "JointPositionController")
    for key, value in (
        ("joint_name", "implement_lift"),
        ("topic", IMPLEMENT_LIFT_TOPIC),
        ("use_velocity_commands", "true"),
        ("p_gain", LIFT_SERVO_GAIN_PER_S),
        ("cmd_max", 4 * implement["lift_rate_mps"] + 0.5),
        ("cmd_min", -(4 * implement["lift_rate_mps"] + 0.5)),
        ("initial_position", 0.0),
    ):
        element(servo, key, value)


def opponent_name(properties: WheeledProperties, index: int) -> str:
    """Gazebo model name of opponent ``index`` (1-based): ``car_1``, ..."""
    return f"{properties.model_name}_{index}"


# Friction of the bump-stop cylinders (rubber on ground, sliding).
BUMP_STOP_FRICTION = 0.7


def _pose(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _cylinder(link, name, kind, radius, length, pose):
    shape = element(link, kind, name=name)
    element(shape, "pose", pose)
    geometry = element(element(shape, "geometry"), "cylinder")
    element(geometry, "radius", radius)
    element(geometry, "length", length)
    return shape


def _add_wheels(
    base: ET.Element, properties: WheeledProperties, collisions: bool = True
) -> None:
    """Bump-stop collisions at full compression (unless ``collisions`` is
    False); visuals at ride height."""
    radius, width = properties.tyre.radius_m, properties.wheel_width_m
    travel, sag = properties.suspension.travel_m, properties.ride_drop_m
    for wheel in properties.wheels:
        x, y, z = wheel.position_m
        # Cylinder axis is its z: roll by 90° puts it along body y.
        if collisions:
            stop = _cylinder(
                base,
                f"wheel_{wheel.name}",
                "collision",
                radius,
                width,
                _pose((x, y, z + travel, 1.5707963, 0, 0)),
            )
            friction = element(element(element(stop, "surface"), "friction"), "ode")
            element(friction, "mu", BUMP_STOP_FRICTION)
            element(friction, "mu2", BUMP_STOP_FRICTION)
        visual = _cylinder(
            base,
            f"wheel_{wheel.name}_visual",
            "visual",
            radius,
            width,
            _pose((x, y, z + min(sag, travel), 1.5707963, 0, 0)),
        )
        material = element(visual, "material")
        element(material, "ambient", TYRE_COLOR)
        element(material, "diffuse", TYRE_COLOR)


def build_wheeled_vehicle(
    world: ET.Element,
    properties: WheeledProperties,
    settings: dict,
    pose: SpawnPose,
    *,
    name: str | None = None,
    sensors: bool = True,
    odometry_topic: str = "/robotlab/odometry",
    color: str | None = None,
    kinematic_rate_hz: float | None = None,
) -> ET.Element:
    """Add the vehicle model to ``world`` (see module docs).

    Opponents pass their ``name``, ``sensors=False`` (truth odometry only,
    on ``odometry_topic``), and a body ``color``. Kinematic opponents also
    pass ``kinematic_rate_hz``: no gravity, a VelocityControl system on
    ``OPPONENT_COMMAND_TOPIC``, and odometry at that rate.
    """
    model = element(world, "model", name=name or properties.model_name)
    element(model, "pose", _pose((*pose.position_m, 0, 0, pose.yaw_rad)))
    base = element(model, "link", name="base_link")
    inertial = element(base, "inertial")
    element(inertial, "mass", properties.mass_kg)
    inertia = element(inertial, "inertia")
    for key, value in zip(("ixx", "iyy", "izz"), properties.inertia_kg_m2):
        element(inertia, key, value)
    size, offset = properties.collision_size_m, properties.collision_offset_m
    box(
        base,
        "body",
        list(size),
        KIND_COLORS[properties.kind],
        _pose((*offset, 0, 0, 0)),
    )
    # Heading marker: a light stripe on the front half of the body top.
    box(
        base,
        "nose",
        [size[0] / 2, size[1] / 6, 0.01],
        ".95 .95 .95 1",
        _pose(
            (
                offset[0] + size[0] / 4,
                offset[1],
                offset[2] + size[2] / 2 + 0.005,
                0,
                0,
                0,
            )
        ),
        collision=False,
    )
    # Kinematic opponents are obstacles: the body box is their only
    # collision (fewer contact pairs for Gazebo).
    _add_wheels(base, properties, collisions=kinematic_rate_hz is None)
    if sensors:
        add_body_sensors(base, settings["sensors"])
    implement = settings["ground_vehicle"]["implement"]
    if sensors and implement["enabled"]:
        add_implement(model, implement)
    trailer = settings["ground_vehicle"]["trailer"]
    if sensors and trailer["enabled"]:
        add_trailer(model, trailer, properties)
    rate_hz = settings["control"]["odometry_rate_hz"]
    if kinematic_rate_hz is not None:
        # Moving obstacle: Python commands the body velocity (including the
        # vertical ground-following term), so gravity stays off.
        element(base, "gravity", "false")
        velocity = plugin(model, "velocity-control", "VelocityControl")
        element(
            velocity, "topic", OPPONENT_COMMAND_TOPIC.format(name=model.get("name"))
        )
        rate_hz = kinematic_rate_hz
    odometry = plugin(model, "odometry-publisher", "OdometryPublisher")
    element(odometry, "dimensions", 3)
    element(odometry, "odom_publish_frequency", rate_hz)
    element(odometry, "odom_topic", odometry_topic)
    return model
