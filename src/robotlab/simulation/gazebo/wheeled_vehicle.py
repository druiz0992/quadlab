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

import xml.etree.ElementTree as ET

from ...core.wheeled_vehicle import WheeledProperties
from ..wheeled.spawn import SpawnPose
from .sdf import box, element
from .sdf import plugin
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


def _add_wheels(base: ET.Element, properties: WheeledProperties) -> None:
    """Bump-stop collisions at full compression; visuals at ride height."""
    radius, width = properties.tyre.radius_m, properties.wheel_width_m
    travel, sag = properties.suspension.travel_m, properties.ride_drop_m
    for wheel in properties.wheels:
        x, y, z = wheel.position_m
        # Cylinder axis is its z: roll by 90° puts it along body y.
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
) -> ET.Element:
    """Add the vehicle model to ``world`` (see module docs).

    Opponents pass their ``name``, ``sensors=False`` (truth odometry only,
    on ``odometry_topic``), and a body ``color``.
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
    _add_wheels(base, properties)
    if sensors:
        add_body_sensors(base, settings["sensors"])
    odometry = plugin(model, "odometry-publisher", "OdometryPublisher")
    element(odometry, "dimensions", 3)
    element(odometry, "odom_publish_frequency", settings["control"]["odometry_rate_hz"])
    element(odometry, "odom_topic", odometry_topic)
    return model
