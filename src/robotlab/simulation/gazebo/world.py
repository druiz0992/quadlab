"""Generate a portable Gazebo Harmonic world from editable JSON boxes."""

import json
from ...sensors.lidar_returns import RETRO_PER_REFLECTIVITY
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .sdf import element, plugin, box, textured_box
from .textures import (
    texture_image,
    validate_color,
    validate_texture,
    write_box_obj,
    write_png,
)
from ..moving_obstacles import validate_moving_obstacle
from ..terrain import Terrain, validate_terrain
from .plugins import LOCKSTEP_FILENAME, LOCKSTEP_NAME, lockstep_barrier
from .vehicle import add_payload, build_drone
from .wheeled_vehicle import (
    OPPONENT_COLORS,
    OPPONENT_ODOMETRY_TOPIC,
    build_wheeled_vehicle,
    opponent_name,
)
from ...core.track import Track
from ..wheeled.config import WHEELED_KINDS
from ..wheeled.ground import GroundModel, surfaces_from_scene, validate_surfaces
from ..wheeled.spawn import (
    kinematic_spawn_pose,
    spawn_pose,
    spawn_xy_yaw,
    validate_spawn_points,
)
from .config import WORLD_ELEVATION_M, load_drone_config, wind_effects_enabled

# Physics engine, written explicitly so a Gazebo upgrade cannot change it
# silently. These are gz-sim 8 / gz-physics 7's own defaults (verified: an
# unsupported detector logs "defaulting to [ode]"; lockstep outputs are
# byte-identical with and without them). See docs/2_simulation.md.
PHYSICS_ENGINE_PLUGIN = "gz-physics-dartsim-plugin"
DART_COLLISION_DETECTOR = "ode"
DART_SOLVER = "dantzig"


# Model names the generated world uses itself (vehicles, ground, terrain).
RESERVED_NAMES = frozenset({"drone", "ground", "terrain", *WHEELED_KINDS})


def load_scene(path):
    scene = json.loads(Path(path).read_text())
    names = set()
    for obstacle in scene["obstacles"]:
        name = obstacle["name"]
        if not name.isidentifier() or name in names or name in RESERVED_NAMES:
            raise ValueError(f"Invalid or duplicate obstacle name: {name}")
        names.add(name)
        for field in ["position", "size"]:
            values = np.asarray(obstacle[field], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name}: {field} must have three finite values")
        if min(obstacle["size"]) <= 0:
            raise ValueError(f"{name}: sizes must be positive")
        _check_reflectivity(obstacle.get("reflectivity"), name)
        _check_appearance(obstacle, name)
        moment = obstacle.get("magnetic_moment_am2")
        if moment is not None and (
            np.asarray(moment, dtype=float).shape != (3,)
            or not np.isfinite(np.asarray(moment, dtype=float)).all()
        ):
            raise ValueError(f"{name}: magnetic_moment_am2 needs three finite values")
    _check_reflectivity(scene.get("ground_reflectivity"), "ground_reflectivity")
    if "ground_texture" in scene:
        scene["ground_texture"] = validate_texture(
            scene["ground_texture"], "ground_texture"
        )
    size = scene.get("ground_size_m", 100.0)
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, float))
        or not 10 <= size <= 20000
    ):
        raise ValueError("ground_size_m must be between 10 and 20000 m")
    if "terrain" in scene:
        scene["terrain"] = validate_terrain(scene["terrain"])
    for entry in scene.get("moving_obstacles", []):
        validate_moving_obstacle(entry)
        name = entry["name"]
        if not name.isidentifier() or name in names or name in RESERVED_NAMES:
            raise ValueError(f"Invalid or duplicate obstacle name: {name}")
        names.add(name)
    # Wheeled-robot scene keys (ignored by multirotors): friction zones and
    # start positions; see simulation/wheeled/.
    validate_surfaces(scene)
    validate_spawn_points(scene.get("spawn_points", []))
    Track.from_scene(scene)
    return scene


def _check_appearance(obstacle, name):
    """Optional ``texture``, ``color`` (RGB 0..1), and ``visual_only`` (a
    marking without collision, e.g. a floor patch); validated in place."""
    if "texture" in obstacle:
        obstacle["texture"] = validate_texture(obstacle["texture"], name)
    if "color" in obstacle:
        obstacle["color"] = validate_color(obstacle["color"], f"{name}: color")
    if not isinstance(obstacle.get("visual_only", False), bool):
        raise ValueError(f"{name}: visual_only must be true or false")


def _check_reflectivity(value, name):
    """Optional lidar reflectivity: positive; above 1 means retroreflective."""
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= 10
    ):
        raise ValueError(f"{name}: reflectivity must be in (0, 10]")


def add_moving_obstacle(world, entry, mass_kg=1000.0):
    """Heavy, gravity-free box driven by Gazebo's VelocityControl system.

    The mass makes the drone, not the obstacle, yield in collisions; Python
    publishes velocity commands on /model/<name>/cmd_vel.
    """
    model = element(world, "model", name=entry["name"])
    element(model, "pose", " ".join(map(str, entry["waypoints"][0])) + " 0 0 0")
    link = element(model, "link", name="link")
    element(link, "gravity", "false")
    inertial = element(link, "inertial")
    element(inertial, "mass", mass_kg)
    x, y, z = entry["size"]
    inertia = element(inertial, "inertia")
    for key, value in (
        ("ixx", y * y + z * z),
        ("iyy", x * x + z * z),
        ("izz", x * x + y * y),
    ):
        element(inertia, key, mass_kg * value / 12)
    box(link, "box", entry["size"], ".75 .25 .2 1")
    plugin(model, "velocity-control", "VelocityControl")
    return model


def moving_obstacle_sdf(entry):
    """Standalone SDF text for spawning a moving obstacle at run time."""
    root = ET.Element("sdf", version="1.9")
    add_moving_obstacle(root, entry)
    return ET.tostring(root, encoding="unicode")


def scene_terrain(scene):
    """Terrain object for a loaded scene, or None for flat ground."""
    return Terrain(scene["terrain"]) if "terrain" in scene else None


def add_terrain(world, terrain, mesh_path, reflectivity=None):
    """Static mesh model for the terrain; collision and visual share the mesh."""
    terrain.write_obj(mesh_path)
    model = element(world, "model", name="terrain")
    element(model, "static", "true")
    link = element(model, "link", name="link")
    for kind in ("collision", "visual"):
        shape = element(link, kind, name=f"terrain_{kind}")
        mesh = element(element(shape, "geometry"), "mesh")
        element(mesh, "uri", Path(mesh_path).resolve().as_uri())
        if kind == "visual":
            material = element(shape, "material")
            element(material, "ambient", ".42 .5 .35 1")
            element(material, "diffuse", ".42 .5 .35 1")
            if reflectivity is not None:
                element(shape, "laser_retro", RETRO_PER_REFLECTIVITY * reflectivity)


GROUND_COLOR = ".45 .5 .45 1"


OBSTACLE_COLOR = ".65 .45 .3 1"


def add_box(
    link,
    name,
    size,
    color,
    texture,
    texture_dir: Path,
    pose="0 0 0 0 0 0",
    collision=True,
    reflectivity=None,
) -> None:
    """A plain box, or with ``texture`` a textured one: the tile PNG and
    box mesh are written to ``texture_dir`` as ``<name>.png``/``.obj``."""
    if texture is None:
        box(link, name, size, color, pose, collision, reflectivity)
        return
    texture_dir.mkdir(parents=True, exist_ok=True)
    image = (texture_dir / f"{name}.png").resolve()
    mesh = (texture_dir / f"{name}.obj").resolve()
    write_png(image, texture_image(texture))
    write_box_obj(mesh, size, texture["tile_m"])
    textured_box(link, name, size, mesh, image, pose, collision, reflectivity)


def add_obstacle(world, obstacle, texture_dir: Path) -> None:
    """Static box obstacle with its colour, texture, and collision setting."""
    name = obstacle["name"]
    model = element(world, "model", name=name)
    element(model, "static", "true")
    element(model, "pose", " ".join(map(str, obstacle["position"])) + " 0 0 0")
    color = obstacle.get("color")
    add_box(
        element(model, "link", name="link"),
        "box",
        obstacle["size"],
        OBSTACLE_COLOR if color is None else " ".join(map(str, [*color, 1])),
        obstacle.get("texture"),
        texture_dir / name,
        collision=not obstacle.get("visual_only", False),
        reflectivity=obstacle.get("reflectivity"),
    )


def add_ground(world: ET.Element, scene: dict, texture_dir: Path) -> None:
    """Static ground: one 0.1 m box with its top at z = 0, or, when the
    terrain has ditches (cut below z = 0), a frame of four boxes around the
    terrain square so the terrain mesh is the only surface inside it.
    ``ground_texture`` textures the top (and each strip) of the ground."""
    size = scene.get("ground_size_m", 100.0)
    texture = scene.get("ground_texture")
    model = element(world, "model", name="ground")
    element(model, "static", "true")
    link = element(model, "link", name="link")
    terrain = scene.get("terrain")
    reflectivity = scene.get("ground_reflectivity")
    if not terrain or not terrain.get("ditches"):
        add_box(
            link,
            "ground_collision",
            [size, size, 0.1],
            GROUND_COLOR,
            texture,
            texture_dir / "ground",
            "0 0 -.05 0 0 0",
            reflectivity=reflectivity,
        )
        return
    half_x, half_y = (v / 2 for v in terrain["size_m"])
    outer = size / 2
    # Two full-width strips (north/south) and two between them (east/west).
    strips = (
        ("north", (0, (outer + half_y) / 2), (size, outer - half_y)),
        ("south", (0, -(outer + half_y) / 2), (size, outer - half_y)),
        ("east", ((outer + half_x) / 2, 0), (outer - half_x, 2 * half_y)),
        ("west", (-(outer + half_x) / 2, 0), (outer - half_x, 2 * half_y)),
    )
    for name, (x, y), (length_x, length_y) in strips:
        add_box(
            link,
            f"ground_{name}",
            [length_x, length_y, 0.1],
            GROUND_COLOR,
            texture,
            texture_dir / "ground",
            f"{x:.6g} {y:.6g} -.05 0 0 0",
            reflectivity=reflectivity,
        )


def world_elevation_m(path):
    """Spherical-coordinates elevation of an SDF world in metres (0 if absent)."""
    text = ET.parse(path).findtext("./world/spherical_coordinates/elevation")
    return 0.0 if text is None else float(text)


SURFACE_COLORS = {
    "asphalt": ".22 .22 .24 1",
    "tarmac": ".22 .22 .24 1",
    "kerb": ".85 .2 .2 1",
    "gravel": ".72 .66 .52 1",
    "grass": ".35 .6 .3 1",
    "mud": ".4 .3 .2 1",
    "soil": ".5 .38 .25 1",
    "sand": ".86 .78 .55 1",
    "ice": ".85 .92 .98 1",
    "snow": ".95 .95 .97 1",
}
SURFACE_DEFAULT_COLOR = ".55 .55 .5 1"
# Surface visuals float this far above z = 0 (and above earlier zones).
SURFACE_LAYER_M = 0.002


def zone_mesh(zone):
    """Flat triangle mesh of a surface zone: vertices (n×2) and faces (m×3).

    Rectangles are two triangles; strips are a ribbon of quads whose edge
    points are offset along the mitred normal at every vertex (miter length
    capped at twice the half width for very sharp joints).
    """
    if zone.shape == "rectangle":
        c, s = np.cos(zone.yaw_rad), np.sin(zone.yaw_rad)
        hx, hy = zone.size_m[0] / 2, zone.size_m[1] / 2
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        rotated = corners @ np.array([[c, s], [-s, c]]) + np.asarray(zone.center_m)
        return rotated, np.array([[0, 1, 2], [0, 2, 3]])
    points = np.asarray(zone.points_m, float)
    closed = zone.closed
    count = len(points)
    previous = (
        np.roll(points, 1, axis=0) if closed else np.vstack([points[:1], points[:-1]])
    )
    following = (
        np.roll(points, -1, axis=0) if closed else np.vstack([points[1:], points[-1:]])
    )
    tangent_in = points - previous
    tangent_out = following - points
    if not closed:
        tangent_in[0] = tangent_out[0]
        tangent_out[-1] = tangent_in[-1]
    tangent_in /= np.maximum(np.linalg.norm(tangent_in, axis=1, keepdims=True), 1e-9)
    tangent_out /= np.maximum(np.linalg.norm(tangent_out, axis=1, keepdims=True), 1e-9)
    tangent = tangent_in + tangent_out
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    in_normal = np.column_stack([-tangent_in[:, 1], tangent_in[:, 0]])
    miter = 1 / np.maximum(np.sum(normal * in_normal, axis=1), 0.5)
    offset = (zone.width_m / 2) * miter[:, None] * normal
    vertices = np.vstack([points + offset, points - offset])
    faces = []
    for i in range(count if closed else count - 1):
        j = (i + 1) % count
        faces += [(i, j, count + j), (i, count + j, count + i)]
    return vertices, np.asarray(faces, np.int64)


def write_zone_obj(path, vertices, faces):
    """Wavefront OBJ of a flat mesh with upward normals (DART-safe format)."""
    lines = [f"v {x:.3f} {y:.3f} 0" for x, y in vertices]
    lines.append("vn 0 0 1")
    lines += [f"f {a + 1}//1 {b + 1}//1 {c + 1}//1" for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")


def add_surface_visuals(world, scene, mesh_dir):
    """Visual-only meshes for flat-ground surface zones (no collision), one
    per zone in ``mesh_dir`` (``surface_<k>.obj``), layered in zone order.

    Zones on terrain are not drawn (a flat patch would not follow the height
    field); their friction still applies.
    """
    if "terrain" in scene:
        return
    _, zones = surfaces_from_scene(scene)
    if not zones:
        return
    model = element(world, "model", name="surfaces")
    element(model, "static", "true")
    link = element(model, "link", name="link")
    for index, zone in enumerate(zones):
        color = SURFACE_COLORS.get(zone.surface.name, SURFACE_DEFAULT_COLOR)
        path = Path(mesh_dir) / f"surface_{index}.obj"
        write_zone_obj(path, *zone_mesh(zone))
        visual = element(link, "visual", name=f"zone_{index}")
        element(visual, "pose", f"0 0 {SURFACE_LAYER_M * (index + 1)} 0 0 0")
        mesh = element(element(visual, "geometry"), "mesh")
        element(mesh, "uri", path.resolve().as_uri())
        material = element(visual, "material")
        element(material, "ambient", color)
        element(material, "diffuse", color)


def _add_wheeled(world, scene, config, terrain, mesh_dir):
    """The wheeled vehicle, its wrench system, and surface visuals."""
    plugin(world, "apply-link-wrench", "ApplyLinkWrench")
    add_surface_visuals(world, scene, mesh_dir)
    ground = GroundModel.from_scene(scene, terrain)
    x, y, yaw = spawn_xy_yaw(config.data, scene)
    pose = spawn_pose(config.wheeled, ground, x, y, yaw)
    build_wheeled_vehicle(world, config.wheeled, config.data, pose)
    names = {obstacle["name"] for obstacle in scene["obstacles"]}
    opponents = config.data["ground_vehicle"]["opponents"]
    kinematic = opponents["mode"] == "kinematic"
    for index in range(1, opponents["count"] + 1):
        name = opponent_name(config.wheeled, index)
        if name in names:
            raise ValueError(f"scene obstacle name {name!r} is used by an opponent")
        x, y, yaw = spawn_xy_yaw(config.data, scene, offset=index)
        place = kinematic_spawn_pose if kinematic else spawn_pose
        build_wheeled_vehicle(
            world,
            config.wheeled,
            config.data,
            place(config.wheeled, ground, x, y, yaw),
            name=name,
            sensors=False,
            odometry_topic=OPPONENT_ODOMETRY_TOPIC.format(name=name),
            color=OPPONENT_COLORS[(index - 1) % len(OPPONENT_COLORS)],
            kinematic_rate_hz=opponents["update_rate_hz"] if kinematic else None,
        )


def build_world(
    scene, output, render_sensors=False, control_mode="velocity", config=None
):
    config = config or load_drone_config(render_sensors=render_sensors)
    root = ET.Element("sdf", version="1.9")
    world = element(root, "world", name="robotlab")
    # Project convention (and the EKF): 9.81 m/s². Gazebo's default is 9.8.
    element(world, "gravity", "0 0 -9.81")
    physics = element(world, "physics", name="fixed", type="dart")
    element(physics, "max_step_size", config.data["simulation"]["step_s"])
    # Wheeled robots: Gazebo runs bursts unthrottled and Python paces the
    # run to real_time_factor (its own processing would otherwise add to
    # Gazebo's paced time; experiments/wheeled_mission.py).
    element(
        physics,
        "real_time_factor",
        (
            0
            if config.wheeled is not None
            else config.data["simulation"]["real_time_factor"]
        ),
    )
    dart = element(physics, "dart")
    element(dart, "collision_detector", DART_COLLISION_DETECTOR)
    element(element(dart, "solver"), "solver_type", DART_SOLVER)
    if lockstep_barrier(config.data) == "plugin":
        # First system: its PreUpdate holds each step until Python grants it.
        element(world, "plugin", filename=LOCKSTEP_FILENAME, name=LOCKSTEP_NAME)
    engine = element(plugin(world, "physics", "Physics"), "engine")
    element(engine, "filename", PHYSICS_ENGINE_PLUGIN)
    for library, system in [
        ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"),
        ("contact", "Contact"),
        ("imu", "Imu"),
        ("navsat", "NavSat"),
        ("air-pressure", "AirPressure"),
    ]:
        plugin(world, library, system)
    if config.render_sensors:
        element(plugin(world, "sensors", "Sensors"), "render_engine", "ogre2")
    wind = config.data["wind"]
    if wind_effects_enabled(wind):
        # Still-air drag alone uses zero wind velocity.
        velocity = wind["velocity_enu_mps"] if wind["enabled"] else [0.0, 0.0, 0.0]
        element(element(world, "wind"), "linear_velocity", " ".join(map(str, velocity)))
        effects = plugin(world, "wind-effects", "WindEffects")
        element(
            effects, "force_approximation_scaling_factor", wind["force_scale_per_s"]
        )
        horizontal = element(effects, "horizontal")
        magnitude = element(horizontal, "magnitude")
        element(magnitude, "time_for_rise", wind["rise_time_s"])
        gust = element(magnitude, "sin")
        element(gust, "amplitude_percent", wind["gust_amplitude_fraction"])
        element(gust, "period", wind["gust_period_s"])
        element(element(horizontal, "direction"), "time_for_rise", wind["rise_time_s"])
        element(element(effects, "vertical"), "time_for_rise", wind["rise_time_s"])
    if (
        config.data["aerodynamics"]["body_drag"]["enabled"]
        or config.data["fixed_wing"]["enabled"]
    ):
        # Python applies quadratic body drag as a persistent link wrench.
        plugin(world, "apply-link-wrench", "ApplyLinkWrench")
    # Gazebo computes the magnetic field from spherical_coordinates (an
    # internal declination/inclination/strength table), ignoring any world
    # <magnetic_field>. Its defaults publish gauss in a north-east-down frame;
    # request tesla in the world's ENU frame instead.
    magnetometer = plugin(world, "magnetometer", "Magnetometer")
    element(magnetometer, "use_earth_frame_ned", "false")
    element(magnetometer, "use_units_gauss", "false")
    coords = element(world, "spherical_coordinates")
    for key, value in dict(
        surface_model="EARTH_WGS84",
        world_frame_orientation="ENU",
        latitude_deg=40.4168,
        longitude_deg=-3.7038,
        elevation=WORLD_ELEVATION_M,
        heading_deg=0,
    ).items():
        element(coords, key, value)
    light = element(world, "light", name="sun", type="directional")
    element(light, "pose", "0 0 20 0 0 0")
    element(light, "diffuse", ".9 .9 .9 1")
    element(light, "direction", "-.3 -.4 -1")
    texture_dir = Path(output).parent / "textures"
    add_ground(world, scene, texture_dir)
    for obstacle in scene["obstacles"]:
        add_obstacle(world, obstacle, texture_dir)

    for entry in scene.get("moving_obstacles", []):
        add_moving_obstacle(world, entry)
    terrain = scene_terrain(scene)
    if terrain is not None:
        add_terrain(
            world,
            terrain,
            Path(output).parent / "terrain.obj",
            scene.get("ground_reflectivity"),
        )
    if config.wheeled is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _add_wheeled(world, scene, config, terrain, Path(output).parent)
    else:
        build_drone(world, render_sensors, control_mode, config, terrain=terrain)
    if config.data["payload"]["enabled"]:
        add_payload(world, config.data, terrain)
    ET.indent(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)
    return output
