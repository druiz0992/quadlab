"""Generate a portable Gazebo Harmonic world from editable JSON boxes."""

import json
from ...sensors.lidar_returns import RETRO_PER_REFLECTIVITY
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .sdf import element, plugin, box
from ..moving_obstacles import validate_moving_obstacle
from ..terrain import Terrain, validate_terrain
from .plugins import LOCKSTEP_FILENAME, LOCKSTEP_NAME, lockstep_barrier
from .vehicle import add_payload, build_drone
from .config import WORLD_ELEVATION_M, load_drone_config, wind_effects_enabled

# Physics engine, written explicitly so a Gazebo upgrade cannot change it
# silently. These are gz-sim 8 / gz-physics 7's own defaults (verified: an
# unsupported detector logs "defaulting to [ode]"; lockstep outputs are
# byte-identical with and without them). See docs/2_simulation.md.
PHYSICS_ENGINE_PLUGIN = "gz-physics-dartsim-plugin"
DART_COLLISION_DETECTOR = "ode"
DART_SOLVER = "dantzig"


def load_scene(path):
    scene = json.loads(Path(path).read_text())
    names = set()
    for obstacle in scene["obstacles"]:
        name = obstacle["name"]
        if not name.isidentifier() or name in names or name in {"drone", "ground"}:
            raise ValueError(f"Invalid or duplicate obstacle name: {name}")
        names.add(name)
        for field in ["position", "size"]:
            values = np.asarray(obstacle[field], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name}: {field} must have three finite values")
        if min(obstacle["size"]) <= 0:
            raise ValueError(f"{name}: sizes must be positive")
        _check_reflectivity(obstacle.get("reflectivity"), name)
        moment = obstacle.get("magnetic_moment_am2")
        if moment is not None and (
            np.asarray(moment, dtype=float).shape != (3,)
            or not np.isfinite(np.asarray(moment, dtype=float)).all()
        ):
            raise ValueError(f"{name}: magnetic_moment_am2 needs three finite values")
    _check_reflectivity(scene.get("ground_reflectivity"), "ground_reflectivity")
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
        if (
            not name.isidentifier()
            or name in names
            or name in {"drone", "ground", "terrain"}
        ):
            raise ValueError(f"Invalid or duplicate obstacle name: {name}")
        names.add(name)
    return scene


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


def world_elevation_m(path):
    """Spherical-coordinates elevation of an SDF world in metres (0 if absent)."""
    text = ET.parse(path).findtext("./world/spherical_coordinates/elevation")
    return 0.0 if text is None else float(text)


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
    element(physics, "real_time_factor", config.data["simulation"]["real_time_factor"])
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
    ground = element(world, "model", name="ground")
    element(ground, "static", "true")
    box(
        element(ground, "link", name="link"),
        "ground_collision",
        [scene.get("ground_size_m", 100.0), scene.get("ground_size_m", 100.0), 0.1],
        ".45 .5 .45 1",
        "0 0 -.05 0 0 0",
        reflectivity=scene.get("ground_reflectivity"),
    )
    for obstacle in scene["obstacles"]:
        model = element(world, "model", name=obstacle["name"])
        element(model, "static", "true")
        element(model, "pose", " ".join(map(str, obstacle["position"])) + " 0 0 0")
        box(
            element(model, "link", name="link"),
            "box",
            obstacle["size"],
            ".65 .45 .3 1",
            reflectivity=obstacle.get("reflectivity"),
        )

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
    build_drone(world, render_sensors, control_mode, config, terrain=terrain)
    if config.data["payload"]["enabled"]:
        add_payload(world, config.data, terrain)
    ET.indent(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)
    return output
