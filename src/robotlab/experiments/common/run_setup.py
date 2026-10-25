"""Mission setup before Gazebo starts: option checks and world preparation.

Validates the combination of command-line options and configuration
(raising ValueError with the reason), draws the true vehicle for this run,
generates the world SDF, and writes ``drone-resolved.json``. Nothing here
touches Gazebo transport or processes, so it runs in ``--generate-only``.
"""

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...simulation.gazebo.config import DroneConfig
from ...simulation.gazebo.world import build_world, load_scene, scene_terrain
from ...simulation.randomization import randomize_config
from ...simulation.terrain import Terrain


@dataclass(frozen=True)
class RunMode:
    """Control mode of a run: rotor thrust (Python controller) and lockstep."""

    rotor_mode: bool
    lockstep: bool


@dataclass(frozen=True)
class PreparedWorld:
    """The true vehicle, scene, and world file of one run.

    ``true_config`` is what Gazebo simulates (possibly randomized);
    algorithms keep the nominal configuration. ``randomization_draws`` is
    empty without randomization. ``world`` is the generated or given SDF.
    """

    true_config: DroneConfig
    randomization_draws: dict[str, Any]
    output: Path
    scene: dict[str, Any]
    terrain: Terrain | None
    world: Path


def validate_run(
    args: Any, settings: dict[str, Any], has_command_source: bool
) -> RunMode:
    """Check options against the configuration; raise ValueError if they conflict.

    ``args`` is a ``GazeboConfig``; ``has_command_source`` is True when a
    programmatic pilot drives the run.
    """
    timing, algorithms = settings["control"], settings["algorithms"]
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
    _validate_algorithms(args, algorithms, rotor_mode)
    if settings["mission"]["type"] != "none":
        if not rotor_mode or args.keyboard or has_command_source:
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
    return RunMode(rotor_mode=rotor_mode, lockstep=lockstep)


def _validate_algorithms(
    args: Any, algorithms: dict[str, Any], rotor_mode: bool
) -> None:
    if algorithms["control_state_source"] == "estimate" and not rotor_mode:
        raise ValueError("Estimate-driven control needs rotor-thrust mode")
    if (algorithms["estimator"] or algorithms["controller"]) and args.world:
        raise ValueError("Algorithm plug-ins require a generated world; omit --world")


def prepare_world(
    args: Any,
    config: DroneConfig,
    rotor_mode: bool,
    randomize: Callable[[DroneConfig], tuple[DroneConfig, dict]] = randomize_config,
) -> PreparedWorld:
    """Draw the true vehicle, build (or reuse) the world, record the config.

    Creates the output directory. Generated worlds also get
    ``drone-resolved.json``; a given ``--world`` is used as is.
    """
    # Gazebo simulates the true (possibly randomized) vehicle; everything
    # algorithm-side keeps the nominal configuration.
    true_config, draws = randomize(config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.scene)
    terrain = scene_terrain(scene)
    if args.world:
        world = Path(args.world).resolve()
    else:
        world = build_world(
            scene,
            output / "world.sdf",
            args.render_sensors,
            "rotor-thrust" if rotor_mode else "velocity",
            true_config,
        )
        write_resolved_config(output, config, true_config, draws)
    return PreparedWorld(true_config, draws, output, scene, terrain, world)


def write_resolved_config(
    output: Path,
    config: DroneConfig,
    true_config: DroneConfig,
    draws: dict[str, Any],
) -> None:
    """``drone-resolved.json``: settings, derived values, true vehicle, sources."""
    resolved = dict(config.data)
    resolved["derived"] = dict(
        total_mass_kg=config.properties.mass,
        weight_n=config.properties.mass * 9.81,
        max_rotor_thrust_n=config.properties.max_rotor_thrust,
        config_sha256=config.content_hash,
    )
    if draws:
        true = true_config.data
        resolved["randomized_truth"] = dict(
            draws=draws,
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
    (output / "drone-resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
