"""Entry point of every mission: run settings and the choice of robot.

``run(GazeboConfig(...))`` validates the shared options, then dispatches on
the robot kind (``--robot`` or ``robot.kind`` in the config layers):

- ``biped`` → MuJoCo (biped/biped_run.py);
- ``rover``, ``car``, ``tractor`` → Gazebo wheeled mission (wheeled/wheeled_run.py);
- ``multirotor`` (default) → Gazebo drone mission (multirotor/mission_run.py).

``GazeboConfig`` keeps its historical name: it holds the command-line
options of every robot, not only Gazebo ones.
"""

from dataclasses import dataclass

from ..simulation.gazebo.world import WORLD_ELEVATION_M, world_elevation_m


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
    # Robot family (``--robot``); None: from the config layers (default
    # multirotor). See simulation/wheeled/config.py; ``biped`` runs in
    # MuJoCo instead (experiments/biped/biped_run.py).
    robot: str | None = None


def run(args: GazeboConfig, command_source=None):
    """Run a mission; optional command_source(sim_time) drives rotor-mode experiments."""
    if args.keyboard and args.edit:
        raise ValueError("--keyboard requires mission mode; omit --edit")
    if args.keyboard and command_source is not None:
        raise ValueError("Choose keyboard or a programmatic command source")
    if command_source is not None and args.control_mode != "rotor-thrust":
        raise ValueError("A command source requires rotor-thrust mode")
    windows = []

    def open_keyboard():
        from ..visualization.keyboard import KeyboardWindow

        windows.append(KeyboardWindow())
        return windows[-1]

    try:
        # The mission opens the window only once the run is validated (never for
        # generate-only), so option errors need no desktop. Wheeled robots
        # open their own driving window (experiments/wheeled/wheeled_run.py).
        return _dispatch(args, open_keyboard if args.keyboard else None, command_source)
    finally:
        for window in windows:
            window.close()


def _dispatch(args, open_keyboard, command_source):
    """Pick the robot's simulator and mission (see module docs)."""
    from ..simulation.gazebo.config import load_drone_config, requested_robot_kind

    if requested_robot_kind(args.robot, args.drone) == "biped":
        from .biped.biped_run import run_biped

        return run_biped(args, command_source is not None)

    if args.drone and args.world:
        raise ValueError("--drone/--config requires a generated world; omit --world")
    if args.robot not in (None, "multirotor") and args.world:
        raise ValueError(
            f"{args.robot}: needs a generated world (the wheel model reads the "
            "scene); omit --world"
        )
    # A saved world keeps its own elevation (for the "site" air density).
    config = load_drone_config(
        args.drone,
        render_sensors=args.render_sensors,
        site_elevation_m=(
            world_elevation_m(args.world) if args.world else WORLD_ELEVATION_M
        ),
        robot=args.robot,
    )
    if config.wheeled is not None:
        from .wheeled.wheeled_run import run_wheeled

        return run_wheeled(args, config, command_source is not None)
    from .multirotor.mission_run import fly

    return fly(args, config, open_keyboard, command_source)
