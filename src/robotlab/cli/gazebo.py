"""Command-line options for Gazebo experiments and world editing."""

import argparse

from ..experiments.run import GazeboConfig, run
from .common import run_reporting_errors


def main():
    parser = argparse.ArgumentParser(
        description="Gazebo world, flight controller and Rerun recorder"
    )
    parser.add_argument("--scene", default="worlds/training.json")
    parser.add_argument(
        "--world", help="Use a world saved from Gazebo instead of generating one"
    )
    parser.add_argument(
        "--drone",
        "--config",
        dest="drone",
        action="append",
        help="JSON configuration; repeat to layer files (later ones override)",
    )
    parser.add_argument(
        "--robot",
        choices=["multirotor", "rover", "car", "tractor", "biped"],
        help="Robot type (default: multirotor, or robot.kind from --config); "
        "rover, car, and tractor load that vehicle's preset; biped runs in "
        "MuJoCo (biped.model_path; default: the built-in test biped)",
    )
    parser.add_argument("--output", default="outputs/gazebo")
    parser.add_argument(
        "--duration",
        type=float,
        help="Simulation seconds (default: 20; keyboard: 3600)",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Fly (multirotor: arrows and Q/A), drive (wheeled robots: arrows), "
        "or walk (bipeds: arrows, A/D)",
    )
    parser.add_argument(
        "--control-mode",
        choices=["velocity", "rotor-thrust"],
        default="velocity",
        help="rotor-thrust holds the initial pose unless keyboard input is enabled",
    )
    parser.add_argument("--target", nargs=3, type=float, default=[3.0, 0.0, 2.0])
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Gazebo server only; keyboard or --rerun still opens Rerun",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Open live Rerun even with --headless (automatic with --keyboard)",
    )
    parser.add_argument(
        "--render-sensors",
        action="store_true",
        help="Enable camera and GPU lidar (requires working rendering)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Write world without starting Gazebo",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Open a paused Gazebo world for editing, without a flight mission",
    )
    args = parser.parse_args()
    if args.duration is None:
        args.duration = 3600.0 if args.keyboard else 20.0
    # A failed run's results and health were reported above (see common.py).
    run_reporting_errors(parser, lambda: run(GazeboConfig(**vars(args))))


if __name__ == "__main__":
    main()
