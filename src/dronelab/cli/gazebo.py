"""Command-line options for Gazebo experiments and world editing."""

import argparse
import sys

from ..experiments.gazebo import GazeboConfig, run


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
        help="Drone JSON configuration; repeat to layer files (later ones override)",
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
        help="Fly with arrows and Q/A using per-rotor thrust control",
    )
    parser.add_argument(
        "--control-mode",
        choices=["velocity", "rotor-thrust"],
        default="velocity",
        help="rotor-thrust holds the initial pose unless keyboard input is enabled",
    )
    parser.add_argument(
        "--ekf",
        action="store_true",
        help="Run GPS/IMU EKF live alongside truth control; record and visualize estimates",
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
    try:
        run(GazeboConfig(**vars(args)))
    except ValueError as error:
        # Invalid options or configuration: show usage.
        parser.error(str(error))
    except RuntimeError as error:
        # A failed run, not a usage problem; results were reported above.
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
