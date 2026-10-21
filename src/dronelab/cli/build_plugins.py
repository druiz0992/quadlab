"""Command line: build DroneLab's Gazebo plugins (simulation/gazebo/plugins.py)."""

import argparse
from pathlib import Path

from ..simulation.gazebo.plugins import LOCKSTEP_LIBRARY, build_plugins, plugin_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DroneLab's Gazebo system plugins (needs cmake, g++, Gazebo dev files)"
    )
    parser.add_argument(
        "--build-dir", help=f"Output directory (default {plugin_dir()})"
    )
    args = parser.parse_args()
    try:
        directory = build_plugins(Path(args.build_dir) if args.build_dir else None)
    except RuntimeError as error:
        parser.error(str(error))
    print(f"Built {directory / LOCKSTEP_LIBRARY}")
    if args.build_dir:
        print(f"Set DRONELAB_PLUGIN_DIR={directory} so runs find it")


if __name__ == "__main__":
    main()
