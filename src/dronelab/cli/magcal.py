"""Command line for the automated magnetometer calibration (see experiments)."""

import argparse
import json

from ..experiments.magnetometer_calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the configured vehicle's magnetometer in Gazebo"
    )
    parser.add_argument(
        "--drone", "--config", dest="drone", action="append", help="Drone config layers"
    )
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--orientations", type=int, default=120)
    parser.add_argument(
        "--samples", type=int, default=3, help="Readings per orientation"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = run_calibration(
            args.drone,
            args.output,
            orientations=args.orientations,
            samples=args.samples,
            seed=args.seed,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report["evaluation"], indent=2))
    print(f"Calibration layer: {args.output}/magnetometer-calibration.json")


if __name__ == "__main__":
    main()
