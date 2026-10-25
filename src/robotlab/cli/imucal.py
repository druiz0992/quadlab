"""Command line for the automated IMU calibration (see experiments)."""

import argparse
import json

from ..experiments.calibration.imu_calibration import run_imu_calibration
from .common import run_reporting_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the configured vehicle's IMU in Gazebo (static poses)"
    )
    parser.add_argument(
        "--drone", "--config", dest="drone", action="append", help="Drone config layers"
    )
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--extra-poses", type=int, default=12)
    parser.add_argument("--samples", type=int, default=40, help="IMU samples per pose")
    parser.add_argument(
        "--startup-gyro-s",
        type=float,
        default=0.0,
        help="Store a startup gyro calibration of this length (needs ground starts)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_reporting_errors(
        parser,
        lambda: run_imu_calibration(
            args.drone,
            args.output,
            extra_poses=args.extra_poses,
            samples=args.samples,
            startup_gyro_s=args.startup_gyro_s,
            seed=args.seed,
        ),
    )
    print(json.dumps(report["evaluation"], indent=2))
    print(f"Calibration layer: {args.output}/imu-calibration.json")


if __name__ == "__main__":
    main()
