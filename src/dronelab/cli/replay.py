"""Command line for offline estimator replay (see experiments/replay.py)."""

import argparse
import json
from pathlib import Path
import sys

from ..experiments.replay import replay


def parse_overrides(items: list[str]) -> dict:
    """``key=value`` pairs; values are JSON when they parse, else strings."""
    overrides = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set needs key=value, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an estimator on a recorded run (replay.jsonl) without Gazebo"
    )
    parser.add_argument("run", help="Run directory or replay.jsonl file")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument(
        "--estimator", help="Estimator: gps_imu_ekf or package.module:Class"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key (JSON value), e.g. ekf.gps_std_m=2",
    )
    args = parser.parse_args()
    log = Path(args.run)
    if log.is_dir():
        log = log / "replay.jsonl"
    try:
        overrides = parse_overrides(args.set)
        if args.estimator:
            overrides["algorithms.estimator"] = args.estimator
        summary = replay(log, args.output, overrides)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    metrics = json.loads((Path(args.output) / "metrics.json").read_text())
    errors = (metrics["estimation"] or {}).get("errors", {})
    print(
        json.dumps(dict(summary=summary, position_m=errors.get("position_m")), indent=2)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
