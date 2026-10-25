"""Command line for batch experiments (see experiments/harness/batch.py)."""

import argparse
import json
from pathlib import Path
import sys

from ..experiments.harness.batch import (
    aggregate,
    baseline_from,
    batch_comparison,
    check_baseline,
    execute,
    expand_runs,
    health_failed,
    load_experiment,
    resolved_seed_paths,
    DEFAULT_METRICS,
)
from .common import run_reporting_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a seed/parameter sweep of robotlab missions and aggregate metrics"
    )
    parser.add_argument("experiment", help="Experiment JSON (configs/experiments/)")
    parser.add_argument("--output", required=True, help="Fresh batch output directory")
    parser.add_argument(
        "--parallel", type=int, help="Concurrent runs (default: file or 1)"
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs only")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write group means to the baseline file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare with the baseline; exit 1 on regression or a failed run",
    )
    args = parser.parse_args()
    try:
        experiment = load_experiment(args.experiment)
        runs = expand_runs(experiment, resolved_seed_paths(experiment))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        for run in runs:
            print(run.name, run.group, json.dumps(run.overrides))
        return
    output = Path(args.output)
    # Ctrl-C/SIGTERM stop the running children, which save their results.
    results = run_reporting_errors(
        parser,
        lambda: execute(
            experiment,
            runs,
            output,
            parallel=args.parallel or experiment.get("parallel", 1),
        ),
    )
    metrics = list(experiment.get("metrics") or DEFAULT_METRICS)
    table = aggregate(results, metrics)
    report = dict(experiment=experiment, runs=results, aggregate=table)
    failed_runs = [r["name"] for r in results if r["exit_code"] != 0]
    status = 0
    if failed_runs:
        print(f"Runs failed: {failed_runs} (see batch.log in each run directory)")
        status = 1
    unhealthy = [r["name"] for r in results if health_failed(r)]
    if unhealthy:
        print(f"Health verdict failed (excluded from aggregates): {unhealthy}")
    baseline_path = experiment.get("baseline")
    if args.save_baseline:
        if not baseline_path:
            parser.error("--save-baseline needs a baseline path in the experiment")
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        Path(baseline_path).write_text(
            json.dumps(baseline_from(table), indent=2) + "\n"
        )
        print(f"Baseline written: {baseline_path}")
    if args.check:
        if not baseline_path or not Path(baseline_path).exists():
            parser.error("--check needs an existing baseline file")
        baseline = json.loads(Path(baseline_path).read_text())
        failures = check_baseline(table, baseline, experiment.get("tolerances", {}))
        report["regressions"] = failures
        for failure in failures:
            print("REGRESSION", json.dumps(failure))
        report["health_failed_runs"] = unhealthy
        failed_check = bool(failures or unhealthy)
        print("Baseline check:", "FAILED" if failed_check else "passed")
        status = 1 if failed_check else status
    (output / "batch-results.json").write_text(json.dumps(report, indent=2) + "\n")
    comparison = batch_comparison(table, metrics, experiment.get("tolerances", {}))
    (output / "comparison.md").write_text(comparison)
    print(comparison)
    for group, metrics_table in table.items():
        print(f"[{group}]")
        for metric, stats in metrics_table.items():
            if metric == "excluded_runs":
                if stats:
                    print(f"  excluded (health failed): {stats}")
            elif stats.get("count"):
                print(
                    f"  {metric}: mean {stats['mean']:.6g} ± {stats['std']:.3g} (n={stats['count']})"
                )
    print(f"Results: {output / 'batch-results.json'}, {output / 'comparison.md'}")
    sys.exit(status)


if __name__ == "__main__":
    main()
