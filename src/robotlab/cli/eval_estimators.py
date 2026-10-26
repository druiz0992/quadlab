"""Command line for the estimator benchmark (experiments/harness/estimator_benchmark.py)."""

import argparse
import importlib
import json
from pathlib import Path
import sys

from ..experiments.harness.estimator_benchmark import benchmark_markdown, run_benchmark
from ..testing.candidates import CANDIDATES, Candidate
from ..testing.scenarios import SCENARIOS


def load_candidates(spec: str) -> dict[str, Candidate]:
    """``package.module:NAME``, a dict of ``Candidate``; ValueError otherwise."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"--candidates {spec!r} must be 'package.module:NAME'")
    try:
        registry = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load candidates {spec!r}: {error}") from error
    if not isinstance(registry, dict) or not all(
        isinstance(c, Candidate) for c in registry.values()
    ):
        raise ValueError(
            f"{spec} must be a dict of robotlab.testing.candidates.Candidate"
        )
    return registry


def select(args: argparse.Namespace) -> dict[str, Candidate]:
    """Built-in plus loaded candidates, narrowed by ``--candidate``."""
    available = dict(CANDIDATES)
    for spec in args.candidates:
        for name, candidate in load_candidates(spec).items():
            if name in available:
                raise ValueError(f"Candidate {name!r} from {spec} is already defined")
            available[name] = candidate
    unknown = [n for n in args.candidate if n not in available]
    if unknown:
        raise ValueError(f"Unknown candidates {unknown}; known: {sorted(available)}")
    if not args.candidate:
        return available
    return {name: available[name] for name in args.candidate}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score estimators on the synthetic scenarios (no Gazebo)"
    )
    parser.add_argument("--output", help="Fresh output directory")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate to run (repeat; default: all)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario to run (repeat; default: all)",
    )
    parser.add_argument(
        "--candidates",
        action="append",
        default=[],
        metavar="MODULE:NAME",
        help="Load more candidates: a dict of Candidate, e.g. mypkg.bench:CANDIDATES",
    )
    parser.add_argument(
        "--list", action="store_true", help="List candidates and scenarios, then exit"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 on any limit violation or error",
    )
    args = parser.parse_args()
    try:
        candidates = select(args)
        if args.list:
            print("Candidates:", ", ".join(candidates))
            print("Scenarios:", ", ".join(sorted(SCENARIOS)))
            return
        if not args.output:
            raise ValueError("--output is required")
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"{output} is not empty; use a fresh output directory")
        results = run_benchmark(candidates, args.scenario or sorted(SCENARIOS))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    table = benchmark_markdown(results)
    (output / "benchmark.md").write_text(table)
    (output / "benchmark.json").write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n"
    )
    print(table)
    print(f"Benchmark: {output / 'benchmark.md'}")
    failed = [r for r in results if r.status in ("fail", "error")]
    sys.exit(1 if args.check and failed else 0)


if __name__ == "__main__":
    main()
