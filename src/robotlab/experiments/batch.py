"""Batch experiments: seed and parameter sweeps, aggregation, and baselines.

An experiment file (JSON, see ``configs/experiments/``) describes one
mission and how to vary it:

```json
{
  "name": "square_ekf",
  "scene": "worlds/training.json",
  "drone": ["configs/missions/square.json", "configs/control/lockstep.json"],
  "control_mode": "rotor-thrust",
  "duration_s": 45,
  "seeds": [1, 2, 3],
  "parameters": {"wind.velocity_enu_mps": [[0, 0, 0], [3, 0, 0]]},
  "set": {"wind.enabled": true},
  "metrics": ["control.tracking.position_m.rmse"],
  "baseline": "baselines/square_ekf.json",
  "tolerances": {"default": {"relative": 0.05, "absolute": 1e-3}}
}
```

``robot`` (optional: ``rover``, ``car``, ``tractor``) runs a wheeled robot
instead of the multirotor; ``control_mode`` then does not apply.

Runs are the Cartesian product of ``parameters`` values times ``seeds``.
Each run gets an override layer (``set``, then that run's parameter values,
then seeds) written as JSON and appended after ``drone``, so overrides go
through normal config validation. A seed sets every seed in the resolved
configuration (``seed``/``random_seed`` keys) to seed + CRC32(key path)
(mod 2³¹): streams stay independent, and adding new seed keys to the
configuration never changes the values of existing ones. Each run is a separate ``robotlab`` process with a fresh
output directory, ``parallel`` at a time; each Gazebo uses its own private
transport partition.

Each run's health verdict (``summary.json.health``, see
``evaluation/health.py``) is recorded. Runs whose verdict is ``failed`` are
left out of the aggregates (their group reports how many were excluded):
a crashed run's metrics describe the crash, not the algorithm. Runs from
before the verdict existed have no verdict and are included.

Aggregation groups runs by parameter values and reports count, mean,
standard deviation, minimum, maximum, and the per-run values for each metric
(dotted path into ``metrics.json``, or ``summary.<path>`` into
``summary.json``). A baseline stores the group means; ``check`` flags a
metric whose mean moved beyond ``absolute + relative·|baseline|`` in the
configured ``direction`` ("both", "increase" = higher is worse, or
"decrease").
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import zlib
from typing import Any

import numpy as np

from ..simulation.gazebo.process import parent_death_prefix
from .interrupts import shielded

DEFAULT_METRICS = (
    "control.tracking.position_m.rmse",
    "control.tracking.position_m.max",
    "control.effort.mean_thrust_to_weight",
    "estimation.errors.position_m.rmse",
    "estimation.errors.attitude_deg.rmse",
    "estimation.nees.mean",
    "summary.completed",
)
SEED_MODULUS = 2**31


@dataclass(frozen=True)
class RunSpec:
    """One run of a batch: name, group label, and dotted-key overrides."""

    name: str
    group: str
    overrides: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


def nested(dotted: Mapping[str, Any]) -> dict[str, Any]:
    """{"a.b": 1} → {"a": {"b": 1}} (a config layer)."""
    result: dict[str, Any] = {}
    for key, value in dotted.items():
        node = result
        *parents, last = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return result


def seed_keys(tree: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths of every seed in a configuration tree."""
    keys = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys += seed_keys(value, path)
        elif key in ("seed", "random_seed"):
            keys.append(path)
    return keys


def key_seed(seed: int, path: str) -> int:
    """Seed for one configuration key: stable under added or removed keys."""
    return (seed + zlib.crc32(path.encode())) % SEED_MODULUS


def expand_runs(
    experiment: Mapping[str, Any], seed_paths: Sequence[str]
) -> list[RunSpec]:
    """Cartesian product of parameter values times seeds (see module docs)."""
    parameters = experiment.get("parameters", {})
    names = list(parameters)
    combos = list(itertools.product(*(parameters[n] for n in names))) or [()]
    seeds = experiment.get("seeds") or [None]
    runs = []
    for index, values in enumerate(combos):
        group = (
            ", ".join(f"{n}={json.dumps(v)}" for n, v in zip(names, values)) or "all"
        )
        for seed in seeds:
            overrides = dict(experiment.get("set", {}))
            overrides.update(zip(names, values))
            if seed is not None:
                for path in seed_paths:
                    overrides[path] = key_seed(int(seed), path)
            suffix = "" if seed is None else f"-s{seed}"
            runs.append(RunSpec(f"g{index:02d}{suffix}", group, overrides, seed))
    return runs


def lookup(tree: Any, dotted: str) -> Any:
    """Value at a dotted path, or None if absent."""
    for part in dotted.split("."):
        if not isinstance(tree, Mapping) or part not in tree:
            return None
        tree = tree[part]
    return tree


def run_metrics(run_dir: Path, metric_paths: Sequence[str]) -> dict[str, Any]:
    """Extract metrics (``metrics.json``; ``summary.`` prefix: summary.json)."""
    files = {}
    for name in ("metrics", "summary"):
        path = run_dir / f"{name}.json"
        files[name] = json.loads(path.read_text()) if path.exists() else {}
    values = {}
    for metric in metric_paths:
        if metric.startswith("summary."):
            values[metric] = lookup(files["summary"], metric[len("summary.") :])
        else:
            values[metric] = lookup(files["metrics"], metric)
    return values


def run_health(run_dir: Path) -> dict[str, Any] | None:
    """Verdict and findings from ``summary.json.health``, or None if absent."""
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    health = json.loads(path.read_text()).get("health")
    if not isinstance(health, Mapping):
        return None
    return dict(verdict=health.get("verdict"), findings=health.get("findings", []))


def health_failed(result: Mapping[str, Any]) -> bool:
    """Whether a run result's health verdict is ``failed``."""
    return (result.get("health") or {}).get("verdict") == "failed"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def aggregate(
    results: Sequence[Mapping[str, Any]], metric_paths: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Per group and metric: count, mean, std, min, max, values.

    Runs with a failed health verdict are excluded; each group's
    ``excluded_runs`` entry lists them (it is not a metric).
    """
    groups: dict[str, dict[str, list[float]]] = {}
    excluded: dict[str, list[str]] = {}
    for result in results:
        group = groups.setdefault(result["group"], {m: [] for m in metric_paths})
        excluded.setdefault(result["group"], [])
        if health_failed(result):
            excluded[result["group"]].append(result["name"])
            continue
        for metric in metric_paths:
            value = _number(result["metrics"].get(metric))
            if value is not None:
                group[metric].append(value)
    table: dict[str, dict[str, Any]] = {}
    for name, metrics in groups.items():
        table[name] = {}
        for metric, values in metrics.items():
            if not values:
                table[name][metric] = dict(count=0)
                continue
            array = np.asarray(values)
            table[name][metric] = dict(
                count=len(values),
                mean=float(array.mean()),
                std=float(array.std(ddof=1)) if len(values) > 1 else 0.0,
                min=float(array.min()),
                max=float(array.max()),
                values=values,
            )
        table[name]["excluded_runs"] = excluded[name]
    return table


def baseline_from(
    table: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Group means only (what a baseline stores)."""
    return {
        group: {
            m: s["mean"]
            for m, s in metrics.items()
            if isinstance(s, Mapping) and s.get("count")
        }
        for group, metrics in table.items()
    }


def check_baseline(
    table: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, float]],
    tolerances: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return regressions (empty when everything is within tolerance).

    Missing groups or metrics in the current run count as regressions.
    """
    default = dict(relative=0.05, absolute=1e-6, direction="both")
    default.update(tolerances.get("default", {}))
    failures: list[dict[str, Any]] = []
    for group, metrics in baseline.items():
        for metric, expected in metrics.items():
            stats = table.get(group, {}).get(metric, {})
            if not stats.get("count"):
                failures.append(dict(group=group, metric=metric, reason="missing"))
                continue
            rule = dict(default, **tolerances.get(metric, {}))
            limit = rule["absolute"] + rule["relative"] * abs(expected)
            change = stats["mean"] - expected
            worse = {
                "both": abs(change) > limit,
                "increase": change > limit,
                "decrease": -change > limit,
            }[rule["direction"]]
            if worse:
                failures.append(
                    dict(
                        group=group,
                        metric=metric,
                        baseline=expected,
                        current=stats["mean"],
                        limit=limit,
                        direction=rule["direction"],
                    )
                )
    return failures


def run_command(
    experiment: Mapping[str, Any], layer_path: Path, run_dir: Path
) -> list[str]:
    """Command line for one run (a separate robotlab process)."""
    command = [sys.executable, "-m", "robotlab", "--headless"]
    command += ["--scene", experiment.get("scene", "worlds/training.json")]
    robot = experiment.get("robot")
    if robot is None:
        command += ["--control-mode", experiment.get("control_mode", "rotor-thrust")]
    else:
        command += ["--robot", robot]
    for layer in experiment.get("drone", []):
        command += ["--drone", str(layer)]
    command += ["--drone", str(layer_path)]
    command += ["--duration", str(experiment.get("duration_s", 20.0))]
    if experiment.get("render_sensors"):
        command.append("--render-sensors")
    command += ["--output", str(run_dir)]
    return command


def execute(
    experiment: Mapping[str, Any],
    runs: Sequence[RunSpec],
    output: Path,
    *,
    parallel: int = 1,
    runner: Callable[[list[str], Path], int] | None = None,
) -> list[dict[str, Any]]:
    """Run every spec (``parallel`` at a time) and collect metrics.

    On Ctrl-C or SIGTERM (``KeyboardInterrupt``) running children are
    stopped gracefully and the interruption is re-raised; finished runs
    keep their outputs.
    """
    metrics = list(experiment.get("metrics") or DEFAULT_METRICS)
    output.mkdir(parents=True, exist_ok=True)
    children = ChildRuns()
    runner = runner or children.run

    def one(spec: RunSpec) -> dict[str, Any]:
        run_dir = output / spec.name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"{run_dir} is not empty; use a fresh batch output")
        run_dir.mkdir(parents=True, exist_ok=True)
        layer = run_dir / "batch-overrides.json"
        layer.write_text(json.dumps(nested(spec.overrides), indent=2) + "\n")
        code = runner(run_command(experiment, layer, run_dir), run_dir)
        return dict(
            name=spec.name,
            group=spec.group,
            seed=spec.seed,
            exit_code=code,
            health=run_health(run_dir),
            metrics=run_metrics(run_dir, metrics),
        )

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        try:
            return list(pool.map(one, runs))
        except KeyboardInterrupt:
            # Leaving the pool waits for running workers, whose children
            # would fly on to their full duration: stop them first.
            pool.shutdown(wait=False, cancel_futures=True)
            with shielded():
                children.stop_all()
            raise


class ChildRuns:
    """The batch's running ``robotlab`` processes (thread-safe).

    Children start under ``parent_death_prefix`` so they get SIGTERM if the
    batch dies; each turns SIGTERM into a graceful stop that saves its
    results (see interrupts.py).
    """

    def __init__(self, stop_timeout_wall_s: float = 60.0) -> None:
        self._lock = threading.Lock()
        self._running: set[subprocess.Popen] = set()
        self._stopping = False
        self._stop_timeout_wall_s = stop_timeout_wall_s

    def run(self, command: list[str], run_dir: Path) -> int:
        """Run one child to completion; output to ``run_dir/batch.log``.
        Raises RuntimeError once the batch is stopping."""
        with (run_dir / "batch.log").open("w") as log:
            with self._lock:
                if self._stopping:
                    raise RuntimeError("batch interrupted before this run started")
                process = subprocess.Popen(
                    [*parent_death_prefix(), *command],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                self._running.add(process)
            try:
                return process.wait()
            finally:
                with self._lock:
                    self._running.discard(process)

    def stop_all(self) -> None:
        """Start no more runs; SIGTERM running ones (they save results), and
        kill any still running after ``stop_timeout_wall_s``."""
        with self._lock:
            self._stopping = True
            running = list(self._running)
        for process in running:
            process.terminate()
        deadline = time.monotonic() + self._stop_timeout_wall_s
        for process in running:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Read an experiment file; paths in it are relative to the working directory."""
    experiment = json.loads(Path(path).read_text())
    allowed = {
        "name",
        "scene",
        "robot",
        "drone",
        "control_mode",
        "duration_s",
        "render_sensors",
        "seeds",
        "parameters",
        "set",
        "metrics",
        "baseline",
        "tolerances",
        "parallel",
    }
    unknown = set(experiment) - allowed
    if unknown:
        raise ValueError(f"Unknown experiment keys: {sorted(unknown)}")
    if not isinstance(experiment.get("drone", []), list):
        raise ValueError("experiment drone must be a list of config layers")
    robot = experiment.get("robot")
    if robot is not None:
        from ..simulation.wheeled.config import WHEELED_KINDS

        if robot not in WHEELED_KINDS:
            raise ValueError(f"experiment robot must be one of {WHEELED_KINDS}")
        if "control_mode" in experiment:
            raise ValueError("experiment control_mode applies to multirotors only")
    return experiment


def resolved_seed_paths(experiment: Mapping[str, Any]) -> list[str]:
    """Seed keys of the experiment's resolved configuration."""
    from ..simulation.gazebo.config import load_drone_config

    layers = list(experiment.get("drone", []))
    data = deepcopy(load_drone_config(layers, robot=experiment.get("robot")).data)
    return seed_keys(data)


def batch_comparison(
    table: Mapping[str, Mapping[str, Any]],
    metric_paths: Sequence[str],
    tolerances: Mapping[str, Mapping[str, Any]],
) -> str:
    """``comparison.md``: groups side by side (see experiments/comparison.py).

    Best values are marked using each metric's tolerance ``direction``.
    """
    from .comparison import comparison_markdown

    default = tolerances.get("default", {}).get("direction")
    directions = {
        m: tolerances.get(m, {}).get("direction", default) for m in metric_paths
    }
    columns = {
        group: {m: s for m, s in metrics.items() if m != "excluded_runs"}
        for group, metrics in table.items()
    }
    notes = {
        group: f"excluded (health failed): {', '.join(metrics['excluded_runs'])}"
        for group, metrics in table.items()
        if metrics.get("excluded_runs")
    }
    return comparison_markdown(columns, metric_paths, directions, notes)
