"""Offline replay: run an estimator on a recorded run without Gazebo.

Reads ``replay.jsonl`` (see replay_log.py), builds the estimator from the
recorded settings (with optional overrides, validated like any drone
config layer) and the recorded ``AlgorithmContext``, feeds it the delivered
samples in their recorded order, and scores its estimates against the
recorded truth exactly as a live run does (``RunEvaluation``). The
estimator never sees truth.

Replay is open loop: what was flown does not change, so it suits
estimators. Flying a controller on a different estimator still needs the
simulator. Replaying the estimator the run used reproduces the live
``estimates.jsonl`` and estimation metrics exactly.

Outputs in a fresh directory: ``estimates.jsonl``, ``metrics.json``
(``estimation`` section), ``algorithm-debug.jsonl`` (when the estimator
logs), ``replay-settings.json`` (the settings used), and
``replay-summary.json``.
"""

from collections.abc import Iterator, Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...core.serialization import decode
from .algorithm_debug import AlgorithmDebugLog
from .algorithms import build_estimator
from .batch import nested
from ..common.mission import MissionRunner
from .replay_log import FORMAT, VERSION, estimate_record
from .run_evaluation import RunEvaluation


def read_log(path: Path) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    """Header and an iterator over the events; ValueError if not a replay log."""
    stream = path.open()
    header = json.loads(stream.readline() or "{}")
    if header.get("format") != FORMAT:
        stream.close()
        raise ValueError(f"{path} is not a {FORMAT} log")
    if header.get("version") != VERSION:
        stream.close()
        raise ValueError(f"{path}: unsupported replay version {header.get('version')}")

    def events() -> Iterator[dict[str, Any]]:
        with stream:
            for line in stream:
                yield json.loads(line)

    return header, events()


# Settings removed since recordings were made, dropped on replay (each is
# listed in replay-summary.json "migrated_settings"): the --ekf live
# observer's reordering keys (removed 2026-10-25).
REMOVED_SETTINGS = (
    ("ekf", "reorder_capacity"),
    ("ekf", "max_sensor_lag_sim_s"),
    ("ekf", "sensor_watchdog_wall_s"),
)


def migrate_settings(recorded: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A copy of ``recorded`` without ``REMOVED_SETTINGS``, and the dotted
    keys that were dropped."""
    settings = json.loads(json.dumps(recorded))
    dropped = []
    for section, key in REMOVED_SETTINGS:
        if key in settings.get(section, {}):
            del settings[section][key]
            dropped.append(f"{section}.{key}")
    return settings, dropped


def replay_settings(
    recorded: Mapping[str, Any], overrides: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    """Recorded settings plus dotted-key overrides, validated as config layers.

    Writes both layers to ``output`` (they document the replay) and returns
    the validated settings; raises ValueError on invalid overrides.
    """
    from ...simulation.gazebo.config import load_drone_config

    base = output / "replay-base-settings.json"
    base.write_text(json.dumps(recorded, indent=2) + "\n")
    layers = [base]
    if overrides:
        layer = output / "replay-overrides.json"
        layer.write_text(json.dumps(nested(overrides), indent=2) + "\n")
        layers.append(layer)
    settings = load_drone_config(layers).data
    (output / "replay-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    base.unlink()
    return settings


def replay(
    log_path: str | Path,
    output: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one recorded run (see module docs); return the summary.

    ``overrides`` are dotted config keys, e.g. ``{"algorithms.estimator":
    "mypkg.mod:Filter", "algorithms.estimator_options": {...}}``. Raises
    ValueError when there is no estimator to run or the output is not fresh.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a fresh replay output")
    output.mkdir(parents=True, exist_ok=True)
    header, events = read_log(Path(log_path))
    recorded, migrated = migrate_settings(header["settings"])
    settings = replay_settings(recorded, overrides or {}, output)
    debug = AlgorithmDebugLog(output / "algorithm-debug.jsonl")
    context = replace(decode(header["context"]), log=debug)
    estimator = build_estimator(settings, context)
    if estimator is None:
        raise ValueError("No estimator: set algorithms.estimator")
    try:
        counts = _run(settings, header, events, estimator, output)
    finally:
        debug.close()
    summary = dict(
        log=str(log_path),
        estimator=settings["algorithms"]["estimator"],
        estimator_summary=estimator.summary(),
        algorithm_debug_entries=debug.entries,
        migrated_settings=migrated,
        **counts,
    )
    (output / "replay-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _run(
    settings: dict[str, Any],
    header: dict[str, Any],
    events: Iterator[dict[str, Any]],
    estimator: Any,
    output: Path,
) -> dict[str, Any]:
    """Feed events in order; write estimates and metrics; return counts."""
    bias: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(3), np.zeros(3))]
    evaluation = RunEvaluation(
        settings,
        header["weight_n"],
        MissionRunner(settings["mission"]),
        lambda *args: None,
        true_biases=lambda: bias[0],
    )
    inputs = set(estimator.inputs)
    latest = None
    counts = dict(samples=0, delivered=0, estimates=0, truth_samples=0)
    with (output / "estimates.jsonl").open("w") as estimates:
        for event in events:
            if event["event"] == "truth":
                counts["truth_samples"] += 1
                bias[0] = (
                    np.asarray(event["accel_bias_mps2"]),
                    np.asarray(event["gyro_bias_rad_s"]),
                )
                evaluation.on_truth(
                    event["sim_time_ns"],
                    np.asarray(event["position_m"]),
                    np.asarray(event["velocity_mps"]),
                    np.asarray(event["quaternion_xyzw"]),
                    controlled=False,
                )
                continue
            counts["samples"] += 1
            if event["kind"] not in inputs:
                continue
            counts["delivered"] += 1
            estimator.on_sample(event["kind"], decode(event["sample"]))
            new = estimator.latest()
            if new is not None and new is not latest:
                latest = new
                counts["estimates"] += 1
                estimates.write(json.dumps(estimate_record(new)) + "\n")
                evaluation.on_estimate(new)
            evaluation.on_innovations(estimator.drain_innovations())
    metrics = dict(estimation=evaluation.summary()["estimation"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return counts


# Comparing estimators on one recording ------------------------------------------

COMPARISON_METRICS = (
    "errors.position_m.rmse",
    "errors.horizontal_m.rmse",
    "errors.vertical_m.rmse",
    "errors.velocity_mps.rmse",
    "errors.attitude_deg.rmse",
    "errors.yaw_deg.rmse",
    "nees.mean",
    "trajectory.ate.yaw.position_m.rmse",
    "trajectory.rpe.10m.drift_pct.mean",
    "trajectory.rpe.10s.rotation_deg.rmse",
)


def comparison_directions() -> dict[str, str]:
    """Errors are better lower (``increase`` = worse); NEES is unmarked
    (consistent means equal to its degrees of freedom, not minimal)."""
    return {
        m: "increase"
        for m in COMPARISON_METRICS
        if m.startswith(("errors.", "trajectory."))
    }


def estimator_label(spec: str, taken: set[str]) -> str:
    """Directory/column name for an estimator spec (unique within ``taken``)."""
    base = spec.rpartition(":")[2] or spec
    label, index = base, 2
    while label in taken:
        label, index = f"{base}-{index}", index + 1
    taken.add(label)
    return label


def compare_estimators(
    log_path: str | Path,
    output: str | Path,
    estimators: list[str],
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """Replay each estimator into ``output/<label>``; write ``comparison.md``.

    Lower errors are marked best; NEES is shown unmarked (consistent means
    equal to its degrees of freedom, not minimal). Returns the table.
    """
    from .batch import lookup
    from .comparison import comparison_markdown, stats

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a fresh replay output")
    columns: dict[str, dict[str, Any]] = {}
    taken: set[str] = set()
    for spec in estimators:
        label = estimator_label(spec, taken)
        settings = dict(overrides or {}, **{"algorithms.estimator": spec})
        replay(log_path, output / label, settings)
        metrics = json.loads((output / label / "metrics.json").read_text())
        estimation = metrics["estimation"] or {}
        columns[label] = {m: stats([lookup(estimation, m)]) for m in COMPARISON_METRICS}
    table = comparison_markdown(columns, COMPARISON_METRICS, comparison_directions())
    (output / "comparison.md").write_text(table)
    return table
