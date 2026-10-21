"""Batch expansion, aggregation, and baseline checks (no Gazebo)."""

import json

import pytest

from dronelab.experiments.batch import (
    aggregate,
    baseline_from,
    check_baseline,
    execute,
    expand_runs,
    health_failed,
    key_seed,
    nested,
    run_command,
    seed_keys,
)

EXPERIMENT = dict(
    name="t",
    drone=["configs/missions/step_x.json"],
    duration_s=5,
    seeds=[1, 2],
    parameters={"wind.velocity_enu_mps": [[0, 0, 0], [2, 0, 0]]},
    set={"wind.enabled": True},
    metrics=["control.tracking.position_m.rmse", "summary.completed"],
)


def test_nested_and_seed_keys():
    assert nested({"a.b": 1, "a.c": 2, "d": 3}) == {"a": {"b": 1, "c": 2}, "d": 3}
    assert seed_keys({"x": {"seed": 1, "y": {"random_seed": 2}}, "seed_like": 3}) == [
        "x.seed",
        "x.y.random_seed",
    ]


def test_expand_runs_products_and_distinct_seeds():
    runs = expand_runs(EXPERIMENT, ["a.seed", "b.random_seed"])
    assert [r.name for r in runs] == ["g00-s1", "g00-s2", "g01-s1", "g01-s2"]
    assert runs[2].overrides == {
        "wind.enabled": True,
        "wind.velocity_enu_mps": [2, 0, 0],
        "a.seed": key_seed(1, "a.seed"),
        "b.random_seed": key_seed(1, "b.random_seed"),
    }
    assert runs[2].overrides["a.seed"] != runs[2].overrides["b.random_seed"]
    # Adding a seed key never changes the others.
    more = expand_runs(EXPERIMENT, ["new.seed", "a.seed", "b.random_seed"])
    assert more[2].overrides["a.seed"] == runs[2].overrides["a.seed"]
    assert runs[0].group != runs[2].group
    assert [r.name for r in expand_runs({}, [])] == ["g00"]


def test_run_command_appends_override_layer(tmp_path):
    command = run_command(EXPERIMENT, tmp_path / "o.json", tmp_path)
    assert command[1:3] == ["-m", "dronelab"]
    drones = [command[i + 1] for i, c in enumerate(command) if c == "--drone"]
    assert drones == ["configs/missions/step_x.json", str(tmp_path / "o.json")]


def fake_runner(command, run_dir):
    """Write metrics derived from the override layer, like a real run would."""
    overrides = json.loads((run_dir / "batch-overrides.json").read_text())
    wind = overrides["wind"]["velocity_enu_mps"][0]
    seed = run_dir.name.endswith("-s2")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "control": {
                    "tracking": {
                        "position_m": {"rmse": 0.1 + wind * 0.05 + seed * 0.001}
                    }
                }
            }
        )
    )
    (run_dir / "summary.json").write_text(json.dumps({"completed": True}))
    return 0


def test_execute_aggregate_and_baseline(tmp_path):
    runs = expand_runs(EXPERIMENT, ["a.seed"])
    results = execute(
        EXPERIMENT, runs, tmp_path / "out", parallel=2, runner=fake_runner
    )
    assert [r["exit_code"] for r in results] == [0] * 4
    table = aggregate(results, EXPERIMENT["metrics"])
    calm = table[runs[0].group]["control.tracking.position_m.rmse"]
    assert calm["mean"] == pytest.approx(0.1005) and calm["count"] == 2
    assert table[runs[0].group]["summary.completed"]["mean"] == 1.0
    baseline = baseline_from(table)
    assert check_baseline(table, baseline, {}) == []
    worse = json.loads(json.dumps(table))
    worse[runs[0].group]["control.tracking.position_m.rmse"]["mean"] = 0.2
    failures = check_baseline(worse, baseline, {})
    assert failures and failures[0]["metric"] == "control.tracking.position_m.rmse"
    # Improvements pass when only increases count as regressions.
    better = json.loads(json.dumps(table))
    better[runs[0].group]["control.tracking.position_m.rmse"]["mean"] = 0.01
    rules = {"control.tracking.position_m.rmse": {"direction": "increase"}}
    assert check_baseline(better, baseline, rules) == []
    assert check_baseline({}, baseline, {})[0]["reason"] == "missing"


def test_execute_refuses_non_empty_run_directories(tmp_path):
    runs = expand_runs({"seeds": [1]}, [])
    (tmp_path / runs[0].name).mkdir()
    (tmp_path / runs[0].name / "old.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        execute({}, runs, tmp_path, runner=fake_runner)


def test_failed_health_runs_are_excluded_from_aggregates(tmp_path):
    def runner(command, run_dir):
        fake_runner(command, run_dir)
        verdict = "failed" if run_dir.name == "g00-s2" else "ok"
        (run_dir / "summary.json").write_text(
            json.dumps({"completed": True, "health": {"verdict": verdict}})
        )
        return 0

    runs = expand_runs(EXPERIMENT, [])
    results = execute(EXPERIMENT, runs, tmp_path, runner=runner)
    assert [health_failed(r) for r in results] == [False, True, False, False]
    assert results[1]["health"] == {"verdict": "failed", "findings": []}
    table = aggregate(results, EXPERIMENT["metrics"])
    calm = table[runs[0].group]
    assert calm["excluded_runs"] == ["g00-s2"]
    assert calm["control.tracking.position_m.rmse"]["count"] == 1
    assert calm["control.tracking.position_m.rmse"]["mean"] == pytest.approx(0.1)
    assert table[runs[2].group]["excluded_runs"] == []
    baseline = baseline_from(table)
    assert "excluded_runs" not in baseline[runs[0].group]
    assert check_baseline(table, baseline, {}) == []
    # Runs from before the verdict existed are included.
    old = [dict(r, health=None) for r in results]
    assert (
        aggregate(old, EXPERIMENT["metrics"])[runs[0].group][
            "control.tracking.position_m.rmse"
        ]["count"]
        == 2
    )
