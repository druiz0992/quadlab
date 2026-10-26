"""Estimator benchmark: statuses, tables, candidate loading, the CLI."""

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from robotlab.cli import eval_estimators
from robotlab.experiments.harness.estimator_benchmark import (
    benchmark_markdown,
    run_benchmark,
)
from robotlab.testing.candidates import CANDIDATES, Candidate
from robotlab.testing.scenarios import EstimatorScenario

COMPLEMENTARY = CANDIDATES["complementary"]


def broken_options(scenario: EstimatorScenario) -> dict[str, Any]:
    return dict(unknown_option=1.0)


# Loaded by test_cli_loads_extra_candidates as "<this module>:EXTRA".
EXTRA = {
    "strict": Candidate(
        COMPLEMENTARY.factory,
        COMPLEMENTARY.options,
        {"hover": dict(position_rmse_m=1e-6)},
    ),
}


def test_statuses_cover_pass_fail_unchecked_skipped_and_error() -> None:
    candidates = {
        "complementary": COMPLEMENTARY,
        "strict": EXTRA["strict"],
        "unlimited": Candidate(COMPLEMENTARY.factory, COMPLEMENTARY.options),
        "wheels": CANDIDATES["wheel_odometry"],
        "broken": Candidate(COMPLEMENTARY.factory, broken_options),
    }
    results = {r.candidate: r for r in run_benchmark(candidates, ["hover"])}
    assert {name: r.status for name, r in results.items()} == dict(
        complementary="pass",
        strict="fail",
        unlimited="unchecked",
        wheels="skipped",
        broken="error",
    )
    assert results["strict"].violations[0].startswith("hover: position_rmse_m")
    assert results["broken"].error and results["broken"].report is None
    table = benchmark_markdown(list(results.values()))
    assert "| hover | pass | fail | unchecked | skipped | error |" in table
    assert "## hover" in table and "`position_rmse_m`" in table
    assert "strict: hover: position_rmse_m" in table
    json.dumps([r.as_dict() for r in results.values()])  # JSON-ready


def test_unknown_scenarios_raise_before_running() -> None:
    with pytest.raises(ValueError, match="Unknown scenarios"):
        run_benchmark(CANDIDATES, ["hover", "nope"])


@pytest.mark.parametrize(
    "spec, message",
    [
        ("nonsense", "package.module:NAME"),
        ("no_such_module_xyz:C", "Cannot load"),
        ("robotlab.testing.candidates:NIS_BAND", "dict of"),
    ],
)
def test_candidate_loading_errors_are_explicit(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        eval_estimators.load_candidates(spec)


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["robotlab-eval-estimators", *argv])
    with pytest.raises(SystemExit) as exit_info:
        eval_estimators.main()
    code = exit_info.value.code
    return code if isinstance(code, int) else 1


def test_cli_loads_extra_candidates_and_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    extra = "test_estimator_benchmark:EXTRA"
    common = ["--candidates", extra, "--scenario", "hover", "--candidate"]
    assert (
        run_cli(
            monkeypatch,
            *common,
            "complementary",
            "--output",
            str(tmp_path / "a"),
            "--check",
        )
        == 0
    )
    assert (
        run_cli(
            monkeypatch, *common, "strict", "--output", str(tmp_path / "b"), "--check"
        )
        == 1
    )
    results = json.loads((tmp_path / "b" / "benchmark.json").read_text())
    assert [r["status"] for r in results] == ["fail"]
    assert (tmp_path / "b" / "benchmark.md").read_text() in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--candidate", "nope", "--output", "x"], "Unknown candidates"),
        (["--scenario", "hover"], "--output is required"),
        (
            ["--candidates", "robotlab.testing.candidates:CANDIDATES", "--list"],
            "already defined",
        ),
    ],
)
def test_cli_rejects_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    assert run_cli(monkeypatch, *argv) == 2
    assert message in capsys.readouterr().err


def test_cli_refuses_a_used_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "old.txt").write_text("x")
    assert run_cli(monkeypatch, "--scenario", "hover", "--output", str(tmp_path)) == 2
    assert "not empty" in capsys.readouterr().err
