"""Score estimator candidates on the synthetic scenarios (no Gazebo).

``run_benchmark`` runs every selected candidate (testing/candidates.py) on
every selected scenario (testing/scenarios.py) its sensors can run on and
checks the candidate's declared limits; ``benchmark_markdown`` turns the
results into one comparison table per scenario (candidates as columns,
lowest error in bold; NIS unmarked, since consistent means about 1 per
degree of freedom, not minimal). ``robotlab-eval-estimators``
(cli/eval_estimators.py) writes both to an output directory. Recorded
Gazebo runs are compared with ``robotlab-replay --estimator`` instead.

A candidate that raises ``ValueError`` or breaks the plug-in contract
(``PluginContractError``) on a scenario is recorded as an error for that
scenario, not skipped silently; the rest of the benchmark still runs.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from ...testing.candidates import Candidate
from ...testing.plugins import PluginContractError
from ...testing.scenarios import SCENARIOS, ScenarioReport, run_scenario

# ScenarioReport metrics shown, in table order; all but NIS are errors.
REPORT_METRICS = (
    "position_rmse_m",
    "max_position_error_m",
    "final_position_error_m",
    "velocity_rmse_mps",
    "attitude_rmse_deg",
    "tilt_rmse_deg",
    "final_accel_bias_error_mps2",
    "final_gyro_bias_error_rad_s",
    "aligned_position_rmse_m",
    "relative_translation_rmse_m",
    "relative_rotation_rmse_deg",
    "mean_nis_per_dof",
)
# Lower is better; NIS is unmarked (about 1 per dof is consistent).
ERROR_METRICS = frozenset(REPORT_METRICS) - {"mean_nis_per_dof"}


@dataclass(frozen=True)
class BenchmarkResult:
    """One (candidate, scenario) cell.

    ``status`` is ``pass``, ``fail`` (limit violations), ``unchecked`` (no
    limits declared), ``skipped`` (sensors the vehicle lacks), or ``error``
    (``error`` holds the message; ``report`` is then None).
    """

    candidate: str
    scenario: str
    status: str
    report: ScenarioReport | None = None
    violations: tuple[str, ...] = ()
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (``benchmark.json``)."""
        result = asdict(self)
        result["violations"] = list(self.violations)
        return result


def run_benchmark(
    candidates: Mapping[str, Candidate], scenarios: Iterable[str]
) -> list[BenchmarkResult]:
    """Every candidate on every scenario (module docs); ValueError for an
    unknown scenario name, before anything runs."""
    names = list(scenarios)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios {unknown}; known: {sorted(SCENARIOS)}")
    return [
        _score(label, candidate, scenario)
        for scenario in names
        for label, candidate in candidates.items()
    ]


def _score(label: str, candidate: Candidate, scenario_name: str) -> BenchmarkResult:
    scenario = SCENARIOS[scenario_name]
    if not candidate.runs_on(scenario):
        return BenchmarkResult(label, scenario_name, "skipped")
    try:
        report = run_scenario(
            candidate.factory, candidate.options(scenario), scenario, candidate.sensors
        )
    except (ValueError, PluginContractError) as error:
        return BenchmarkResult(label, scenario_name, "error", error=str(error))
    violations = candidate.violations(report)
    if violations is None:
        return BenchmarkResult(label, scenario_name, "unchecked", report)
    status = "fail" if violations else "pass"
    return BenchmarkResult(label, scenario_name, status, report, tuple(violations))


def benchmark_markdown(results: list[BenchmarkResult]) -> str:
    """Overview of statuses, then one table per scenario (module docs)."""
    candidates = list(dict.fromkeys(r.candidate for r in results))
    scenarios = list(dict.fromkeys(r.scenario for r in results))
    cell = {(r.candidate, r.scenario): r for r in results}
    lines = [
        "| Scenario | " + " | ".join(candidates) + " |",
        "| --- |" + " --- |" * len(candidates),
    ]
    for scenario in scenarios:
        statuses = [cell[(c, scenario)].status for c in candidates]
        lines.append(f"| {scenario} | " + " | ".join(statuses) + " |")
    sections = ["# Estimator benchmark\n", "\n".join(lines) + "\n"]
    for scenario in scenarios:
        sections.append(
            _scenario_table(scenario, [cell[(c, scenario)] for c in candidates])
        )
    return "\n".join(sections)


def _scenario_table(scenario: str, row: list[BenchmarkResult]) -> str:
    """Metrics × candidates that ran; lowest error in bold (unique only)."""
    scored = [r for r in row if r.report is not None]
    notes = "".join(
        f"\n{r.candidate}: {'; '.join(r.violations) or r.error}\n"
        for r in row
        if r.violations or r.error
    )
    title = f"## {scenario}\n\n"
    if not scored:
        return title + "No candidate ran.\n" + notes
    lines = [
        "| Metric | " + " | ".join(r.candidate for r in scored) + " |",
        "| --- |" + " --- |" * len(scored),
    ]
    for metric in REPORT_METRICS:
        values = [getattr(r.report, metric) for r in scored]
        present = [v for v in values if v is not None]
        best = min(present) if metric in ERROR_METRICS and len(present) > 1 else None
        unique = best is not None and present.count(best) == 1
        cells = [
            (
                "—"
                if v is None
                else (f"**{v:.4g}**" if unique and v == best else f"{v:.4g}")
            )
            for v in values
        ]
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    return title + "\n".join(lines) + "\n" + notes
