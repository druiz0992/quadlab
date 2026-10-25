"""Run health verdict for biped missions: did the simulation go wrong?

Evaluation only: watches MuJoCo truth and the simulator's warning counters
and never influences the run. Findings and the verdict have the same shape
as the Gazebo verdict (evaluation/health.py), so ``summary.json.health``
reads the same for every robot:

| Kind | Severity | Condition |
| --- | --- | --- |
| ``non_finite`` | failed | NaN/inf in truth or a joint command |
| ``fallen`` | failed | trunk tilt above ``fall_tilt_deg``, or trunk height below ``fall_height_fraction`` × its initial height |
| ``simulator_warning`` | failed | MuJoCo raised a warning (e.g. unstable accelerations); count = occurrences |
| ``incomplete`` | failed | the run ended on an error before its duration |
| ``interrupted`` | warning | the user interrupted the run |

Closing the keyboard window (``keyboard_closed``) is a normal end.

``fallen`` is a failed verdict even when a fall is the experiment: a
standing or walking biped is the expected result. Times are seconds since
the first truth sample.
"""

from collections.abc import Mapping
from dataclasses import asdict
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..simulation.mujoco.state import BipedTruth, tilt_deg
from .health import FAILED, OK, USER_EXITS, WARNING, Finding


class BipedHealth:
    """Accumulate biped findings (see module docs).

    ``settings`` is the validated ``health`` section of the biped config.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.fall_tilt_deg = float(settings["fall_tilt_deg"])
        self.fall_height_fraction = float(settings["fall_height_fraction"])
        self._findings: dict[str, Finding] = {}
        self._start_ns: int | None = None
        self._initial_height_m: float | None = None

    def on_truth(self, truth: BipedTruth) -> None:
        """Check one truth sample for non-finite values and a fall."""
        if self._start_ns is None:
            self._start_ns = truth.sim_time_ns
            self._initial_height_m = float(truth.position_m[2])
        t = (truth.sim_time_ns - self._start_ns) * 1e-9
        values = truth.record()
        arrays = [np.asarray(v) for k, v in values.items() if k != "sim_time_ns"]
        if not all(np.all(np.isfinite(a)) for a in arrays):
            self._report("non_finite", t, "non-finite truth state")
            return
        tilt = tilt_deg(truth.quaternion_xyzw)
        if tilt > self.fall_tilt_deg:
            self._report("fallen", t, f"trunk tilted {tilt:.0f} deg", tilt)
        assert self._initial_height_m is not None
        floor_m = self.fall_height_fraction * self._initial_height_m
        if truth.position_m[2] < floor_m:
            self._report(
                "fallen",
                t,
                f"trunk height {truth.position_m[2]:.3f} m below {floor_m:.3f} m",
                tilt,
            )

    def on_command(self, sim_time_ns: int, target: ArrayLike) -> None:
        """Check one joint-target command for non-finite values."""
        if not np.all(np.isfinite(np.asarray(target, dtype=float))):
            start = self._start_ns if self._start_ns is not None else sim_time_ns
            self._report(
                "non_finite", (sim_time_ns - start) * 1e-9, "non-finite joint command"
            )

    def summary(
        self, *, completed: bool, exit_reason: str, warnings: Mapping[str, int]
    ) -> dict[str, Any]:
        """Add run-level findings and return ``summary.json.health``.

        ``warnings`` counts MuJoCo warnings by name (nonzero entries only).
        Call once, at the end of the run.
        """
        for name, count in warnings.items():
            self._report(
                "simulator_warning", None, f"MuJoCo warning {name}", count=count
            )
        if not completed and exit_reason not in USER_EXITS:
            if exit_reason == "interrupted":
                self._report(
                    "interrupted",
                    None,
                    "run interrupted (Ctrl-C or SIGTERM)",
                    severity=WARNING,
                )
            else:
                self._report("incomplete", None, f"run ended early: {exit_reason}")
        findings = sorted(
            self._findings.values(),
            key=lambda f: (math.inf if f.time_s is None else f.time_s, f.kind),
        )
        severities = {f.severity for f in findings}
        verdict = FAILED if FAILED in severities else WARNING if severities else OK
        return dict(
            verdict=verdict,
            findings=[asdict(f) for f in findings],
            thresholds=dict(
                fall_tilt_deg=self.fall_tilt_deg,
                fall_height_fraction=self.fall_height_fraction,
            ),
        )

    def _report(
        self,
        kind: str,
        time_s: float | None,
        message: str,
        value: float | None = None,
        *,
        count: int = 1,
        severity: str = FAILED,
    ) -> None:
        """Record occurrences: first time and message kept, worst value kept."""
        existing = self._findings.get(kind)
        if existing is None:
            self._findings[kind] = Finding(
                kind, severity, time_s, message, value, count
            )
            return
        existing.count += count
        if value is not None and (existing.value is None or value > existing.value):
            existing.value = value
