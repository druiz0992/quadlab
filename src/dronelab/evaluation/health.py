"""Run health verdict: did the simulated flight go physically wrong?

Evaluation only. The detector watches simulator truth (odometry and
contacts), the commands the controller sent, and the estimates it produced,
and reports findings; it never influences the run. Each finding has a kind,
a severity, the first time it occurred (seconds since the first truth
sample, or None for run-level findings), how often it occurred, and the
worst value seen:

| Kind | Severity | Condition |
| --- | --- | --- |
| ``non_finite`` | failed | NaN/inf in truth, a thrust command, or an estimate |
| ``flipped`` | failed | tilt above ``flip_tilt_deg`` for ``flip_duration_s`` |
| ``hard_impact`` | failed | contact while speed exceeded ``hard_impact_speed_mps`` |
| ``obstacle_contact`` | failed | contact with anything but the ground or terrain |
| ``below_ground`` | failed | body centre more than ``below_ground_margin_m`` below the ground |
| ``incomplete`` | failed | the run ended on an error before its duration |
| ``interrupted`` | warning | the user interrupted the run |
| ``ground_contact`` | warning | touchdown after ``touchdown_gap_s`` without contact |
| ``left_ground_area`` | warning | flew past the ground plane's edge (nothing below collides) |
| ``saturated`` | warning | more than ``saturated_fraction`` of commands saturated |
| ``dropped_messages`` | warning | more than ``max_dropped_messages`` messages dropped |

The verdict is ``failed`` if any finding failed, else ``warning`` if any
finding warned, else ``ok``.

Contacts come from the drone's body contact sensor; the other collision's
model (first ``::`` scope segment) decides whether it is ground. Hard
impacts use the highest true speed in the ``impact_window_s`` before the
contact time, because Gazebo can deliver the contact after post-impact
odometry.
"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

FAILED, WARNING, OK = "failed", "warning", "ok"
GROUND_MODELS = frozenset({"ground", "terrain"})
# The drone's own collisions and the attached payload are not obstacles.
IGNORED_MODELS = frozenset({"drone", "payload"})
WORLD_SCOPE = "dronelab"
# Normal ends of a run that stop before the configured duration.
USER_EXITS = frozenset({"keyboard_closed"})


@dataclass(frozen=True)
class HealthThresholds:
    """Detector limits (the drone config's ``health`` section).

    Units: degrees, seconds, metres, m/s; ``saturated_fraction`` is in
    (0, 1]; ``max_dropped_messages`` is a count. Invalid values raise
    ValueError.
    """

    flip_tilt_deg: float = 90.0
    flip_duration_s: float = 0.5
    hard_impact_speed_mps: float = 2.0
    impact_window_s: float = 0.1
    below_ground_margin_m: float = 0.1
    touchdown_gap_s: float = 0.5
    saturated_fraction: float = 0.5
    max_dropped_messages: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.flip_tilt_deg <= 180:
            raise ValueError("health.flip_tilt_deg must be in (0, 180]")
        if not 0 < self.saturated_fraction <= 1:
            raise ValueError("health.saturated_fraction must be in (0, 1]")
        if self.max_dropped_messages != int(self.max_dropped_messages):
            raise ValueError("health.max_dropped_messages must be an integer")
        for item in fields(self):
            if getattr(self, item.name) < 0:
                raise ValueError(f"health.{item.name} must be nonnegative")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "HealthThresholds":
        """Build from a ``health`` config section; raises ValueError if invalid."""
        unknown = set(settings) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown health keys: {sorted(unknown)}")
        return cls(**settings)


@dataclass
class Finding:
    """One kind of problem (per subject), aggregated over its occurrences."""

    kind: str
    severity: str
    time_s: float | None
    message: str
    value: float | None = None
    count: int = 1
    subject: str | None = None


def contact_model(collision_name: str) -> str:
    """Model name of a scoped collision ("drone::base_link::body" → "drone")."""
    parts = collision_name.split("::")
    if len(parts) > 1 and parts[0] == WORLD_SCOPE:
        parts = parts[1:]
    return parts[0]


def tilt_deg(quaternion_xyzw: ArrayLike) -> float:
    """Angle between body +z and world +z in degrees (R₂₂ = 1 − 2(x² + y²))."""
    q = np.asarray(quaternion_xyzw, float)
    x, y = q[:2] / np.linalg.norm(q)
    return math.degrees(math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def _finite(*arrays: ArrayLike) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(a, float)))) for a in arrays)


class RunHealth:
    """Streaming health detector for one run (see module docs).

    ``ground_height_m(x, y)`` is the ground surface height in metres (flat
    ground: 0); ``ground_half_m`` is half the ground plane's side, or None to
    skip the ground-area check. All ``sim_time_ns`` are simulation
    nanoseconds from the source messages.
    """

    def __init__(
        self,
        thresholds: HealthThresholds,
        ground_height_m: Callable[[float, float], float] = lambda x, y: 0.0,
        ground_half_m: float | None = None,
    ) -> None:
        self.thresholds = thresholds
        self._ground_height_m = ground_height_m
        self._ground_half_m = ground_half_m
        self._origin_ns: int | None = None
        self._findings: dict[tuple[str, str | None], Finding] = {}
        self._speeds: deque[tuple[int, float]] = deque()
        self._flip_start_s: float | None = None
        self._flip_reported = False
        self._last_contact_ns: int | None = None
        self._commands = 0
        self._saturated = 0
        self.left_ground_area_s: float | None = None

    def _time_s(self, sim_time_ns: int) -> float:
        if self._origin_ns is None:
            return 0.0
        return (sim_time_ns - self._origin_ns) * 1e-9

    def _report(
        self,
        kind: str,
        severity: str,
        time_s: float | None,
        message: str,
        value: float | None = None,
        subject: str | None = None,
    ) -> None:
        """Record an occurrence: first time and message kept, worst value kept."""
        existing = self._findings.get((kind, subject))
        if existing is None:
            self._findings[(kind, subject)] = Finding(
                kind, severity, time_s, message, value, 1, subject
            )
            return
        existing.count += 1
        if value is not None and (existing.value is None or value > existing.value):
            existing.value = value

    def on_truth(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        velocity_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
    ) -> None:
        """One true state sample (world position, world velocity, xyzw)."""
        if self._origin_ns is None:
            self._origin_ns = sim_time_ns
        t = self._time_s(sim_time_ns)
        if not _finite(position_m, velocity_mps, quaternion_xyzw):
            self._report("non_finite", FAILED, t, "non-finite truth", subject="truth")
            return
        self._speeds.append((sim_time_ns, float(np.linalg.norm(velocity_mps))))
        # One second of speeds is plenty for any sensible impact window.
        while sim_time_ns - self._speeds[0][0] > 1e9:
            self._speeds.popleft()
        self._check_tilt(t, quaternion_xyzw)
        self._check_ground(t, np.asarray(position_m, float))

    def _check_tilt(self, t: float, quaternion_xyzw: ArrayLike) -> None:
        """Report each over-tilt episode that lasts ``flip_duration_s``."""
        tilt, limits = tilt_deg(quaternion_xyzw), self.thresholds
        if tilt <= limits.flip_tilt_deg:
            self._flip_start_s, self._flip_reported = None, False
            return
        if self._flip_start_s is None:
            self._flip_start_s = t
        if t - self._flip_start_s < limits.flip_duration_s:
            return
        if self._flip_reported:
            finding = self._findings[("flipped", None)]
            finding.value = max(finding.value or 0.0, tilt)
            return
        self._flip_reported = True
        self._report(
            "flipped",
            FAILED,
            self._flip_start_s,
            f"tilt above {limits.flip_tilt_deg:g}° for {limits.flip_duration_s:g} s",
            tilt,
        )

    def _check_ground(self, t: float, p: np.ndarray) -> None:
        depth = float(self._ground_height_m(float(p[0]), float(p[1])) - p[2])
        if depth > self.thresholds.below_ground_margin_m:
            self._report(
                "below_ground", FAILED, t, "body centre below the ground", depth
            )
        half = self._ground_half_m
        if half is not None and float(np.max(np.abs(p[:2]))) > half:
            if self.left_ground_area_s is None:
                self.left_ground_area_s = t
            self._report(
                "left_ground_area",
                WARNING,
                self.left_ground_area_s,
                f"left the {2 * half:g} m ground area; nothing below collides",
            )

    def on_contact(self, sim_time_ns: int, collision1: str, collision2: str) -> None:
        """One contact reported by the drone's contact sensor."""
        models = {contact_model(collision1), contact_model(collision2)}
        others = sorted(models - IGNORED_MODELS)
        if not others:
            return
        t = self._time_s(sim_time_ns)
        for model in others:
            if model not in GROUND_MODELS:
                self._report(
                    "obstacle_contact",
                    FAILED,
                    t,
                    f"contact with {model}",
                    subject=model,
                )
        self._check_impact(sim_time_ns, t, others)
        self._check_touchdown(sim_time_ns, t)
        self._last_contact_ns = sim_time_ns

    def _check_impact(self, sim_time_ns: int, t: float, models: list[str]) -> None:
        window_ns = self.thresholds.impact_window_s * 1e9
        speed = max(
            (s for ns, s in self._speeds if 0 <= sim_time_ns - ns <= window_ns),
            default=0.0,
        )
        if speed > self.thresholds.hard_impact_speed_mps:
            self._report(
                "hard_impact",
                FAILED,
                t,
                f"contact with {', '.join(models)} at {speed:.2f} m/s",
                speed,
            )

    def _check_touchdown(self, sim_time_ns: int, t: float) -> None:
        """Warn on a new touchdown; resting on the ground from the start is fine."""
        gap_ns = self.thresholds.touchdown_gap_s * 1e9
        previous = self._last_contact_ns
        if previous is None:
            previous = self._origin_ns if self._origin_ns is not None else sim_time_ns
        if sim_time_ns - previous > gap_ns:
            self._report("ground_contact", WARNING, t, "touched down after flight")

    def on_command(self, record: Mapping[str, Any]) -> None:
        """One control-loop record (``accepted_thrust_n``, ``saturated``)."""
        if record.get("warmup"):
            return
        if not _finite(record["accepted_thrust_n"]):
            t = self._time_s(record["release_sim_time_ns"])
            self._report(
                "non_finite",
                FAILED,
                t,
                "non-finite thrust command",
                subject="command",
            )
        self._commands += 1
        self._saturated += bool(record["saturated"])

    def on_estimate(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        velocity_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
    ) -> None:
        """One estimator output; only checked for non-finite values."""
        if not _finite(position_m, velocity_mps, quaternion_xyzw):
            t = self._time_s(sim_time_ns)
            self._report(
                "non_finite", FAILED, t, "non-finite estimate", subject="estimate"
            )

    def summary(
        self, *, completed: bool, exit_reason: str, dropped: Mapping[str, int]
    ) -> dict[str, Any]:
        """Add run-level findings and return ``summary.json.health``.

        ``dropped`` counts messages lost per kind (ingress queues, command
        events). Call once, at the end of the run.
        """
        self._run_level_findings(completed, exit_reason, dropped)
        findings = sorted(
            self._findings.values(),
            key=lambda f: (math.inf if f.time_s is None else f.time_s, f.kind),
        )
        severities = {f.severity for f in findings}
        verdict = FAILED if FAILED in severities else WARNING if severities else OK
        return dict(
            verdict=verdict,
            findings=[asdict(f) for f in findings],
            thresholds=asdict(self.thresholds),
        )

    def _run_level_findings(
        self, completed: bool, exit_reason: str, dropped: Mapping[str, int]
    ) -> None:
        if not completed and exit_reason == "interrupted":
            self._report("interrupted", WARNING, None, "run interrupted by the user")
        elif not completed and exit_reason not in USER_EXITS:
            self._report("incomplete", FAILED, None, f"run ended early ({exit_reason})")
        if self._commands:
            fraction = self._saturated / self._commands
            if fraction > self.thresholds.saturated_fraction:
                self._report(
                    "saturated",
                    WARNING,
                    None,
                    f"{fraction:.0%} of commands saturated",
                    fraction,
                )
        lost = {kind: count for kind, count in dropped.items() if count}
        total = sum(lost.values())
        if total > self.thresholds.max_dropped_messages:
            self._report(
                "dropped_messages", WARNING, None, f"dropped messages: {lost}", total
            )
