"""Plug-in contract for state estimators.

An estimator is constructed as ``cls(context, options)`` with an
``AlgorithmContext`` and its ``algorithms.estimator_options`` mapping. It
receives bus samples (``sensors/<kind>`` streams, see sensors/bus.py) in
delivery order, serially on the mission thread, which is deterministic in
lockstep. Samples of one stream arrive in time order; different streams may
interleave out of time order when latencies differ. It never sees truth.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..core.measurements import EstimatedState, Sample


@dataclass(frozen=True)
class Innovation:
    """One measurement update's normalized innovation squared (NIS)."""

    name: str  # e.g. "gps"
    nis: float
    dof: int
    accepted: bool


@runtime_checkable
class StateEstimator(Protocol):
    """See module docs. ``inputs`` lists the sensor kinds to deliver."""

    inputs: tuple[str, ...]

    def on_sample(self, kind: str, sample: Sample[Any]) -> None:
        """Consume one delivered measurement."""

    def latest(self) -> Sample[EstimatedState] | None:
        """Newest estimate (sim_time_ns = the time it is valid for)."""

    def drain_innovations(self) -> list[Innovation]:
        """Innovations since the last call (may be empty), for NIS checks."""

    def summary(self) -> dict[str, Any]:
        """Counters for summary.json."""
