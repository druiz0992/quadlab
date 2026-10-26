"""Observer estimators: extra estimators beside the primary in one live run.

``algorithms.observers`` lists estimators that receive the same bus
deliveries as ``algorithms.estimator`` (the primary) and are scored against
the same truth, but never drive control or the health verdict, e.g.::

    "algorithms": {"estimator": "gps_imu_ekf",
                   "observers": [{"name": "imu_example", "estimator": "filter",
                                  "estimator_options": {"filter": "..."}}]}

Each observer is built like the primary (``build_estimator``; a
``gps_imu_ekf`` observer without options uses the ``ekf`` section) with
its own context debug channel under ``observers/<name>/``. The runner
delivers each sample to the primary first, then to the observers in list
order, serially on the mission thread, so the primary's outputs are the
same with or without observers. Outputs per observer:
``observers/<name>/estimates.jsonl`` (same records as ``estimates.jsonl``),
``metrics.json`` ``observers.<name>`` (the ``estimation`` section's
schema), ``summary.json`` ``algorithms.observers.<name>`` (estimator spec,
estimate count, ``estimator_summary``), and ``comparison.md``, a table of
the primary and every observer (best error per row in bold). Observers
are not drawn in Rerun and are not recorded for replay; replay them with
``robotlab-replay --estimator``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, TextIO

from ...core.algorithms import AlgorithmContext, DebugLog
from ...core.measurements import EstimatedState, Sample
from ...estimation.interface import StateEstimator
from .replay_log import estimate_record
from .run_evaluation import RunEvaluation

PRIMARY_LABEL = "primary"


@dataclass
class Observer:
    """One observer estimator and its outputs (estimates file, count)."""

    name: str
    spec: str
    estimator: StateEstimator
    latest: Sample[EstimatedState] | None = None
    estimates: int = 0
    stream: TextIO | None = field(default=None, repr=False)


class ObserverSet:
    """Feed, record, and score the observers of one run (module docs).

    ``attach`` opens the output files and connects the run's evaluation;
    until then deliveries raise RuntimeError (none may be lost silently).
    """

    def __init__(self, observers: list[Observer]) -> None:
        self.observers = observers
        self._evaluation: RunEvaluation | None = None

    def attach(self, output: Path, evaluation: RunEvaluation) -> None:
        for observer in self.observers:
            directory = output / "observers" / observer.name
            directory.mkdir(parents=True, exist_ok=False)
            observer.stream = (directory / "estimates.jsonl").open("w")
            evaluation.add_observer(observer.name)
        self._evaluation = evaluation

    def deliver(self, delivered: list[tuple[str, Sample[Any]]]) -> None:
        """Give each observer the delivered samples of its input kinds."""
        evaluation = self._evaluation
        if evaluation is None:
            raise RuntimeError("ObserverSet.deliver before attach")
        for observer in self.observers:
            inputs = observer.estimator.inputs
            for kind, sample in delivered:
                if kind in inputs:
                    self._feed(observer, kind, sample, evaluation)
            evaluation.on_observer_innovations(
                observer.name, observer.estimator.drain_innovations()
            )

    def _feed(
        self,
        observer: Observer,
        kind: str,
        sample: Sample[Any],
        evaluation: RunEvaluation,
    ) -> None:
        observer.estimator.on_sample(kind, sample)
        latest = observer.estimator.latest()
        if latest is None or latest is observer.latest:
            return
        observer.latest = latest
        observer.estimates += 1
        if observer.stream is not None:
            observer.stream.write(json.dumps(estimate_record(latest)) + "\n")
        evaluation.on_observer_estimate(observer.name, latest)

    def close(self) -> None:
        for observer in self.observers:
            if observer.stream is not None:
                observer.stream.close()
                observer.stream = None

    def summary(self) -> dict[str, Any]:
        """``summary.json`` ``algorithms.observers``."""
        return {
            o.name: dict(
                estimator=o.spec,
                estimates=o.estimates,
                estimator_summary=o.estimator.summary(),
            )
            for o in self.observers
        }


def build_observers(
    settings: Mapping[str, Any],
    context: AlgorithmContext,
    build: Callable[[str, Mapping[str, Any], AlgorithmContext], StateEstimator],
) -> ObserverSet | None:
    """The configured observers (None without any); ``build(spec, options,
    context)`` constructs one estimator as for the primary."""
    entries = settings["algorithms"]["observers"]
    if not entries:
        return None
    observers = []
    for entry in entries:
        name = entry["name"]
        own = replace(context, log=_prefixed(context.log, f"observers/{name}/"))
        estimator = build(entry["estimator"], entry.get("estimator_options", {}), own)
        observers.append(Observer(name, entry["estimator"], estimator))
    return ObserverSet(observers)


def _prefixed(log: DebugLog, prefix: str) -> DebugLog:
    """``log`` with every debug name under ``prefix``."""

    def prefixed(
        name: str,
        sim_time_ns: int,
        values: Sequence[float],
        labels: Sequence[str] | None = None,
    ) -> None:
        log(prefix + name, sim_time_ns, values, labels)

    return prefixed


def write_comparison(
    output: Path, metrics: Mapping[str, Any], primary: str
) -> str | None:
    """``comparison.md``: the primary (labelled ``primary``) and every
    observer over ``COMPARISON_METRICS``; None (no file) without observers."""
    from .batch import lookup
    from .comparison import comparison_markdown, stats
    from .replay import COMPARISON_METRICS, comparison_directions

    observers = metrics.get("observers")
    if not observers:
        return None
    sections = {PRIMARY_LABEL: metrics.get("estimation"), **observers}
    columns = {
        label: {m: stats([lookup(section or {}, m)]) for m in COMPARISON_METRICS}
        for label, section in sections.items()
    }
    notes = {PRIMARY_LABEL: f"`{primary}`" if primary else "no primary estimator"}
    table = comparison_markdown(
        columns, COMPARISON_METRICS, comparison_directions(), notes
    )
    (output / "comparison.md").write_text(table)
    return table
