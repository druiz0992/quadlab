"""Plug-in debug channel: validated time series to a JSONL file and Rerun.

Backs ``AlgorithmContext.log`` (see core/algorithms.py). Each entry is one
line of ``algorithm-debug.jsonl``: ``{"name", "sim_time_ns", "values",
"labels"}``. Invalid entries raise ValueError in the plug-in's call, so a
mistake shows up where it is made instead of as a corrupt file.
"""

from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any, TextIO

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")


class AlgorithmDebugLog:
    """Write debug entries to ``path`` (opened on the first entry, so runs
    without debug output get no file); optionally forward to Rerun.

    ``rerun(name, sim_time_ns, values, labels)`` is called for each entry
    when given (live runs). Call ``close`` at the end of the run.
    """

    def __init__(
        self,
        path: Path,
        rerun: Callable[[str, int, list[float], list[str]], None] | None = None,
    ) -> None:
        self._path = path
        self._stream: TextIO | None = None
        self._rerun = rerun
        self.entries = 0

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __call__(
        self,
        name: str,
        sim_time_ns: int,
        values: Sequence[float],
        labels: Sequence[str] | None = None,
    ) -> None:
        values, labels = _validated(name, sim_time_ns, values, labels)
        if self._stream is None:
            self._stream = self._path.open("w")
        self._stream.write(
            json.dumps(
                dict(name=name, sim_time_ns=sim_time_ns, values=values, labels=labels)
            )
            + "\n"
        )
        self.entries += 1
        if self._rerun is not None:
            self._rerun(name, sim_time_ns, values, labels)


def _validated(
    name: Any, sim_time_ns: Any, values: Any, labels: Any
) -> tuple[list[float], list[str]]:
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"Debug name {name!r}: use letters, digits, _ . / -")
    if isinstance(sim_time_ns, bool) or not isinstance(sim_time_ns, int):
        raise ValueError("Debug sim_time_ns must be an integer (simulation ns)")
    numbers = [float(v) for v in values]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"Debug values for {name!r} must be finite")
    names = [str(i) for i in range(len(numbers))] if labels is None else list(labels)
    if len(names) != len(numbers):
        raise ValueError(
            f"Debug {name!r}: {len(names)} labels for {len(numbers)} values"
        )
    return numbers, names
