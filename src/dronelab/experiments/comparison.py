"""Side-by-side comparison tables of metric statistics (Markdown).

Columns are algorithms or parameter groups (``dronelab-batch`` groups, or
estimators replayed on one recording); rows are metrics. Each cell shows
``mean ± std (n)``. Where a metric's direction is known, the best column
is marked in bold: ``increase`` means higher is worse (best = lowest),
``decrease`` means lower is worse (best = highest), ``both`` marks nothing,
and ties are not marked.
"""

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


def stats(values: Sequence[float]) -> dict[str, Any]:
    """Count, mean, sample standard deviation, min, max of finite values."""
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return dict(count=0)
    array = np.asarray(finite)
    return dict(
        count=len(finite),
        mean=float(array.mean()),
        std=float(array.std(ddof=1)) if len(finite) > 1 else 0.0,
        min=float(array.min()),
        max=float(array.max()),
    )


def _cell(entry: Mapping[str, Any] | None) -> str:
    if not entry or not entry.get("count"):
        return "—"
    return f"{entry['mean']:.4g} ± {entry['std']:.2g} ({entry['count']})"


def _best(row: dict[str, Mapping[str, Any]], direction: str | None) -> str | None:
    means = {name: e["mean"] for name, e in row.items() if e and e.get("count")}
    if len(means) < 2 or direction not in ("increase", "decrease"):
        return None
    pick = min if direction == "increase" else max
    best = pick(means.values())
    winners = [name for name, mean in means.items() if mean == best]
    return winners[0] if len(winners) == 1 else None  # ties are not marked


def comparison_markdown(
    columns: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics: Sequence[str],
    directions: Mapping[str, str] | None = None,
    notes: Mapping[str, str] | None = None,
) -> str:
    """Markdown table: one row per metric, one column per name in ``columns``.

    ``columns[name][metric]`` is a stats dict (``count``, ``mean``, ``std``);
    ``directions[metric]`` as in batch tolerances; ``notes[name]`` adds a
    line under the table (e.g. excluded runs).
    """
    names = list(columns)
    lines = [
        "| Metric | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for metric in metrics:
        row = {name: columns[name].get(metric) for name in names}
        best = _best(row, (directions or {}).get(metric))
        cells = []
        for name in names:
            text = _cell(row[name])
            cells.append(f"**{text}**" if name == best else text)
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    for name, note in (notes or {}).items():
        if note:
            lines.append(f"\n{name}: {note}")
    return "\n".join(lines) + "\n"
