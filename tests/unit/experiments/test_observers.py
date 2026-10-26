"""Observer estimators: same deliveries as the primary, own outputs, scoring."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robotlab.core.measurements import GpsReading, ImuReading
from robotlab.experiments.common.mission import MissionRunner
from robotlab.experiments.harness.algorithms import (
    AlgorithmRunner,
    build_context,
    build_estimator,
    build_named_estimator,
)
from robotlab.experiments.harness.observers import build_observers, write_comparison
from robotlab.experiments.harness.replay_log import estimate_record
from robotlab.experiments.harness.run_evaluation import RunEvaluation
from robotlab.sensors.bus import SensorBus
from robotlab.simulation.gazebo.config import load_drone_config

MS = 1_000_000
ORIGIN = GpsReading(40.4168, -3.7038, 650.0)
HOVER_IMU = ImuReading(np.array([0, 0, 9.81]), np.zeros(3))
IMU_EXAMPLE = dict(
    filter="robotlab.estimation.examples.imu_gps_filter:ImuGpsFilter",
    filter_options=dict(initial_position_m=[0, 0, 1]),
)


def settings(observers: list[dict[str, Any]], estimator: str = "gps_imu_ekf") -> Any:
    data = load_drone_config().data
    data["algorithms"].update(estimator=estimator, observers=observers)
    data["ekf"]["initial_position_m"] = [0, 0, 1]
    data["sensor_timing"]["reorder_window_s"] = 0.0
    return data


def run(data: Any, output: Path) -> tuple[list[Any], RunEvaluation, list[str], Any]:
    """Two seconds of hover samples; returns primary estimates, evaluation,
    debug names, and the observer set."""
    names: list[str] = []
    context = build_context(
        data,
        load_drone_config().properties,
        ORIGIN,
        ("imu", "gps"),
        log=lambda name, *_: names.append(name),
    )
    observers = build_observers(
        data,
        context,
        lambda spec, options, own: build_named_estimator(data, spec, options, own),
    )
    bus = SensorBus(data["sensor_timing"], ("imu", "gps"))
    runner = AlgorithmRunner(bus, build_estimator(data, context), observers)
    evaluation = RunEvaluation(
        data, 10.0, MissionRunner(data["mission"]), lambda *args: None
    )
    if observers is not None:
        observers.attach(output, evaluation)
    estimates = []
    for i in range(1, 501):
        t = i * 4 * MS
        bus.submit("imu", HOVER_IMU, t)
        if i % 25 == 0:
            bus.submit(
                "gps", GpsReading(ORIGIN.latitude_deg, ORIGIN.longitude_deg, 651.0), t
            )
        _, new, innovations = runner.advance(t)
        estimates += new
        for sample in new:
            evaluation.on_estimate(sample)
        evaluation.on_innovations(innovations)
        evaluation.on_truth(
            t,
            np.array([0, 0, 1.0]),
            np.zeros(3),
            np.array([0, 0, 0, 1.0]),
            controlled=False,
        )
    if observers is not None:
        observers.close()
    return estimates, evaluation, names, observers


def test_observers_see_the_primary_deliveries_without_changing_it(
    tmp_path: Path,
) -> None:
    alone, _, _, _ = run(settings([]), tmp_path / "alone")
    observed: list[dict[str, Any]] = [
        dict(name="twin", estimator="gps_imu_ekf"),
        dict(name="imu_example", estimator="filter", estimator_options=IMU_EXAMPLE),
    ]
    estimates, evaluation, names, observers = run(settings(observed), tmp_path)
    records = [json.dumps(estimate_record(s)) for s in estimates]
    assert records == [json.dumps(estimate_record(s)) for s in alone]
    # The same filter as observer reproduces the primary exactly.
    twin = (tmp_path / "observers/twin/estimates.jsonl").read_text().splitlines()
    assert twin == records
    example = tmp_path / "observers/imu_example/estimates.jsonl"
    assert len(example.read_text().splitlines()) > 400
    metrics = evaluation.summary()
    assert set(metrics["observers"]) == {"twin", "imu_example"}
    assert metrics["observers"]["twin"] == metrics["estimation"]
    # Debug channels: the primary's as before, observers' under their names.
    assert "gps_imu_ekf/gps" in names and "observers/twin/gps_imu_ekf/gps" in names
    summary = observers.summary()
    assert summary["twin"]["estimator"] == "gps_imu_ekf"
    assert summary["imu_example"]["estimates"] == len(example.read_text().splitlines())


def test_observers_run_without_a_primary(tmp_path: Path) -> None:
    observed = [dict(name="ekf", estimator="gps_imu_ekf")]
    estimates, evaluation, _, _ = run(settings(observed, estimator=""), tmp_path)
    assert estimates == []
    metrics = evaluation.summary()
    assert metrics["estimation"] is None and metrics["observers"]["ekf"] is not None
    table = write_comparison(tmp_path, metrics, "")
    assert table is not None and "| primary | ekf |" in table
    assert (tmp_path / "comparison.md").read_text() == table


def test_no_observers_no_comparison_and_unchanged_metrics(tmp_path: Path) -> None:
    _, evaluation, _, observers = run(settings([]), tmp_path)
    assert observers is None and "observers" not in evaluation.summary()
    assert write_comparison(tmp_path, evaluation.summary(), "gps_imu_ekf") is None
    assert not (tmp_path / "comparison.md").exists()


def test_delivery_before_attach_raises() -> None:
    data = settings([dict(name="twin", estimator="gps_imu_ekf")])
    context = build_context(
        data, load_drone_config().properties, ORIGIN, ("imu", "gps")
    )
    observers = build_observers(
        data,
        context,
        lambda spec, options, own: build_named_estimator(data, spec, options, own),
    )
    assert observers is not None
    with pytest.raises(RuntimeError, match="before attach"):
        observers.deliver([])


@pytest.mark.parametrize(
    "observers, message",
    [
        ({"name": "a"}, "list"),
        ([{"name": "a"}], "need name and estimator"),
        ([{"name": "a b", "estimator": "x"}], "letters"),
        ([{"name": "primary", "estimator": "x"}], "not primary"),
        ([{"name": "a", "estimator": "x"}, {"name": "a", "estimator": "y"}], "unique"),
        ([{"name": "a", "estimator": ""}], "nonempty"),
        ([{"name": "a", "estimator": "x", "options": {}}], "Unknown"),
        ([{"name": "a", "estimator": "x", "estimator_options": []}], "object"),
    ],
)
def test_config_validation_of_observers(
    tmp_path: Path, observers: Any, message: str
) -> None:
    path = tmp_path / "observers.json"
    path.write_text(json.dumps({"algorithms": {"observers": observers}}))
    with pytest.raises(ValueError, match=message):
        load_drone_config(path)
