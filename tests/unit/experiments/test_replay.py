"""Replay recording, codec, debug channel, and offline replay (no Gazebo)."""

import json
from types import MappingProxyType

import numpy as np
import pytest

from robotlab.cli.replay import parse_overrides
from robotlab.core.algorithms import AlgorithmContext, SensorDescription
from robotlab.core.measurements import (
    CameraFrame,
    GpsReading,
    GpsStatus,
    ImuReading,
    Sample,
)
from robotlab.core.serialization import decode, encode
from robotlab.core.vehicle import DroneProperties
from robotlab.experiments.harness.algorithm_debug import AlgorithmDebugLog
from robotlab.experiments.harness.replay import replay
from robotlab.experiments.harness.replay_log import ReplayRecorder
from robotlab.simulation.gazebo.config import load_drone_config


def roundtrip(value):
    return decode(json.loads(json.dumps(encode(value))))


def imu_sample(sequence, t_ns):
    reading = ImuReading(np.array([0.0, 0.0, 9.81]), np.array([1e-3, -2e-3, 0.0]))
    return Sample("sensors/imu", sequence, 0, t_ns, 0, "drone/imu", reading)


# Codec ---------------------------------------------------------------------------


def test_codec_roundtrips_samples_exactly():
    sample = imu_sample(3, 20_000_000)
    restored = roundtrip(sample)
    assert type(restored) is Sample and restored.sim_time_ns == 20_000_000
    np.testing.assert_array_equal(
        restored.payload.angular_rate_rad_s, sample.payload.angular_rate_rad_s
    )
    gps = GpsReading(
        40.123456789,
        -3.7,
        650.25,
        np.array([0.1, 0.0, -0.3]),
        GpsStatus("3d", 9, 0.8, 1.2, 2.0, 0.1),
    )
    back = roundtrip(gps)
    assert back.latitude_deg == gps.latitude_deg and back.status == gps.status
    np.testing.assert_array_equal(back.velocity_enu_mps, gps.velocity_enu_mps)
    frame = roundtrip(CameraFrame(np.zeros((2, 3, 3), dtype=np.uint8)))
    assert frame.image.dtype == np.uint8 and frame.image.shape == (2, 3, 3)


def test_codec_context_skips_log_and_refuses_foreign_types():
    context = AlgorithmContext(
        GpsReading(40.0, -3.7, 650.0),
        DroneProperties(),
        {"imu": SensorDescription("imu", "drone/imu", 200.0)},
        log=lambda *a: None,
    )
    encoded = encode(context)
    assert "log" not in encoded["fields"]
    restored = decode(json.loads(json.dumps(encoded)))
    assert isinstance(restored.sensors, MappingProxyType)
    assert restored.sensors["imu"].rate_hz == 200.0 and restored == context
    foreign = {"__dataclass__": "subprocess:Popen", "fields": {}}
    with pytest.raises(ValueError, match="not a robotlab type"):
        decode(foreign)


# Debug channel -------------------------------------------------------------------


def test_debug_log_validates_and_opens_lazily(tmp_path):
    path = tmp_path / "algorithm-debug.jsonl"
    forwarded = []
    log = AlgorithmDebugLog(path, rerun=lambda *entry: forwarded.append(entry))
    assert not path.exists()
    log("ekf/nis", 5, [1.5, 2.0], ["gps", "baro"])
    log("gain", 6, [0.25])
    for bad in (
        ("bad name!", 1, [1.0], None),
        ("x", 1.5, [1.0], None),
        ("x", 1, [float("nan")], None),
        ("x", 1, [1.0, 2.0], ["one"]),
    ):
        with pytest.raises(ValueError):
            log(*bad)
    log.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [
        dict(name="ekf/nis", sim_time_ns=5, values=[1.5, 2.0], labels=["gps", "baro"]),
        dict(name="gain", sim_time_ns=6, values=[0.25], labels=["0"]),
    ]
    assert log.entries == 2 and forwarded[1] == ("gain", 6, [0.25], ["0"])


# Recording and replay -------------------------------------------------------------

ESTIMATOR = '''
import numpy as np
from robotlab.core.measurements import EstimatedState, Sample

class Echo:
    """Test estimator: position = configured offset; logs every sample."""
    inputs = ("imu",)

    def __init__(self, context, options):
        self.context, self.offset = context, float(options.get("offset", 0.0))
        self.latest_sample, self.seen = None, 0

    def on_sample(self, kind, sample):
        self.seen += 1
        state = EstimatedState(
            np.array([self.offset, 0.0, 1.0]), np.zeros(3), np.array([0, 0, 0, 1.0]),
            np.zeros(3), np.zeros(3), np.eye(15),
        )
        self.latest_sample = Sample(
            "estimate", self.seen, 0, sample.sim_time_ns, 0, "world", state
        )
        self.context.log("echo/seen", sample.sim_time_ns, [self.seen])

    def latest(self):
        return self.latest_sample

    def drain_innovations(self):
        return []

    def summary(self):
        return dict(seen=self.seen)
'''


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A recorded run: 3 s of 50 Hz truth at z = 1 m and 200 Hz IMU samples."""
    (tmp_path / "replay_echo.py").write_text(ESTIMATOR)
    monkeypatch.syspath_prepend(str(tmp_path))
    settings = load_drone_config().data
    settings["algorithms"]["estimator"] = "replay_echo:Echo"
    context = AlgorithmContext(
        GpsReading(40.0, -3.7, 650.0),
        DroneProperties(),
        {"imu": SensorDescription("imu", "drone/imu", 200.0)},
    )
    path = tmp_path / "replay.jsonl"
    recorder = ReplayRecorder(path, "light", settings, context, weight_n=14.7)
    sequence = 0
    for tick in range(1, 151):
        t_ns = tick * 20_000_000
        for sub in range(4):
            sequence += 1
            recorder.on_sample(
                t_ns, "imu", imu_sample(sequence, t_ns - 15_000_000 + sub * 5_000_000)
            )
        recorder.on_sample(t_ns, "camera", "skipped in light mode")
        recorder.on_truth(
            t_ns, [0, 0, 1], [0, 0, 0], [0, 0, 0, 1], (np.zeros(3), np.zeros(3))
        )
    recorder.close()
    assert recorder.summary() == dict(
        mode="light", samples=600, truth_samples=150, skipped={"camera": 150}
    )
    return path


def test_replay_scores_estimator_and_records_debug(recorded, tmp_path):
    summary = replay(recorded, tmp_path / "out")
    assert summary["estimator"] == "replay_echo:Echo"
    assert summary["delivered"] == 600 and summary["estimates"] == 600
    assert summary["estimator_summary"] == {"seen": 600}
    assert summary["algorithm_debug_entries"] == 600
    errors = json.loads((tmp_path / "out" / "metrics.json").read_text())["estimation"]
    assert errors["errors"]["position_m"]["rmse"] == pytest.approx(0.0)
    debug = (tmp_path / "out" / "algorithm-debug.jsonl").read_text().splitlines()
    assert json.loads(debug[0])["name"] == "echo/seen"


def test_replay_overrides_are_validated_and_applied(recorded, tmp_path):
    replay(
        recorded,
        tmp_path / "shift",
        {"algorithms.estimator_options": {"offset": 0.5}},
    )
    metrics = json.loads((tmp_path / "shift" / "metrics.json").read_text())
    assert metrics["estimation"]["errors"]["position_m"]["rmse"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        replay(recorded, tmp_path / "bad", {"algorithms.record_bus": "sometimes"})
    with pytest.raises(ValueError, match="not empty"):
        replay(recorded, tmp_path / "shift")
    with pytest.raises(ValueError, match="No estimator"):
        replay(recorded, tmp_path / "none", {"algorithms.estimator": ""})


def test_replay_rejects_other_files(tmp_path):
    other = tmp_path / "telemetry.jsonl"
    other.write_text('{"time": 0}\n')
    with pytest.raises(ValueError, match="not a robotlab-replay log"):
        replay(other, tmp_path / "out")


def test_cli_override_parsing():
    assert parse_overrides(["ekf.gps_std_m=2.5", "a.b=text", 'c={"k": [1]}']) == {
        "ekf.gps_std_m": 2.5,
        "a.b": "text",
        "c": {"k": [1]},
    }
    with pytest.raises(ValueError):
        parse_overrides(["novalue"])


def test_compare_estimators_side_by_side(recorded, tmp_path):
    from robotlab.experiments.harness.replay import compare_estimators, estimator_label

    taken = set()
    assert estimator_label("a.b:Echo", taken) == "Echo"
    assert estimator_label("other:Echo", taken) == "Echo-2"
    table = compare_estimators(
        recorded,
        tmp_path / "cmp",
        ["replay_echo:Echo", "replay_echo:Echo"],
        {"algorithms.estimator_options": {"offset": 0.0}},
    )
    assert (tmp_path / "cmp" / "Echo" / "metrics.json").exists()
    assert (tmp_path / "cmp" / "Echo-2" / "estimates.jsonl").exists()
    assert "| Metric | Echo | Echo-2 |" in table
    assert (tmp_path / "cmp" / "comparison.md").read_text() == table


def test_removed_live_ekf_settings_are_migrated() -> None:
    from robotlab.experiments.harness.replay import migrate_settings

    recorded: dict = {"ekf": {"gps_std_m": [1, 1, 1], "reorder_capacity": 4096}, "x": 1}
    settings, dropped = migrate_settings(recorded)
    assert dropped == ["ekf.reorder_capacity"]
    assert settings == {"ekf": {"gps_std_m": [1, 1, 1]}, "x": 1}
    assert "reorder_capacity" in recorded["ekf"]  # input untouched
