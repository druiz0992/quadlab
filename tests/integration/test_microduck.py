"""Pollen Robotics' Microduck in MuJoCo (opt-in: needs a microduck_rl checkout).

Set ``ROBOTLAB_MICRODUCK_MODEL`` to its MJCF scene, e.g.
``../microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml``; the
model is not part of this repository (docs/9_bipeds.md). The walking test
also needs ``ROBOTLAB_MICRODUCK_POLICY`` (``velstand.onnx`` from
huggingface.co/pollen-robotics/microduck-policies) and onnxruntime.
"""

import json
import os
from pathlib import Path

import pytest

from robotlab.experiments.run import GazeboConfig, run

MODEL = os.environ.get("ROBOTLAB_MICRODUCK_MODEL")
POLICY = os.environ.get("ROBOTLAB_MICRODUCK_POLICY")
pytestmark = pytest.mark.skipif(
    not MODEL, reason="set ROBOTLAB_MICRODUCK_MODEL to microduck_rl's scene.xml"
)
pytest.importorskip("mujoco")


def test_microduck_loads_and_runs(tmp_path: Path) -> None:
    layer = tmp_path / "local.json"
    layer.write_text(json.dumps({"biped": {"model_path": MODEL}}))
    summary = run(
        GazeboConfig(
            drone=("configs/bipeds/microduck.json", str(layer)),
            output=str(tmp_path / "out"),
            duration=2.0,
        )
    )
    assert summary is not None and summary["completed"]
    assert len(summary["robot"]["joint_names"]) == 14
    assert summary["robot"]["model"]["overrides"] == {}
    # Without a balance policy the held STAND pose tips over (phase 3 runs
    # Pollen's policies); the simulation itself stays healthy.
    kinds = {finding["kind"] for finding in summary["health"]["findings"]}
    assert kinds <= {"fallen"}


@pytest.mark.skipif(not POLICY, reason="set ROBOTLAB_MICRODUCK_POLICY to velstand.onnx")
def test_microduck_stands_and_walks_with_velstand(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    layer = tmp_path / "local.json"
    options = {"policy_path": POLICY, "twist": [0.3, 0.0, 0.0], "twist_after_s": 3.0}
    layer.write_text(
        json.dumps(
            {
                "biped": {"model_path": MODEL},
                "algorithms": {"controller_options": options},
            }
        )
    )
    out = tmp_path / "out"
    summary = run(
        GazeboConfig(
            drone=("configs/bipeds/microduck_walk.json", str(layer)),
            output=str(out),
            duration=10.0,
        )
    )
    assert summary is not None and summary["health"]["verdict"] == "ok"
    telemetry = [
        json.loads(line) for line in (out / "telemetry.jsonl").read_text().splitlines()
    ]
    at = {round(t["sim_time_ns"] * 1e-9, 2): t["position_m"] for t in telemetry}
    # Verified 2026-10-25 (microduck_rl b02fdea, velstand from the policy set
    # at 665ae4e, MuJoCo position actuators): stands still, then walks about
    # 0.1 m/s for a 0.3 m/s command (Pollen trains with BAM servo models).
    assert abs(at[3.0][0]) < 0.01
    assert 0.4 < at[10.0][0] - at[3.0][0] < 1.0
