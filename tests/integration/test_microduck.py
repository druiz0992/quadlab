"""Pollen Robotics' Microduck in MuJoCo (opt-in: needs a microduck_rl checkout).

Set ``ROBOTLAB_MICRODUCK_MODEL`` to its MJCF scene, e.g.
``../microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml``; the
model is not part of this repository (docs/9_bipeds.md).
"""

import json
import os
from pathlib import Path

import pytest

from robotlab.experiments.gazebo import GazeboConfig, run

MODEL = os.environ.get("ROBOTLAB_MICRODUCK_MODEL")
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
