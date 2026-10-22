"""Opt-in real-time Gazebo check: the mission loop keeps up at 250 Hz.

Run with ``ROBOTLAB_TEST_GAZEBO=1``. Unlike the lockstep regression suite
this runs in real time, so it measures this machine: it flies the
``racing_5in`` quad (250 Hz odometry and control, battery, Rerun file
recording) and checks the loop-timing summary. The bounds leave headroom
for a loaded desktop; a failure means the loop now costs most of the
real-time budget (see ``loop_timing`` in docs/2_simulation.md).
"""

import json
import os

import pytest

from robotlab.experiments.gazebo import GazeboConfig, run

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GAZEBO") != "1",
    reason="requires native Gazebo transport and physics (ROBOTLAB_TEST_GAZEBO=1)",
)

# Busy wall seconds per simulated second; measured 0.49 on the reference
# desktop (headless, 2026-10-21; 0.58 before Rerun scalar batching).
MAX_LOAD = 0.85


def test_250hz_real_time_loop_keeps_up(tmp_path):
    output = tmp_path / "run"
    run(
        GazeboConfig(
            drone=("configs/vehicles/racing_5in.json",),
            output=str(output),
            headless=True,
            control_mode="rotor-thrust",
            duration=10.0,
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    timing = summary["loop_timing"]
    assert summary["completed"] and summary["health"]["verdict"] == "ok"
    assert timing["odometry_queue_age_ms"]["late"] == 0
    assert timing["load"] < MAX_LOAD
