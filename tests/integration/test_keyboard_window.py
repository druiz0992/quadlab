"""Opt-in desktop event check: ROBOTLAB_TEST_GUI=1 pytest this_file."""

import os
import time

import pytest

from robotlab.visualization.keyboard import KeyboardWindow


@pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GUI") != "1", reason="requires a desktop display"
)
def test_real_keyboard_window():
    window = KeyboardWindow()
    try:
        window.root.focus_force()
        window.poll()
        time.sleep(0.1)
        window.poll()
        window.root.event_generate("<KeyPress-Up>")
        window.root.event_generate("<KeyPress-q>")
        command = window.poll()
        assert command.forward_mps == 1
        assert command.climb_mps == 0.6
        window.root.event_generate("<KeyRelease-Up>")
        window.poll()
        assert window.poll().forward_mps == 0
        window.root.event_generate("<FocusOut>")
        assert window.poll().climb_mps == 0
        window.root.event_generate("<KeyPress-a>")
        assert window.poll().climb_mps == -0.6
        window.root.event_generate("<KeyRelease-a>")
        window.poll()
        assert window.poll().climb_mps == 0
        window.root.event_generate("<KeyPress-Escape>")
        window.poll()
        assert window.closed
    finally:
        window.close()


@pytest.mark.skipif(
    os.environ.get("ROBOTLAB_TEST_GUI") != "1", reason="requires a desktop display"
)
def test_biped_keyboard_walk_paced(tmp_path):
    """Key window on the test biped, paced to real time; closing ends the run."""
    pytest.importorskip("mujoco")
    from robotlab.control.legged_keyboard import WalkInput
    from robotlab.experiments.biped.biped_keyboard import open_keyboard_operator
    from robotlab.experiments.biped.biped_loop import BipedLoop
    from robotlab.experiments.biped.biped_run import _parts
    from robotlab.simulation.mujoco.config import load_biped_config
    from robotlab.simulation.mujoco.scene import build_scene

    config = load_biped_config()
    scene = build_scene(config.data, config.model_path)
    loop = BipedLoop(_parts(config, scene, tmp_path, 30.0))
    operator = open_keyboard_operator(
        WalkInput.from_settings(config.data["biped"]["keyboard"])
    )
    loop.parts.operator = operator
    window = operator.window
    window.root.focus_force()
    window.root.event_generate("<KeyPress-Up>")
    # Closing the window is what the user's Esc or close button does.
    window.root.after(1000, window.close)
    try:
        summary = loop.run()
    finally:
        operator.close()
    assert summary["exit_reason"] == "keyboard_closed"
    assert 0.8 < summary["timing"]["sim_time_s"] < 1.5  # paced to real time
