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
