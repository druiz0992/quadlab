"""Keyboard walking: ramped commands from held keys, and the operator."""

from typing import Any

import pytest

from robotlab.control.legged_keyboard import WalkInput
from robotlab.experiments.biped_keyboard import KeyboardOperator


def walk() -> WalkInput:
    return WalkInput(
        max_forward_mps=0.3, max_left_mps=0.2, max_yaw_rate_rad_s=1.5, ramp_per_s=2.0
    )


def test_held_keys_ramp_up_and_release_returns_faster() -> None:
    w = walk()
    assert w.update({"Up"}, 0.25).forward_mps == pytest.approx(0.15)
    command = w.update({"Up", "Left", "d"}, 0.5)
    assert command.forward_mps == pytest.approx(0.3)
    assert command.yaw_rate_rad_s == pytest.approx(1.5)
    assert command.left_mps == pytest.approx(-0.2)
    assert w.update(set(), 0.25).forward_mps == 0.0  # 2x ramp back to zero


class FakeWindow:
    def __init__(self) -> None:
        self.closed = False
        self.keys: frozenset[str] = frozenset({"Up"})

    def poll_keys(self) -> frozenset[str]:
        return self.keys

    def close(self) -> None:
        self.closed = True


def test_operator_paces_to_wall_time() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    operator = KeyboardOperator(FakeWindow(), walk(), lambda: now[0], sleep)
    assert operator.request(0).forward_mps == 0.0  # first poll: no elapsed time
    for step in range(1, 51):  # 0.1 s of 2 ms steps, computed instantly
        operator.after_step(None, step * 2_000_000)
    assert sum(sleeps) == pytest.approx(0.1)
    assert operator.request(0).forward_mps == pytest.approx(0.3 * 2.0 * 0.1)


def test_operator_stops_when_the_window_closes() -> None:
    window = FakeWindow()
    operator = KeyboardOperator(window, walk())
    assert operator.running()
    operator.close()
    assert window.closed and not operator.running()
