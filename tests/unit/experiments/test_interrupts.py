"""SIGTERM handling, shielded cleanup, and batch child shutdown."""

import os
from pathlib import Path
import signal
import sys
import threading
import time

import pytest

from robotlab.experiments.batch import ChildRuns
from robotlab.experiments.interrupts import shielded, sigterm_as_interrupt


def test_sigterm_becomes_keyboard_interrupt_and_handler_is_restored():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with sigterm_as_interrupt() as termination:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)  # the handler raises before this returns
    assert termination.received
    assert signal.getsignal(signal.SIGTERM) is before


def test_shielded_defers_signals_without_raising():
    with sigterm_as_interrupt():
        with shielded() as received:
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
        assert received == [signal.SIGINT, signal.SIGTERM]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_helpers_do_nothing_outside_the_main_thread():
    results = []

    def worker():
        with sigterm_as_interrupt() as termination, shielded() as received:
            results.append((termination.received, received))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert results == [(False, [])]


IGNORES_SIGTERM = (
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print('ready', flush=True); time.sleep(60)"
)


def run_in_thread(children, command, run_dir):
    codes = []
    thread = threading.Thread(
        target=lambda: codes.append(children.run(command, run_dir))
    )
    thread.start()
    log = run_dir / "batch.log"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and "ready" not in (
        log.read_text() if log.exists() else ""
    ):
        time.sleep(0.02)
    return thread, codes


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX signals")
def test_child_runs_stop_all_terminates_then_kills(tmp_path: Path):
    children = ChildRuns(stop_timeout_wall_s=0.3)
    polite = tmp_path / "polite"
    stubborn = tmp_path / "stubborn"
    polite.mkdir()
    stubborn.mkdir()
    sleeper = "import time; print('ready', flush=True); time.sleep(60)"
    first, first_codes = run_in_thread(
        children, [sys.executable, "-c", sleeper], polite
    )
    second, second_codes = run_in_thread(
        children, [sys.executable, "-c", IGNORES_SIGTERM], stubborn
    )
    children.stop_all()
    first.join(5)
    second.join(5)
    assert first_codes == [-signal.SIGTERM]
    assert second_codes == [-signal.SIGKILL]
    with pytest.raises(RuntimeError, match="interrupted"):
        children.run([sys.executable, "-c", "pass"], tmp_path)
