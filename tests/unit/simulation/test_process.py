"""GazeboProcess with stand-in processes (no Gazebo needed; Linux /proc)."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from robotlab.simulation.gazebo.process import (
    GazeboProcess,
    descendants,
    parent_death_prefix,
)

linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="needs Linux /proc"
)

# A launcher like ``gz sim`` with the GUI: its child runs in its own session
# (so a process-group kill misses it); this one also ignores SIGTERM.
STUBBORN_LAUNCHER = """
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         start_new_session=True)
print(child.pid, flush=True)
time.sleep(60)
"""


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return False
    return state != "Z"  # a zombie has exited


def wait_until(condition, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_parent_death_prefix_only_with_setpriv_on_linux():
    found = lambda name: f"/usr/bin/{name}"  # noqa: E731
    assert parent_death_prefix(found, "linux") == [
        "/usr/bin/setpriv",
        "--pdeathsig",
        "TERM",
        "--",
    ]
    assert parent_death_prefix(lambda name: None, "linux") == []
    assert parent_death_prefix(found, "darwin") == []


def test_descendants_walks_the_proc_tree(tmp_path):
    def task(pid, children):
        path = tmp_path / str(pid) / "task" / str(pid)
        path.mkdir(parents=True)
        (path / "children").write_text(" ".join(map(str, children)))

    task(10, [11, 12])
    task(11, [13])
    task(12, [])
    task(13, [])
    assert sorted(descendants(10, tmp_path)) == [11, 12, 13]
    assert descendants(99, tmp_path) == []


def test_empty_command_rejected():
    with pytest.raises(ValueError):
        GazeboProcess([], None)


@linux
def test_stop_kills_a_stubborn_launcher_and_its_detached_child(tmp_path):
    log = (tmp_path / "log").open("w+")
    process = GazeboProcess([sys.executable, "-c", STUBBORN_LAUNCHER], log, prefix=[])
    assert wait_until(lambda: Path(log.name).read_text().strip())
    child = int(Path(log.name).read_text().split()[0])
    assert alive(child)
    process.stop(timeout_s=0.3)
    assert process.poll() is not None
    assert process.pid in process.killed and child in process.killed
    assert wait_until(lambda: not alive(child))
    process.stop()  # idempotent


@linux
def test_stop_returns_status_of_a_cooperative_process():
    process = GazeboProcess(
        [sys.executable, "-c", "import time; time.sleep(60)"], None, prefix=[]
    )
    assert process.stop(timeout_s=5) == -signal.SIGTERM
    assert process.killed == []


@linux
@pytest.mark.skipif(not parent_death_prefix(), reason="needs setpriv")
def test_child_stops_when_its_parent_is_killed(tmp_path):
    # The parent starts a GazeboProcess and is then SIGKILLed: no cleanup
    # code runs, yet the parent-death signal ends the child.
    parent_code = (
        "import sys, time\n"
        "from robotlab.simulation.gazebo.process import GazeboProcess\n"
        "p = GazeboProcess([sys.executable, '-c', 'import time; time.sleep(60)'], None)\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True
    )
    child = int(parent.stdout.readline())
    # setpriv requests the signal before it execs the command: wait for the
    # exec, since a parent that dies before the request is not reported.
    cmdline = Path(f"/proc/{child}/cmdline")
    assert wait_until(lambda: b"setpriv" not in cmdline.read_bytes())
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait()
    assert wait_until(lambda: not alive(child))
