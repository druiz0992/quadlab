"""The Gazebo process a run owns: launch, liveness, and complete shutdown.

``gz sim`` is a launcher. Server-only (``-s``) it runs the server itself;
with the GUI it forks ``gz sim server`` and ``gz sim gui``, each in its own
process group. SIGTERM to the launcher stops both, but SIGKILL to it, or
the death of the Python process that started it, leaves them running
(verified with gz sim 8 on Jazzy). Hence:

- launch under ``setpriv --pdeathsig TERM`` (util-linux) where available,
  so the kernel sends the launcher SIGTERM when its parent dies, even by
  SIGKILL. Without ``setpriv`` (non-Linux), a force-killed parent still
  leaves Gazebo running; ``parent_death_prefix`` returns [] there.
- ``stop()`` terminates the launcher and, if it does not exit in time,
  kills it and every descendant found under ``/proc`` before the kill.

Linux delivers the parent-death signal when the *thread* that started the
child exits, so launch from a thread that outlives the child (the main
thread, or a pool worker that waits for it). A parent that dies in the
microseconds before ``setpriv`` has requested the signal is not reported.
"""

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import IO, Any


def parent_death_prefix(
    which: Callable[[str], str | None] = shutil.which, platform: str = sys.platform
) -> list[str]:
    """Command prefix that makes the child receive SIGTERM when its parent
    dies (``setpriv --pdeathsig TERM --``), or [] where unsupported."""
    if not platform.startswith("linux"):
        return []
    setpriv = which("setpriv")
    return [setpriv, "--pdeathsig", "TERM", "--"] if setpriv else []


def descendants(pid: int, proc: Path = Path("/proc")) -> list[int]:
    """Process ids below ``pid`` (children first, then theirs), from
    ``/proc/<pid>/task/*/children``; [] when ``/proc`` is unavailable or
    the process is gone."""
    found: list[int] = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            tasks = list((proc / str(parent) / "task").iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                children = (task / "children").read_text().split()
            except OSError:
                continue
            for child in map(int, children):
                if child not in found:
                    found.append(child)
                    pending.append(child)
    return found


class GazeboProcess:
    """An owned ``gz sim`` (or any launcher) process.

    ``command`` runs with output to ``log`` (None: this terminal); ``prefix`` defaults to
    ``parent_death_prefix()``. ``stop`` is idempotent and never raises for
    a process that has already exited.
    """

    def __init__(
        self,
        command: Sequence[str],
        log: IO[Any] | None,
        env: dict[str, str] | None = None,
        *,
        prefix: Sequence[str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("empty Gazebo command")
        prefix = parent_death_prefix() if prefix is None else list(prefix)
        self.command = [*prefix, *command]
        self._process = subprocess.Popen(
            self.command,
            stdout=log,
            stderr=None if log is None else subprocess.STDOUT,
            env=env,
        )
        self.killed: list[int] = []

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def poll(self) -> int | None:
        """Exit status, or None while running."""
        return self._process.poll()

    def wait(self, timeout_s: float | None = None) -> int:
        """Wait for exit (raises ``subprocess.TimeoutExpired`` on timeout)."""
        return self._process.wait(timeout=timeout_s)

    def stop(self, timeout_s: float = 10.0) -> int:
        """SIGTERM the launcher; after ``timeout_s`` wall seconds SIGKILL it
        and all its descendants. Returns the launcher's exit status;
        ``killed`` lists force-killed process ids."""
        if self._process.poll() is not None:
            return self._process.returncode
        tree = descendants(self._process.pid)
        self._process.terminate()
        try:
            return self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass
        # Children may have changed while it shut down: collect again.
        tree += [p for p in descendants(self._process.pid) if p not in tree]
        self._process.kill()
        self.killed.append(self._process.pid)
        for pid in tree:
            try:
                os.kill(pid, signal.SIGKILL)
                self.killed.append(pid)
            except ProcessLookupError:
                pass
        return self._process.wait()
