"""Shared error and signal handling for the command-line entry points."""

import argparse
from collections.abc import Callable
import signal
import sys
from typing import TypeVar

from ..experiments.interrupts import sigterm_as_interrupt

T = TypeVar("T")


def run_reporting_errors(parser: argparse.ArgumentParser, action: Callable[[], T]) -> T:
    """Run ``action`` with SIGTERM treated like Ctrl-C (see
    experiments/interrupts.py) and map failures to exit statuses:

    - ValueError (invalid options or configuration): usage message, 2;
    - RuntimeError (a run that failed): the message without usage, 1;
    - an interruption that escapes ``action``: 130 (Ctrl-C) or 143 (SIGTERM);
    - SIGTERM absorbed by ``action`` (a mission saves its results and
      returns): 143 after it returns.
    """
    with sigterm_as_interrupt() as termination:
        try:
            result = action()
        except ValueError as error:
            parser.error(str(error))
        except RuntimeError as error:
            print(f"{parser.prog}: error: {error}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"{parser.prog}: interrupted", file=sys.stderr)
            sys.exit(_interrupted_status(termination.received))
    if termination.received:
        sys.exit(_interrupted_status(True))
    return result


def _interrupted_status(terminated: bool) -> int:
    """Shell convention: 128 + the signal number."""
    return 128 + (signal.SIGTERM if terminated else signal.SIGINT)
