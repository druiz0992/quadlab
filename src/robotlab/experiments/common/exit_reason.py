"""The exit reason a run reports, shared by every robot's summary."""


def final_exit_reason(completed: bool, exit_reason: str) -> str:
    """``exit_reason`` as reported: a run that neither completed nor was
    ended deliberately ended on an error."""
    return exit_reason if completed or exit_reason != "duration" else "error"
