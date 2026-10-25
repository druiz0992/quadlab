"""Focused Tk keyboard window for driving wheeled robots.

Reports held keys (``poll_keys``) and one-shot presses (``drain_presses``);
``control/wheeled_keyboard.py`` turns them into driver input. No global
keyboard hooks: only keys pressed while this window has focus count.
"""

# Keys held for continuous input, and keys that act once per press.
HELD_KEYS = frozenset({"Up", "Down", "Left", "Right", "space"})
PRESS_KEYS = frozenset({"1", "2", "3", "h"})

HELP = {
    "skid": "↑/↓ Forward/back · ←/→ Turn on the spot\nSpace Brake · Esc Finish",
    "ackermann": "↑ Throttle · ↓ Brake (stopped: reverse)\n←/→ Steer · Space Brake · Esc Finish",
    "four_wheel": (
        "↑ Throttle · ↓ Brake (stopped: reverse)\n←/→ Steer · Space Brake · Esc Finish\n"
        "1 Front steer · 2 Coordinated · 3 Crab\nH Raise/lower implement"
    ),
}


class DrivingKeyboardWindow:
    """Small always-on-top window; raises ValueError without a display."""

    def __init__(
        self,
        title: str,
        steering_layout: str,
        held_keys: frozenset[str] = HELD_KEYS,
        help_text: str | None = None,
    ) -> None:
        import tkinter as tk

        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise ValueError(
                f"Keyboard mode needs a desktop display and Tk: {error}"
            ) from error
        self.root.title(f"RobotLab — {title}")
        self.root.geometry("360x220")
        self.root.attributes("-topmost", True)
        self.held_keys = held_keys
        self.keys: set[str] = set()
        self.presses: list[str] = []
        self.releases: dict[str, str] = {}
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        tk.Label(self.root, text=f"RobotLab {title}", font=("sans", 12, "bold")).pack(
            pady=4
        )
        tk.Label(
            self.root,
            text=help_text or HELP.get(steering_layout, HELP["ackermann"]),
            justify="center",
        ).pack()
        self.status = tk.StringVar(value="Click here to drive")
        tk.Label(self.root, textvariable=self.status, fg="#b45309").pack(pady=6)
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.root.bind("<FocusOut>", self._clear)
        self.root.bind("<Button-1>", lambda event: self.root.focus_set())
        self.root.after_idle(self.root.focus_set)

    @staticmethod
    def _key(event) -> str:
        return event.keysym if len(event.keysym) > 1 else event.keysym.lower()

    def _press(self, event) -> None:
        key = self._key(event)
        if key == "Escape":
            self.close()
            return
        if key in self.releases:
            self.root.after_cancel(self.releases.pop(key))
        if key in self.held_keys:
            self.keys.add(key)
        elif key in PRESS_KEYS and key not in self.keys:
            self.presses.append(key)
            self.keys.add(key)

    def _release(self, event) -> None:
        key = self._key(event)
        if key in self.releases:
            self.root.after_cancel(self.releases.pop(key))
        # X11 key repeat can emit release/press pairs; defer until idle.
        self.releases[key] = self.root.after_idle(lambda: self._finish_release(key))

    def _finish_release(self, key: str) -> None:
        self.releases.pop(key, None)
        self.keys.discard(key)

    def _clear(self, event=None) -> None:
        for callback in self.releases.values():
            self.root.after_cancel(callback)
        self.releases.clear()
        self.keys.clear()

    def poll_keys(self) -> frozenset[str]:
        """Service the window; return the held driving keys."""
        if self.closed:
            return frozenset()
        try:
            self.root.update()
        except Exception:
            self.closed = True
            return frozenset()
        if self.closed:  # Esc or the close button, handled inside update()
            return frozenset()
        focused = self.root.focus_get() is not None
        self.status.set("Driving" if focused else "Click here to drive")
        return frozenset(self.keys & self.held_keys)

    def drain_presses(self) -> list[str]:
        presses, self.presses = self.presses, []
        return presses

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.root.destroy()
            except Exception:
                pass
