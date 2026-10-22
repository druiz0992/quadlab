"""Focused Tk keyboard window; no global keyboard hooks or control algorithms."""

from ..control.manual import FlightCommand

# Display names for keys whose Tk keysym is a word.
KEY_LABELS = {"bracketleft": "[", "bracketright": "]"}


class KeyboardWindow:
    def __init__(self):
        import tkinter as tk

        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise ValueError(
                f"Keyboard mode needs a desktop display and Tk: {error}"
            ) from error
        self.root.title("RobotLab — keyboard flight")
        self.root.geometry("340x380")
        self.root.attributes("-topmost", True)
        self.keys = set()
        self.pilot_rates = None
        self.releases = {}
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        tk.Label(
            self.root, text="RobotLab keyboard flight", font=("sans", 12, "bold")
        ).pack(pady=4)
        tk.Label(
            self.root,
            text="↑/↓ Forward/back · ←/→ Turn\nQ/A Up/down · Esc Finish\nRelease keys to brake and hover",
            justify="center",
        ).pack()
        self.input_status = tk.StringVar(value="Click here to fly")
        tk.Label(self.root, textvariable=self.input_status, fg="#b45309").pack(pady=4)
        self.status = tk.StringVar(value="Waiting for Gazebo…")
        tk.Label(self.root, textvariable=self.status).pack(pady=4)
        self.effects_text = tk.StringVar(value="")
        tk.Label(
            self.root, textvariable=self.effects_text, justify="left", font=("mono", 9)
        ).pack(pady=4)
        self.effects = []  # (key, name, label, available)
        self.actions = []  # one-shot (key, name, label, available)
        self.pending_actions = []
        self.effect_state = {}
        self.pending_toggles = []
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.root.bind("<FocusOut>", self._clear)
        self.root.bind("<Button-1>", lambda event: self.root.focus_set())
        self.root.after_idle(self.root.focus_set)

    def _press(self, event):
        key = event.keysym if len(event.keysym) > 1 else event.keysym.lower()
        if key == "Escape":
            self.close()
            return
        if key in self.releases:
            self.root.after_cancel(self.releases.pop(key))
        if key in {"Up", "Down", "Left", "Right", "q", "a", "w", "s"}:
            self.keys.add(key)
        self._toggle_effect(key)

    def _release(self, event):
        key = event.keysym if len(event.keysym) > 1 else event.keysym.lower()
        if key in self.releases:
            self.root.after_cancel(self.releases.pop(key))
        # X11 key repeat can emit release/press pairs; defer until idle.
        self.releases[key] = self.root.after_idle(lambda: self._finish_release(key))

    def _finish_release(self, key):
        self.releases.pop(key, None)
        self.keys.discard(key)

    def _clear(self, event=None):
        for callback in self.releases.values():
            self.root.after_cancel(callback)
        self.releases.clear()
        self.keys.clear()

    def configure_effects(self, effects, state):
        """Show effect switches: effects is [(key, name, label, available)]."""
        self.effects = list(effects)
        self.effect_state = dict(state)
        self._show_effects()

    def configure_actions(self, actions):
        """One-shot keys such as battery refill: [(key, name, label, available)]."""
        self.actions = list(actions)
        self._show_effects()

    def _toggle_effect(self, key):
        for action_key, name, _, available in getattr(self, "actions", []):
            if key == action_key and available:
                self.pending_actions.append(name)
        for effect_key, name, _, available in self.effects:
            if key == effect_key and available:
                self.effect_state[name] = not self.effect_state[name]
                self.pending_toggles.append((name, self.effect_state[name]))
                self._show_effects()

    def _show_effects(self):
        lines = []
        for key, name, label, available in self.effects:
            state = "ON" if self.effect_state[name] else "off"
            lines.append(f"{key} {label:<18} {state if available else 'n/a'}")
        for key, _, label, available in getattr(self, "actions", []):
            key = KEY_LABELS.get(key, key)
            lines.append(f"{key} {label:<18} {'press' if available else 'n/a'}")
        if hasattr(self, "effects_text"):
            self.effects_text.set("\n".join(lines))

    def winch_rate(self):
        """Held winch keys: W lowers (+1), S raises (−1), both or none 0."""
        return ("w" in self.keys) - ("s" in self.keys)

    def drain_actions(self):
        """Return and clear one-shot actions pressed since the last call."""
        actions, self.pending_actions = self.pending_actions, []
        return actions

    def drain_toggles(self):
        """Return and clear effect switches pressed since the last call."""
        toggles, self.pending_toggles = self.pending_toggles, []
        return toggles

    def poll(self):
        if not self.closed:
            self.root.update()
            if not self.closed:
                focused = self.root.focus_displayof() is not None
                text = (
                    "Keys: " + (", ".join(sorted(self.keys)) or "none")
                    if focused
                    else "Click here to enable input"
                )
                # Setting a Tk variable schedules a redraw even when unchanged.
                if text != self.input_status.get():
                    self.input_status.set(text)
        return FlightCommand.from_keys(self.keys, self.pilot_rates)

    def show_position(self, position):
        if not self.closed:
            text = "Position: " + ", ".join(f"{v:.1f}" for v in position) + " m"
            if text != self.status.get():
                self.status.set(text)

    def close(self):
        if not self.closed:
            self._clear()
            self.closed = True
            self.root.destroy()
