from types import SimpleNamespace

from dronelab.control.manual import FlightCommand
from dronelab.visualization.keyboard import KeyboardWindow


class FakeRoot:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0
        self.destroyed = False

    def after_idle(self, callback):
        self.next_id += 1
        self.callbacks[self.next_id] = callback
        return self.next_id

    def after_cancel(self, token):
        self.callbacks.pop(token, None)

    def destroy(self):
        self.destroyed = True


def window():
    w = KeyboardWindow.__new__(KeyboardWindow)
    w.root = FakeRoot()
    w.keys = set()
    w.releases = {}
    w.closed = False
    w.effects = []
    w.effect_state = {}
    w.pending_toggles = []
    return w


def event(key):
    return SimpleNamespace(keysym=key)


def test_repeat_release_and_focus_loss():
    w = window()
    w._press(event("Up"))
    w._release(event("Up"))
    w._press(event("Up"))  # X11 auto-repeat pair must not stop movement
    assert FlightCommand.from_keys(w.keys).forward_mps == 1
    assert not w.root.callbacks
    w._release(event("Up"))
    list(w.root.callbacks.values())[0]()
    assert not w.keys
    w._press(event("Q"))
    assert FlightCommand.from_keys(w.keys).climb_mps == 0.6
    w._clear()
    assert not w.keys
    assert not w.releases


def test_escape_stops_and_closes_once():
    w = window()
    w._press(event("a"))
    w._press(event("Escape"))
    w.close()
    assert w.closed and w.root.destroyed and not w.keys


def test_descent_and_clockwise_turn():
    w = window()
    w._press(event("A"))
    w._press(event("Right"))
    assert FlightCommand.from_keys(w.keys) == FlightCommand(0, -0.7, -0.6)
    w._press(event("Q"))
    w._press(event("Left"))
    assert FlightCommand.from_keys(w.keys) == FlightCommand()


def test_number_keys_toggle_available_effects_only():
    w = window()
    w.effects = [("1", "motor_stop", "Motor stop", True), ("3", "wind", "Wind", False)]
    w.effect_state = {"motor_stop": False, "wind": False}
    w._press(event("1"))
    w._press(event("3"))  # unavailable: ignored
    w._press(event("1"))
    assert w.drain_toggles() == [("motor_stop", True), ("motor_stop", False)]
    assert w.drain_toggles() == []
    assert w.keys == set()  # number keys never become flight keys


def test_action_keys_queue_one_shot_actions():
    w = window()
    w.actions = [
        ("9", "battery_drain", "Battery −", True),
        ("0", "battery_refill", "Battery full", False),
    ]
    w.pending_actions = []
    w._press(event("9"))
    w._press(event("9"))
    w._press(event("0"))  # unavailable without a battery
    assert w.drain_actions() == ["battery_drain", "battery_drain"]
    assert w.drain_actions() == []


def test_winch_keys():
    w = window()
    w._press(event("w"))
    assert w.winch_rate() == 1
    w._press(event("s"))
    assert w.winch_rate() == 0  # both held cancel
