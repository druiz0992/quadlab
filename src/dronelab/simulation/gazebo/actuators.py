"""Per-rotor thrust boundary; Gazebo still models motor response and forces."""

import numpy as np

from .properties import DEFAULT_DRONE


def thrust_to_motor_speeds(thrust_n, properties=DEFAULT_DRONE):
    thrust = np.asarray(thrust_n, dtype=float)
    if (
        thrust.shape != (len(properties.rotors),)
        or not np.isfinite(thrust).all()
        or np.any(thrust < 0)
        or np.any(thrust > properties.max_rotor_thrust)
    ):
        raise ValueError(
            "Thrust must have one finite, in-range nonnegative value per rotor"
        )
    return np.sqrt(thrust / properties.thrust_coefficient)


class RotorThrustControl:
    """Publish thrust requests as motor-speed targets.

    An optional enabled ``Powertrain`` applies battery speed limits and ground
    effect to the published speeds (see simulation/powertrain.py).
    """

    def __init__(self, node, properties=DEFAULT_DRONE, powertrain=None, esc=None):
        from gz.msgs10.actuators_pb2 import Actuators

        self.properties = properties
        self.powertrain = powertrain
        # Optional EscModel (simulation/esc.py): resolution and update rate.
        self.esc = esc
        self.message_type = Actuators
        self.publisher = node.advertise("/drone/command/motor_speed", Actuators)

    def send(self, thrust_n):
        speeds = thrust_to_motor_speeds(thrust_n, self.properties)
        if self.powertrain is not None:
            speeds = self.powertrain.motor_speeds(np.asarray(thrust_n, dtype=float))
        if self.esc is not None and self.esc.enabled:
            accepted = self.esc.accept(speeds)
            if accepted is None:
                return speeds  # held: motors keep the previous target
            speeds = accepted
        message = self.message_type()
        message.velocity.extend(map(float, speeds))
        if not self.publisher.publish(message):
            raise RuntimeError("Unable to publish rotor command")
        return speeds

    def stop(self):
        self.send(np.zeros(len(self.properties.rotors)))
