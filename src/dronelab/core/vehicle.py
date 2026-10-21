"""Shared properties of the built-in Gazebo quad; JSON configuration is loaded by the Gazebo adapter."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DroneProperties:
    body_mass: float = 1.5
    rotor_mass: float = 0.01
    body_inertia: tuple = (0.03, 0.03, 0.05)
    rotor_inertia: float = 0.00001
    # x, y, rotation direction (ccw = +1), in actuator index order.
    rotors: tuple = ((0.2, 0.2, 1), (-0.2, -0.2, 1), (0.2, -0.2, -1), (-0.2, 0.2, -1))
    thrust_coefficient: float = 8.55e-6
    moment_ratio: float = 0.016
    max_rotor_speed: float = 1000.0
    thrust_limit_n: float | None = None
    # Rotor plane above the body origin and propeller diameter, in metres.
    rotor_height: float = 0.08
    propeller_diameter: float = 0.25
    # Optional full geometry (general layouts): per-rotor body position (m)
    # and unit thrust axis. None: (x, y, rotor_height) and +z for every rotor.
    rotor_positions: tuple | None = None
    rotor_axes: tuple | None = None

    @property
    def positions(self) -> tuple:
        """Per-rotor (x, y, z) in body axes, metres."""
        if self.rotor_positions is not None:
            return self.rotor_positions
        return tuple((x, y, self.rotor_height) for x, y, _ in self.rotors)

    @property
    def axes(self) -> tuple:
        """Per-rotor unit thrust axis in body axes."""
        if self.rotor_axes is not None:
            return self.rotor_axes
        return tuple((0.0, 0.0, 1.0) for _ in self.rotors)

    @property
    def lift_rotors(self) -> tuple:
        """Indices of rotors whose thrust axis is vertical (hover rotors)."""
        return tuple(i for i, axis in enumerate(self.axes) if axis[2] > 0.99)

    @property
    def mass(self):
        return self.body_mass + len(self.rotors) * self.rotor_mass

    @property
    def max_rotor_thrust(self):
        physical = self.thrust_coefficient * self.max_rotor_speed**2
        return (
            min(physical, self.thrust_limit_n)
            if self.thrust_limit_n is not None
            else physical
        )


DEFAULT_DRONE = DroneProperties()
