"""Applying a magnetometer calibration, as a driver does before data reach
filters.

The parameter set is read from and written to
``sensors.magnetometer.calibration``; the fits that produce it are
estimation (estimation/magnetometer_calibration.py).

b̂ = C (m − o − k_I I − k_T T)

with C the soft-iron correction, o the hard-iron offset, and k_I, k_T the
battery-current and thrust interference slopes; ``plausible`` checks |b̂|
against the local field strength.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MagnetometerCalibration:
    """Calibration result as used by drivers and filters (µT, sensor axes).

    ``field_strength_ut``, ``inclination_deg`` (positive down), and
    ``declination_deg`` (east of true north) describe the local Earth field;
    ``field_tolerance_ut`` is how far |b̂| may stray before a reading should
    be distrusted; ``heading_std_deg`` is the residual heading uncertainty.
    """

    hard_iron_offset_ut: NDArray[np.float64]
    soft_iron_correction: NDArray[np.float64]
    current_compensation_ut_per_a: NDArray[np.float64]
    thrust_compensation_ut_per_n: NDArray[np.float64]
    field_strength_ut: float
    inclination_deg: float
    declination_deg: float
    field_tolerance_ut: float
    heading_std_deg: float

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "MagnetometerCalibration":
        return cls(
            np.asarray(section["hard_iron_offset_ut"], float),
            np.asarray(section["soft_iron_correction"], float),
            np.asarray(section["current_compensation_ut_per_a"], float),
            np.asarray(section["thrust_compensation_ut_per_n"], float),
            float(section["field_strength_ut"]),
            float(section["inclination_deg"]),
            float(section["declination_deg"]),
            float(section["field_tolerance_ut"]),
            float(section["heading_std_deg"]),
        )

    def to_config(self) -> dict[str, Any]:
        return dict(
            enabled=True,
            hard_iron_offset_ut=self.hard_iron_offset_ut.tolist(),
            soft_iron_correction=self.soft_iron_correction.tolist(),
            current_compensation_ut_per_a=self.current_compensation_ut_per_a.tolist(),
            thrust_compensation_ut_per_n=self.thrust_compensation_ut_per_n.tolist(),
            field_strength_ut=self.field_strength_ut,
            inclination_deg=self.inclination_deg,
            declination_deg=self.declination_deg,
            field_tolerance_ut=self.field_tolerance_ut,
            heading_std_deg=self.heading_std_deg,
        )

    def apply(
        self, raw_ut: ArrayLike, current_a: float = 0.0, thrust_n: float = 0.0
    ) -> NDArray[np.float64]:
        """Corrected field b̂ (µT, sensor axes)."""
        compensated = (
            np.asarray(raw_ut, float)
            - self.hard_iron_offset_ut
            - self.current_compensation_ut_per_a * current_a
            - self.thrust_compensation_ut_per_n * thrust_n
        )
        return self.soft_iron_correction @ compensated

    def plausible(self, corrected_ut: ArrayLike) -> bool:
        """Whether |b̂| is within tolerance of the local field strength."""
        magnitude = float(np.linalg.norm(corrected_ut))
        return abs(magnitude - self.field_strength_ut) <= self.field_tolerance_ut
