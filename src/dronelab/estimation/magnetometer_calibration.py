"""Magnetometer calibration: hard iron, soft iron, and current/throttle compensation.

A distorted magnetometer measures (sensor axes, µT)

    m = S b + h + k_I I + k_T T + noise

for the local Earth field b (fixed magnitude |b| = B), soft iron S, hard iron
h, and interference proportional to battery current I or total thrust T.
Calibration recovers a correction C ≈ S⁻¹ (symmetric part; a pure rotation
is indistinguishable from the sensor mount) and an offset o ≈ h, so that

    b̂ = C (m − o − k_I I − k_T T),    |b̂| ≈ B.

**Ellipsoid fit.** Rotating the sensor through many orientations at constant
interference makes m trace an ellipsoid (m − o)ᵀ M (m − o) = 1. A general
quadric aᵀv = 0 with v = [x², y², z², 2yz, 2xz, 2xy, 2x, 2y, 2z, 1] is fitted
by total least squares (the right singular vector of the smallest singular
value). Writing the quadric as xᵀAx + 2nᵀx + d = 0 gives the centre
o = −A⁻¹n and M = A / (oᵀAo − d); then C = B · M^{1/2} (symmetric square
root), scaling corrected fields to the known local field strength B (from a
geomagnetic model for the site). Needs well-spread orientations; the fit
reports its residual and the condition of A.

**Interference fit** ("compassmot"). Held at one orientation while the
throttle is swept, m changes only through the interference, so a per-axis
linear regression of m on I (or T) gives k. If the orientation dance ran at
a nonzero current I₀, its fitted centre includes k I₀, so the absolute
offset is o − k I₀.

Transport-free; arrays in µT.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EllipsoidFit:
    """Fitted offset (µT), correction matrix, and quality figures."""

    offset_ut: NDArray[np.float64]
    correction: NDArray[np.float64]
    residual_rms_ut: float  # RMS of |C(m − o)| − B over the fit samples
    condition: float  # condition number of the quadric matrix A


def _symmetric_sqrt(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0):
        raise ValueError("Ellipsoid fit is not positive definite; spread orientations")
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def fit_ellipsoid(samples_ut: ArrayLike, field_strength_ut: float) -> EllipsoidFit:
    """Hard-iron offset and soft-iron correction from well-spread samples."""
    m = np.asarray(samples_ut, float).reshape(-1, 3)
    if len(m) < 10:
        raise ValueError("Ellipsoid fit needs at least 10 samples")
    # Normalize for conditioning; undo afterwards.
    shift = m.mean(axis=0)
    scale = np.abs(m - shift).max()
    x, y, z = ((m - shift) / scale).T
    design = np.column_stack(
        (
            x * x,
            y * y,
            z * z,
            2 * y * z,
            2 * x * z,
            2 * x * y,
            2 * x,
            2 * y,
            2 * z,
            np.ones_like(x),
        )
    )
    a, b, c, f, g, h, p, q, r, d = np.linalg.svd(design, full_matrices=False)[2][-1]
    quadric = np.array([[a, h, g], [h, b, f], [g, f, c]])
    centre = -np.linalg.solve(quadric, [p, q, r])
    shape = quadric / (centre @ quadric @ centre - d)
    if np.linalg.eigvalsh(shape).max() <= 0:
        shape = -shape  # the SVD sign is arbitrary
    correction = field_strength_ut * _symmetric_sqrt(shape) / scale
    offset = shift + centre * scale
    corrected = (m - offset) @ correction.T
    residual = np.linalg.norm(corrected, axis=1) - field_strength_ut
    return EllipsoidFit(
        offset,
        correction,
        float(np.sqrt(np.mean(residual**2))),
        float(np.linalg.cond(quadric)),
    )


def fit_interference(
    raw_ut: ArrayLike, regressor: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-axis slope and intercept of raw field against current or thrust."""
    m = np.asarray(raw_ut, float).reshape(-1, 3)
    u = np.asarray(regressor, float).reshape(-1)
    if len(u) != len(m) or np.ptp(u) <= 0:
        raise ValueError("Interference fit needs a varying regressor per sample")
    design = np.column_stack((u, np.ones_like(u)))
    (slope, intercept), *_ = np.linalg.lstsq(design, m, rcond=None)
    return slope, intercept


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


def heading_error_deg(estimated: ArrayLike, true: ArrayLike, to_world: Any) -> float:
    """Horizontal-direction error between two sensor-axis fields after
    rotating both into world axes with ``to_world`` (a scipy Rotation)."""
    e = to_world.apply(np.asarray(estimated, float))
    t = to_world.apply(np.asarray(true, float))
    angle = math.atan2(e[1], e[0]) - math.atan2(t[1], t[0])
    return math.degrees((angle + math.pi) % (2 * math.pi) - math.pi)
