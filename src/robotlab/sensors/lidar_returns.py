"""Per-beam lidar return physics: reflectivity, weather, and beam spread.

These effects need the whole scan grid, including beams that hit nothing, so
they run before ``LidarEffects`` turns surviving returns into points.

- **Detection** (radar equation for an extended Lambertian target): received
  power ∝ ρ · exp(−2αr) / r², with ρ the surface reflectivity and α the
  atmospheric extinction (two-way). A return is detected when this reaches
  the threshold set by a datasheet point, "``reference_range_m`` at
  ``reference_reflectivity``": ρ_ref / R_ref². Reflectivity comes from the
  scan's intensities: the world builder writes ``laser_retro = 100 · ρ`` and
  Gazebo's gpu_lidar reports it per beam (0 when unset, which maps to
  ``default_reflectivity``).
- **Extinction** from meteorological visibility V (Koschmieder, 2 % contrast)
  scaled to 905 nm with the Kim et al. (2001) model:
  α = 3.912 / V · (λ / 550 nm)^(−q).
- **Weather clutter** (rain, snow, dust, fog backscatter): along each beam,
  false returns occur as a Poisson process of ``lidar_clutter_per_m`` within
  ``lidar_clutter_max_range_m``. The first one (exponentially distributed
  distance) replaces the beam's return if it comes before the real target;
  beams with no target can return clutter too.
- **Beam spread** (mixed pixels): a beam's footprint has full-angle
  divergence δ. Where neighbouring beams (angular step Δ) see a depth jump,
  the edge lies uniformly between their centres; a beam whose footprint
  covers it receives a fraction w of its energy from its own surface and
  1 − w from the other. It reports the power-weighted mean range, a "ghost"
  point between foreground and background. Energy falling on empty space is
  lost, which can drop the return. One edge per beam; scans do not wrap.

Inputs are the rendered scan; all randomness uses the caller's seeded RNG.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

LIDAR_WAVELENGTH_NM = 905.0
# Gazebo laser_retro value per unit reflectivity (written by the world builder).
RETRO_PER_REFLECTIVITY = 100.0


@dataclass(frozen=True)
class LidarScan:
    """A decoded scan as a rows × columns grid (rows = vertical channels).

    ``ranges_m`` is inf where no return was rendered; ``reflectivity`` is the
    per-beam surface reflectivity (dimensionless, 1 = white Lambertian).
    """

    ranges_m: NDArray[np.float64]
    reflectivity: NDArray[np.float64]
    azimuth_rad: NDArray[np.float64]
    elevation_rad: NDArray[np.float64]
    azimuth_step_rad: float
    elevation_step_rad: float
    range_min_m: float
    range_max_m: float

    def in_range(self, ranges_m: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (
            np.isfinite(ranges_m)
            & (ranges_m >= self.range_min_m)
            & (ranges_m < self.range_max_m)
        )


def decode_scan(message: Any, default_reflectivity: float = 1.0) -> LidarScan:
    """Grid view of a Gazebo LaserScan (row-major, like ``lidar_points``)."""
    ranges = np.asarray(message.ranges, dtype=float)
    rows = max(1, getattr(message, "vertical_count", 1) or 1)
    columns = len(ranges) // rows
    intensities = np.asarray(getattr(message, "intensities", ()) or (), float)
    if intensities.size != ranges.size:
        intensities = np.zeros_like(ranges)
    reflectivity = np.where(
        intensities > 0, intensities / RETRO_PER_REFLECTIVITY, default_reflectivity
    )
    elevation_step = getattr(message, "vertical_angle_step", 0.0) if rows > 1 else 0.0
    return LidarScan(
        ranges_m=ranges.reshape(rows, columns),
        reflectivity=reflectivity.reshape(rows, columns),
        azimuth_rad=message.angle_min + np.arange(columns) * message.angle_step,
        elevation_rad=getattr(message, "vertical_angle_min", 0.0)
        + np.arange(rows) * elevation_step,
        azimuth_step_rad=abs(message.angle_step),
        elevation_step_rad=abs(elevation_step),
        range_min_m=message.range_min,
        range_max_m=message.range_max,
    )


def scan_points(
    scan: LidarScan, ranges_m: NDArray[np.float64], keep: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sensor-frame points of the kept beams plus each one's sweep fraction."""
    beam = np.flatnonzero(keep)
    return beam_points(scan, beam, ranges_m.reshape(-1)[beam])


def beam_points(
    scan: LidarScan, beam: NDArray[np.int64], range_m: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Points for returns given by flat beam index and range, plus sweep fraction."""
    columns = scan.ranges_m.shape[1]
    row, column = np.divmod(np.asarray(beam, int), columns)
    az, el = scan.azimuth_rad[column], scan.elevation_rad[row]
    r = np.asarray(range_m, float)
    points = np.column_stack(
        (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el))
    )
    return points.reshape(-1, 3), column / columns


def extinction_per_m(
    visibility_m: float | None, wavelength_nm: float = LIDAR_WAVELENGTH_NM
) -> float:
    """Atmospheric extinction α (1/m) at the lidar wavelength; 0 for clear air."""
    if visibility_m is None:
        return 0.0
    v_km = visibility_m / 1000
    if v_km > 50:
        q = 1.6
    elif v_km > 6:
        q = 1.3
    elif v_km > 1:
        q = 0.16 * v_km + 0.34
    elif v_km > 0.5:
        q = v_km - 0.5
    else:
        q = 0.0
    return 3.912 / visibility_m * (wavelength_nm / 550.0) ** (-q)


def detected(
    ranges_m: NDArray[np.float64],
    reflectivity: NDArray[np.float64],
    alpha_per_m: float,
    detection: Mapping[str, Any],
) -> NDArray[np.bool_]:
    """Whether each return's power reaches the datasheet threshold."""
    threshold = (
        detection["reference_reflectivity"] / detection["reference_range_m"] ** 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        power = reflectivity * np.exp(-2 * alpha_per_m * ranges_m) / ranges_m**2
    return np.isfinite(ranges_m) & (power >= threshold)


def weather_clutter(
    ranges_m: NDArray[np.float64],
    rate_per_m: float,
    max_range_m: float,
    range_min_m: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Replace returns by earlier backscatter; returns (ranges, clutter mask)."""
    first = rng.exponential(1 / rate_per_m, size=ranges_m.shape)
    clutter = (first >= range_min_m) & (first < np.minimum(ranges_m, max_range_m))
    return np.where(clutter, first, ranges_m), clutter


def edge_split(
    ranges_m: NDArray[np.float64],
    reflectivity: NDArray[np.float64],
    step_rad: float,
    divergence_rad: float,
    min_jump_m: float,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    """Footprint split across depth edges along the last axis.

    Returns (own share w, other surface range, other reflectivity, mixed
    mask); unaffected beams have w = 1 and other range inf. A beam touched by
    edges on both sides keeps the later (right-hand) one.
    """
    share = np.ones(ranges_m.shape)
    other_range = np.full(ranges_m.shape, np.inf)
    other_rho = np.zeros(ranges_m.shape)
    mixed = np.zeros(ranges_m.shape, bool)
    if ranges_m.shape[-1] < 2 or step_rad <= 0:
        return share, other_range, other_rho, mixed
    left, right = ranges_m[..., :-1], ranges_m[..., 1:]
    with np.errstate(invalid="ignore"):
        jump = np.isfinite(left) != np.isfinite(right)
        jump |= np.isfinite(left) & (np.abs(left - right) > min_jump_m)
    # Edge position measured from the left beam centre, uniform in (0, Δ).
    edge = rng.uniform(0, step_rad, size=left.shape)
    first, second = np.s_[..., :-1], np.s_[..., 1:]
    # Each edge can reach the beam on its left and the one on its right.
    for own, other, gap in ((first, second, edge), (second, first, step_rad - edge)):
        # Footprint [−δ/2, δ/2] around the centre: share on the own side.
        own_share = 0.5 + gap / divergence_rad
        hit = jump & (own_share < 1)
        share[own] = np.where(hit, np.clip(own_share, 0.5, 1), share[own])
        other_range[own] = np.where(hit, ranges_m[other], other_range[own])
        other_rho[own] = np.where(hit, reflectivity[other], other_rho[own])
        mixed[own] |= hit
    return share, other_range, other_rho, mixed


def mix_edge_returns(
    ranges_m: NDArray[np.float64],
    reflectivity: NDArray[np.float64],
    step_rad: float,
    divergence_rad: float,
    min_jump_m: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Mixed-pixel returns across depth edges along the last axis.

    Returns (ranges, effective reflectivity, mixed mask). The effective
    reflectivity keeps ``ρ_eff / r²`` equal to the beam's total received
    power so detection can be applied afterwards.
    """
    share, other_r, other_rho, mixed = edge_split(
        ranges_m, reflectivity, step_rad, divergence_rad, min_jump_m, rng
    )
    r, rho = _mix(ranges_m, reflectivity, other_r, other_rho, share)
    return np.where(mixed, r, ranges_m), np.where(mixed, rho, reflectivity), mixed


RETURN_MODES = ("first", "last", "strongest", "dual")


@dataclass(frozen=True)
class BeamReturns:
    """Reported returns: flat beam index, range (m), reported reflectivity.

    Reported reflectivity is the range-normalized return strength P·r² =
    share · ρ · exp(−2αr), so edges and fog read darker, as on real lidars.
    """

    beam: NDArray[np.int64]
    range_m: NDArray[np.float64]
    reflectivity: NDArray[np.float64]
    kind: NDArray[np.int64]  # 0 own surface, 1 other surface (edge), 2 clutter


def select_returns(
    ranges: NDArray[np.float64],
    strength: NDArray[np.float64],
    valid: NDArray[np.bool_],
    mode: str,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Per-beam choice among K candidates (arrays N×K): (beam, candidate) pairs."""
    beams = np.nonzero(valid.any(axis=1))[0]
    if len(beams) == 0:
        empty = np.zeros(0, int)
        return [(empty, empty)]
    r = np.where(valid, ranges, np.inf)[beams]
    p = np.where(valid, strength, -np.inf)[beams]
    nearest = np.argmin(r, axis=1)
    farthest = np.argmax(np.where(np.isfinite(r), r, -np.inf), axis=1)
    strongest = np.argmax(p, axis=1)
    if mode == "first":
        return [(beams, nearest)]
    if mode == "last":
        return [(beams, farthest)]
    if mode == "strongest":
        return [(beams, strongest)]
    second = strongest != nearest  # dual: first plus a distinct strongest
    return [(beams, nearest), (beams[second], strongest[second])]


def _mix(r_own, rho_own, r_other, rho_other, share):
    """Power-weighted range and effective reflectivity of a split footprint."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p_own = np.where(np.isfinite(r_own), share * rho_own / r_own**2, 0.0)
        p_other = np.where(
            np.isfinite(r_other), (1 - share) * rho_other / r_other**2, 0.0
        )
        total = p_own + p_other
        r_own_f = np.where(np.isfinite(r_own), r_own, 0.0)
        r_other_f = np.where(np.isfinite(r_other), r_other, 0.0)
        r = (p_own * r_own_f + p_other * r_other_f) / total
        r = np.where(total > 0, r, np.inf)
        return r, np.where(total > 0, total * r**2, 0.0)
