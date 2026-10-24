"""Trajectory drift metrics: absolute trajectory error and relative pose error.

Following Sturm et al., "A benchmark for the evaluation of RGB-D SLAM
systems" (IROS 2012) and Zhang & Scaramuzza, "A tutorial on quantitative
trajectory evaluation for visual(-inertial) odometry" (IROS 2018):

- **ATE** (absolute trajectory error): position error after aligning the
  estimated trajectory to truth with one rigid transform. ``none`` is the
  raw error, right for estimators referenced to the world (GPS-aided);
  ``yaw`` (rotation about world z plus translation, 4 DoF) is the
  standard for visual- or lidar-inertial odometry, whose global position
  and yaw are unobservable; ``se3`` (6 DoF) and ``sim3`` (plus scale, for
  monocular vision) use Umeyama's closed form (IEEE PAMI 1991). With
  attitudes, the rotation ATE is the angle of ``R_trueᵀ (A R_est)``.
- **RPE** (relative pose error): for every start sample i and the first
  sample j a segment later (in time, or in distance travelled along the
  true path), compare the relative motion: translation error
  ``|R_est,iᵀ (p_est,j − p_est,i) − R_true,iᵀ (p_true,j − p_true,i)|`` and
  rotation error ``angle((R_true,iᵀ R_true,j)ᵀ (R_est,iᵀ R_est,j))``.
  Distance segments also report drift as a percentage of the segment.
  RPE needs positions and attitudes; it is independent of alignment.

Inputs are matched samples (same timestamps), world ENU, metres; attitudes
are body-to-world quaternions (xyzw). Functions raise ``ValueError`` for
mismatched or too-short inputs and unknown modes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .statistics import error_statistics

ALIGNMENTS = ("none", "yaw", "se3", "sim3")


@dataclass(frozen=True)
class Alignment:
    """``p_aligned = scale · rotation.apply(p) + translation_m``."""

    rotation: Rotation
    translation_m: NDArray[np.float64]
    scale: float

    def apply(self, positions_m: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.scale * self.rotation.apply(positions_m) + self.translation_m

    def summary(self) -> dict[str, Any]:
        return dict(
            rotation_deg=float(np.degrees(self.rotation.magnitude())),
            yaw_deg=float(np.degrees(self.rotation.as_euler("ZYX")[0])),
            translation_m=self.translation_m.tolist(),
            scale=self.scale,
        )


def _positions(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite N×3 array")
    return array


def align(estimated_m: ArrayLike, true_m: ArrayLike, mode: str = "yaw") -> Alignment:
    """Least-squares alignment of estimated onto true positions (module docs)."""
    est, true = _positions(estimated_m, "estimated"), _positions(true_m, "true")
    if est.shape != true.shape:
        raise ValueError("estimated and true positions must match in shape")
    if mode not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {mode!r}")
    if mode == "none":
        return Alignment(Rotation.identity(), np.zeros(3), 1.0)
    minimum = 2 if mode == "yaw" else 3
    if len(est) < minimum:
        raise ValueError(f"{mode} alignment needs at least {minimum} samples")
    est_mean, true_mean = est.mean(axis=0), true.mean(axis=0)
    a, b = est - est_mean, true - true_mean
    if mode == "yaw":
        # argmin_θ Σ |Rz(θ) a_i − b_i|² over the horizontal components.
        angle = np.arctan2(
            np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
            np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]),
        )
        rotation, scale = Rotation.from_euler("z", angle), 1.0
    else:
        rotation, scale = _umeyama(a, b, with_scale=mode == "sim3")
    return Alignment(rotation, true_mean - scale * rotation.apply(est_mean), scale)


def _umeyama(
    a: NDArray[np.float64], b: NDArray[np.float64], with_scale: bool
) -> tuple[Rotation, float]:
    """Rotation (and scale) minimizing Σ |s R a_i − b_i|² (centred inputs)."""
    u, singular, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0  # a proper rotation, not a reflection
    rotation = u @ sign @ vt
    variance = float(np.mean(np.sum(a**2, axis=1)))
    scale = float(np.trace(np.diag(singular) @ sign) / variance) if with_scale else 1.0
    return Rotation.from_matrix(rotation), scale


def absolute_trajectory_error(
    estimated_m: ArrayLike,
    true_m: ArrayLike,
    mode: str = "yaw",
    estimated_xyzw: ArrayLike | None = None,
    true_xyzw: ArrayLike | None = None,
) -> dict[str, Any]:
    """ATE statistics after ``mode`` alignment (see module docs)."""
    est, true = _positions(estimated_m, "estimated"), _positions(true_m, "true")
    alignment = align(est, true, mode)
    errors = np.linalg.norm(alignment.apply(est) - true, axis=1)
    result = dict(
        alignment=mode,
        position_m=error_statistics(errors),
        transform=alignment.summary(),
    )
    if estimated_xyzw is not None and true_xyzw is not None:
        rotated = alignment.rotation * Rotation.from_quat(estimated_xyzw)
        angles = (Rotation.from_quat(true_xyzw).inv() * rotated).magnitude()
        result["rotation_deg"] = error_statistics(np.degrees(angles))
    return result


def relative_pose_error(
    times_s: ArrayLike,
    estimated_m: ArrayLike,
    estimated_xyzw: ArrayLike,
    true_m: ArrayLike,
    true_xyzw: ArrayLike,
    *,
    segment_s: float | None = None,
    segment_m: float | None = None,
) -> dict[str, Any]:
    """RPE over segments of ``segment_s`` seconds or ``segment_m`` metres of
    true travel (exactly one); ``count`` 0 when no segment fits."""
    if (segment_s is None) == (segment_m is None):
        raise ValueError("give exactly one of segment_s and segment_m")
    length = segment_s if segment_s is not None else segment_m
    if length is None or not length > 0:
        raise ValueError("segment length must be positive")
    times = np.asarray(times_s, float)
    est, true = _positions(estimated_m, "estimated"), _positions(true_m, "true")
    if not len(times) == len(est) == len(true):
        raise ValueError("times, estimates, and truth must match in length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must increase")
    if segment_s is not None:
        axis = times
    else:
        steps = np.linalg.norm(np.diff(true, axis=0), axis=1)
        axis = np.concatenate([[0.0], np.cumsum(steps)])
    starts = np.arange(len(axis))
    # First sample at least one segment later; the relative tolerance keeps
    # a sample exactly one segment away (up to rounding) in the segment.
    ends = np.searchsorted(axis, axis + length * (1 - 1e-9))
    valid = ends < len(axis)
    starts, ends = starts[valid], ends[valid]
    result: dict[str, Any] = dict(segment=length, count=int(starts.size))
    if not starts.size:
        return result
    r_est, r_true = Rotation.from_quat(estimated_xyzw), Rotation.from_quat(true_xyzw)
    moved_est = r_est[starts].inv().apply(est[ends] - est[starts])
    moved_true = r_true[starts].inv().apply(true[ends] - true[starts])
    translation = np.linalg.norm(moved_est - moved_true, axis=1)
    turn_est = r_est[starts].inv() * r_est[ends]
    turn_true = r_true[starts].inv() * r_true[ends]
    rotation = np.degrees((turn_true.inv() * turn_est).magnitude())
    result.update(
        translation_m=error_statistics(translation),
        rotation_deg=error_statistics(rotation),
    )
    if segment_m is not None:
        travelled = axis[ends] - axis[starts]
        result["drift_pct"] = error_statistics(100 * translation / travelled)
    return result


def segment_key(length: float, unit: str) -> str:
    """Summary key for a segment, e.g. ``10m`` or ``1s`` (``0.5s``)."""
    return f"{length:g}{unit}"


def trajectory_summary(
    times_s: Sequence[float],
    estimated_m: NDArray[np.float64] | None,
    estimated_xyzw: NDArray[np.float64] | None,
    true_m: NDArray[np.float64],
    true_xyzw: NDArray[np.float64],
    segments_s: Sequence[float],
    segments_m: Sequence[float],
) -> dict[str, Any]:
    """ATE (none and yaw alignment) and RPE for every segment length.

    Without estimated positions there is nothing to report; without
    attitudes, ATE has no rotation part and RPE is omitted.
    """
    if estimated_m is None or len(estimated_m) < 2:
        return {}
    ate = {
        mode: absolute_trajectory_error(
            estimated_m, true_m, mode, estimated_xyzw, true_xyzw
        )
        for mode in ("none", "yaw")
    }
    result: dict[str, Any] = dict(ate=ate)
    if estimated_xyzw is not None:
        rpe = {}
        for length in segments_s:
            rpe[segment_key(length, "s")] = relative_pose_error(
                times_s,
                estimated_m,
                estimated_xyzw,
                true_m,
                true_xyzw,
                segment_s=length,
            )
        for length in segments_m:
            rpe[segment_key(length, "m")] = relative_pose_error(
                times_s,
                estimated_m,
                estimated_xyzw,
                true_m,
                true_xyzw,
                segment_m=length,
            )
        result["rpe"] = rpe
    return result
