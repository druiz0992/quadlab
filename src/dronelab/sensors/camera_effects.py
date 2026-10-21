"""Camera image effects Gazebo does not render: distortion, blur, rolling shutter.

Applied in Python to each RGB frame before recording. Gazebo still renders
the scene, including native wide-angle (fisheye) lenses; these effects are
added on top:

- **Lens distortion** (Brown–Conrady ``k1, k2, k3, p1, p2``): gz-rendering's
  ogre2 backend reports "ImageBrownDistortionModel is not supported", so the
  frame is remapped here. Output pixel at normalized coordinate x_d samples
  the rendered image at the undistorted point x_u with distort(x_u) = x_d
  (solved by fixed-point iteration, precomputed once).
- **Motion blur** over ``exposure_time_s`` and **rolling shutter** with
  rows read top to bottom over ``rolling_shutter_readout_s``. Both use the
  image motion caused by the camera's rotation, evaluated at the image
  centre: yaw about camera +z moves the scene right by f·ω_z per second and
  pitch about camera +y moves it up by f·ω_y (camera axes: x forward,
  y left, z up). Translation and roll are ignored.

Pixel shifts are rounded to whole pixels and image borders are extended.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_BLUR_SAMPLES = 31


def focal_length_px(width: int, horizontal_fov_rad: float, lens_type: str) -> float:
    """Pixels per radian near the image centre for the lens projection."""
    if lens_type == "pinhole":
        return (width / 2) / math.tan(horizontal_fov_rad / 2)
    return width / horizontal_fov_rad  # wide-angle projections: ~linear in angle


def image_motion_px_s(
    focal_px: float, rates_camera_rad_s: ArrayLike
) -> NDArray[np.float64]:
    """Image velocity (du, dv) in px/s at the centre (u right, v down)."""
    _, wy, wz = np.asarray(rates_camera_rad_s, float)
    return np.array([focal_px * wz, -focal_px * wy])


def shift_image(image: NDArray[np.uint8], du: int, dv: int) -> NDArray[np.uint8]:
    """Shift content by (du, dv) pixels, extending edge pixels."""
    h, w = image.shape[:2]
    rows = np.clip(np.arange(h) - dv, 0, h - 1)
    cols = np.clip(np.arange(w) - du, 0, w - 1)
    return image[rows[:, None], cols[None, :]]


def motion_blur(image: NDArray[np.uint8], motion_px: ArrayLike) -> NDArray[np.uint8]:
    """Average copies shifted along the motion vector (centred on the frame)."""
    motion = np.asarray(motion_px, float)
    samples = min(int(math.ceil(np.hypot(*motion))) + 1, MAX_BLUR_SAMPLES)
    if samples <= 1:
        return image
    total = np.zeros(image.shape, float)
    for fraction in np.linspace(-0.5, 0.5, samples):
        du, dv = np.rint(fraction * motion).astype(int)
        total += shift_image(image, int(du), int(dv))
    return np.rint(total / samples).astype(np.uint8)


def rolling_shutter(
    image: NDArray[np.uint8], motion_px_s: ArrayLike, readout_s: float
) -> NDArray[np.uint8]:
    """Row r captured (r / (h−1) − ½)·readout after mid-frame, shifted by motion."""
    h, w = image.shape[:2]
    offsets = (np.arange(h) / max(h - 1, 1) - 0.5) * readout_s
    du, dv = np.asarray(motion_px_s, float)
    rows = np.clip(np.rint(np.arange(h) - dv * offsets), 0, h - 1).astype(int)
    shifts = np.rint(du * offsets).astype(int)
    cols = np.clip(np.arange(w)[None, :] - shifts[:, None], 0, w - 1)
    return image[rows[:, None], cols]


def distortion_map(
    width: int,
    height: int,
    focal_px: float,
    coefficients: Mapping[str, float],
    iterations: int = 8,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Source (row, column) for each output pixel under Brown–Conrady distortion."""
    k1, k2, k3, p1, p2 = (coefficients[k] for k in ("k1", "k2", "k3", "p1", "p2"))
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    xd, yd = (u - (width - 1) / 2) / focal_px, (v - (height - 1) / 2) / focal_px
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):  # invert distort(x_u) = x_d
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / radial, (yd - dy) / radial
    cols = np.clip(np.rint(x * focal_px + (width - 1) / 2), 0, width - 1).astype(int)
    rows = np.clip(np.rint(y * focal_px + (height - 1) / 2), 0, height - 1).astype(int)
    return rows, cols


class CameraEffects:
    """Apply configured camera effects to frames (``sensors.camera`` settings)."""

    def __init__(self, camera: Mapping[str, Any]) -> None:
        lens = camera["lens"]
        self._exposure_s = camera["exposure_time_s"]
        self._readout_s = camera["rolling_shutter_readout_s"]
        self.latency_s: float = camera["latency_s"]
        self._focal = focal_length_px(
            camera["width"], camera["horizontal_fov_rad"], lens["type"]
        )
        distorted = any(lens[k] for k in ("k1", "k2", "k3", "p1", "p2"))
        self._map = (
            distortion_map(camera["width"], camera["height"], self._focal, lens)
            if distorted
            else None
        )
        self.enabled = bool(
            camera["enabled"]
            and (distorted or self._exposure_s or self._readout_s or self.latency_s)
        )
        self.max_blur_px = 0.0

    def process(
        self, image: NDArray[np.uint8], rates_camera_rad_s: ArrayLike
    ) -> NDArray[np.uint8]:
        """Return the frame with distortion, rolling shutter, and motion blur applied."""
        if self._map is not None:
            rows, cols = self._map
            image = image[rows, cols]
        motion = image_motion_px_s(self._focal, rates_camera_rad_s)
        if self._readout_s:
            image = rolling_shutter(image, motion, self._readout_s)
        if self._exposure_s:
            blur = motion * self._exposure_s
            self.max_blur_px = max(self.max_blur_px, float(np.hypot(*blur)))
            image = motion_blur(image, blur)
        return image
