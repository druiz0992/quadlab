"""Camera calibration as estimators receive it: intrinsics and optical frame.

Pixel coordinates follow the OpenCV convention: (u, v) with u right and v
down, pixel centres at integers, so the principal point of a symmetric
frustum is ((width − 1)/2, (height − 1)/2). The optical frame is z forward
(along the optical axis), x right, y down. Gazebo camera axes, which
``SensorDescription`` mounts use, are x forward, y left, z up; the fixed
rotation between them is ``SENSOR_FROM_OPTICAL``.

Projection (``project``) is pinhole plus Brown–Conrady distortion
(k1, k2, k3 radial; p1, p2 tangential), the model RobotLab applies to
rendered frames (``sensors/imaging/camera_effects.py``):

    x = X/Z, y = Y/Z,  r² = x² + y²
    x_d = x (1 + k1 r² + k2 r⁴ + k3 r⁶) + 2 p1 x y + p2 (r² + 2x²)
    y_d = y (1 + k1 r² + k2 r⁴ + k3 r⁶) + p1 (r² + 2y²) + 2 p2 x y
    u = fx x_d + cx,  v = fy y_d + cy

For wide-angle Gazebo lenses (``lens_type`` other than ``pinhole``) the
focal length is the small-angle scale ``width / horizontal_fov`` and the
pinhole projection is only an approximation; ``project`` raises for them.

The values are the simulated camera's exact configuration, i.e. a perfect
calibration; real calibrations carry errors an estimator must tolerate.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Columns: optical x (right), y (down), z (forward) in Gazebo camera axes.
SENSOR_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics (pixels), distortion, and shutter timing of one camera.

    ``exposure_time_s`` and ``rolling_shutter_readout_s`` (rows read top
    to bottom, 0 for a global shutter) are what a camera driver reports;
    frame latency is not included (an estimator must not know it a priori).
    """

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    lens_type: str = "pinhole"
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    exposure_time_s: float = 0.0
    rolling_shutter_readout_s: float = 0.0

    @classmethod
    def from_settings(cls, camera: Mapping[str, Any]) -> "CameraCalibration":
        """From a validated ``sensors.camera`` configuration section."""
        width, height = int(camera["width"]), int(camera["height"])
        fov = camera["horizontal_fov_rad"]
        lens = camera["lens"]
        if lens["type"] == "pinhole":
            focal = (width / 2) / math.tan(fov / 2)
        else:
            focal = width / fov  # small-angle scale of wide-angle projections
        return cls(
            width,
            height,
            focal,
            focal,
            (width - 1) / 2,
            (height - 1) / 2,
            lens_type=lens["type"],
            k1=float(lens["k1"]),
            k2=float(lens["k2"]),
            k3=float(lens["k3"]),
            p1=float(lens["p1"]),
            p2=float(lens["p2"]),
            exposure_time_s=float(camera["exposure_time_s"]),
            rolling_shutter_readout_s=float(camera["rolling_shutter_readout_s"]),
        )

    @property
    def matrix(self) -> NDArray[np.float64]:
        """3×3 camera matrix K."""
        return np.array(
            [[self.fx_px, 0.0, self.cx_px], [0.0, self.fy_px, self.cy_px], [0, 0, 1]]
        )

    @property
    def distortion(self) -> NDArray[np.float64]:
        """OpenCV-ordered coefficients (k1, k2, p1, p2, k3)."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def project(self, points_optical_m: ArrayLike) -> NDArray[np.float64]:
        """Pixels (N×2, u, v) of optical-frame points (N×3, Z > 0).

        Raises ``ValueError`` for points at or behind the camera and for
        wide-angle lenses.
        """
        if self.lens_type != "pinhole":
            raise ValueError(f"project supports pinhole lenses, not {self.lens_type}")
        points = np.atleast_2d(np.asarray(points_optical_m, float))
        if np.any(points[:, 2] <= 0):
            raise ValueError("points must be in front of the camera (Z > 0)")
        x, y = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        x_d = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_d = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return np.column_stack(
            [self.fx_px * x_d + self.cx_px, self.fy_px * y_d + self.cy_px]
        )
