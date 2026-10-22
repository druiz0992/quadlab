"""Live ground-truth view; rendering never changes the simulation timestep."""

from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation


def drone_geometry(state):
    """Display arms at an enlarged scale so attitude is easy to see."""
    points = np.array(
        [
            [-0.18, -0.18, 0],
            [0.18, 0.18, 0],
            [-0.18, 0.18, 0],
            [0.18, -0.18, 0],
            [0, 0, 0],
            [0.28, 0, 0],
        ]
    )
    return Rotation.from_quat(state["q"]).apply(points) + state["x"]


class LiveViewer:
    def __init__(self, target, speed=1.0):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be positive and finite")
        try:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            self.plt = plt
            self.fig = plt.figure(figsize=(9, 7))
        except (ImportError, RuntimeError) as error:
            raise ValueError(
                "Live viewing needs a desktop display and Python Tk. "
                "Run from your desktop terminal; on Ubuntu install python3-tk if missing."
            ) from error
        self.fig.canvas.manager.set_window_title("RobotLab — live RotorPy flight")
        self.ax = self.fig.add_subplot(111, projection="3d")
        bounds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], target])
        low, high = bounds.min(axis=0) - 0.6, bounds.max(axis=0) + 0.6
        low[2] = min(0.0, low[2])
        self.ax.set(
            xlim=(low[0], high[0]),
            ylim=(low[1], high[1]),
            zlim=(low[2], high[2]),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
        )
        self.ax.set_box_aspect(high - low)
        self.ax.scatter(
            *target, marker="*", s=160, color="darkorange", label="Waypoint"
        )
        x, y = np.meshgrid([low[0], high[0]], [low[1], high[1]])
        self.ax.plot_surface(x, y, np.zeros_like(x), color="gray", alpha=0.12)
        self.arms = [
            self.ax.plot([], [], [], color=c, lw=4)[0]
            for c in ["royalblue", "royalblue", "crimson"]
        ]
        (self.trail,) = self.ax.plot(
            [], [], [], color="teal", lw=1.5, label="True flight path"
        )
        (self.rotors,) = self.ax.plot([], [], [], "o", color="black", ms=8)
        self.ax.legend(loc="upper left")
        self.fig.text(
            0.5,
            0.03,
            "Drag to rotate • Red arm points forward • Drone enlarged for visibility",
            ha="center",
        )
        self.positions = []
        self.speed = speed
        self.last_frame = -float("inf")
        self.started = None
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._close)
        plt.show(block=False)

    def _close(self, event):
        self.closed = True

    def __call__(self, t, state):
        if self.closed:
            return
        if self.started is None:
            self.started = perf_counter()
        # Render at most 30 frames per simulated second. Physics stays at 200 Hz.
        if t - self.last_frame < 1 / 30 - 1e-12:
            return
        self.last_frame = t
        self.positions.append(state["x"].copy())
        points = drone_geometry(state)
        for arm, pair in zip(self.arms, [points[:2], points[2:4], points[4:6]]):
            arm.set_data_3d(*pair.T)
        self.rotors.set_data_3d(*points[:4].T)
        self.trail.set_data_3d(*np.asarray(self.positions).T)
        self.ax.set_title(f"Live flight — t = {t:.2f} s — {self.speed:g}× speed")
        self.fig.canvas.draw_idle()
        # Service window events while pacing against a monotonic clock.
        while not self.closed:
            remaining = self.started + t / self.speed - perf_counter()
            self.plt.pause(min(0.02, max(0.001, remaining)))
            if remaining <= 0.02:
                break

    def finish(self):
        if not self.closed:
            self.ax.set_title("Flight complete — close this window to exit")
            self.plt.show(block=True)
