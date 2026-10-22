"""Static plots work without a display server."""


def plot_flight(data, metadata, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np

    fig = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*data["position"].T, label="Ground truth")
    ax.plot(*data["estimate_position"].T, label="IMU dead reckoning")
    ax.scatter(*data["position_measurement"].T, s=5, alpha=0.4, label="Position sensor")
    ax.scatter(*metadata["target"], marker="*", s=110, color="black", label="Waypoint")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=metadata["backend"])
    ax.legend(fontsize=8)
    error = fig.add_subplot(122)
    error.plot(
        data["time"],
        np.linalg.norm(data["estimate_position"] - data["position"], axis=1),
    )
    error.set(
        xlabel="Time (s)",
        ylabel="Position error (m)",
        title="IMU drift (no corrections)",
    )
    error.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
