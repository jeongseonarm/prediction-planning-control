"""Draw predicted and ground-truth future vehicle trajectories."""

import numpy as np
import matplotlib.patheffects as path_effects


PRED_COLOR = "#F0009B"  # magenta: model prediction
GT_COLOR = "#35D5FF"    # cyan: future ground truth


def _to_image(points, m2p):
    points = np.asarray(points, dtype=float)
    image_points = points[:, :2].copy()
    image_points[:, 0] *= m2p
    image_points[:, 1] *= -m2p
    return image_points


def _outlined_plot(ax, points, color, linewidth, **kwargs):
    """Keep trajectory colors legible over both road and vehicle imagery."""
    return ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        path_effects=[
            path_effects.Stroke(
                linewidth=linewidth + 0.6, foreground="#202020", alpha=0.72
            ),
            path_effects.Normal(),
        ],
        **kwargs,
    )[0]


def Draw_trajectory(ax, Track_data, GT_traj, Pred_traj, M2P):
    """Plot prediction and future ground truth for one surrounding vehicle."""
    if Pred_traj is not None and len(Pred_traj):
        predicted = _to_image(Pred_traj, M2P)
        _outlined_plot(
            ax, predicted, PRED_COLOR, 1.4,
            linestyle=(0, (6, 2.5)), alpha=1.0, zorder=7,
        )

    if GT_traj is not None and len(GT_traj):
        ground_truth = _to_image(GT_traj, M2P)
        _outlined_plot(
            ax, ground_truth, GT_COLOR, 1.1,
            linestyle="-", alpha=0.9, zorder=6,
        )
