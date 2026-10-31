"""Animation of a finished simulation on the inD background image."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

import trajectory_prediction  # noqa: F401  (puts Utils/ on sys.path)
from Utils.Draw_trajectory import Draw_trajectory
from Utils.Draw_vehicle import Draw_vehicle

VEHICLE_COLOR = (1.0, 0.6, 0.2)
EGO_COLOR = "red"


def _to_pixels(points, m2p):
    pixels = np.array(points, dtype=np.float64, copy=True)
    pixels[:, 0] *= m2p
    pixels[:, 1] *= -m2p
    return pixels


class _GifWriter(PillowWriter):
    """PillowWriter that plays the GIF once unless `loop` is True."""

    def __init__(self, fps, loop=False, **kwargs):
        super().__init__(fps=fps, **kwargs)
        self._loop = loop

    def finish(self):
        options = {"loop": 0} if self._loop else {}
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(round(1000 / self.fps)),
            **options,
        )


def animate(cfg, world, result, fps=None, show=True, save_path=None, loop=False):
    """
    Draw every simulated frame: background, surrounding vehicles with their
    ground truth / predicted paths, the global path, the ego vehicle and the
    planned (green) trajectory.

    fps defaults to the replay rate (1 / cfg.frame_dt). A saved GIF plays once unless
    loop=True.
    """
    fps = fps or 1.0 / cfg.frame_dt
    if not result.tracks:
        raise ValueError("The result has no tracks; run with keep_tracks=True.")

    m2p = cfg.m2p
    image = world.background_rgb
    global_path_px = _to_pixels(world.global_path[:, :2], m2p)
    start_px = _to_pixels([cfg.ego_start[:2]], m2p)[0]
    goal_px = _to_pixels([cfg.ego_goal[:2]], m2p)[0]

    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame_index):
        ax.clear()
        ax.imshow(image)

        for track in result.tracks[frame_index]:
            track_data = track["track_data"]
            polygon = np.asarray(Draw_vehicle(track_data, m2p))

            if polygon.size > 0:
                ax.fill(polygon[:, 0], polygon[:, 1], color=VEHICLE_COLOR, alpha=0.5)

            if track["gt_traj"] is not None and track["pred_traj"] is not None:
                Draw_trajectory(
                    ax, track_data, track["gt_traj"], track["pred_traj"], m2p
                )

        ax.plot(
            global_path_px[:, 0], global_path_px[:, 1],
            "--", color="yellow", linewidth=1.5, label="Global Path",
        )

        state = result.ego_states[min(frame_index, len(result.ego_states) - 1)]
        ego_data = pd.DataFrame(
            {
                "xCenter": [state[0]],
                "yCenter": [state[1]],
                "heading": [np.rad2deg(state[2])],
                "length": [cfg.ego_length],
                "width": [cfg.ego_width],
            }
        )
        ego_polygon = Draw_vehicle(ego_data, m2p)
        ax.fill(
            ego_polygon[:, 0], ego_polygon[:, 1],
            color=EGO_COLOR, alpha=0.9, label="Ego Vehicle",
        )

        planned_px = _to_pixels(
            result.planned_xy[min(frame_index, len(result.planned_xy) - 1)], m2p
        )
        ax.plot(planned_px[:, 0], planned_px[:, 1], color="green", linewidth=2.5)

        ax.scatter(
            start_px[0], start_px[1], s=100, marker="s", color="white",
            edgecolors="black", linewidths=1.5, label="Ego Start",
        )
        ax.scatter(
            goal_px[0], goal_px[1], s=180, marker="*", color="yellow",
            edgecolors="black", linewidths=1.5, label="Ego Goal",
        )

        ax.set_title(f"inD Environment | Frame {result.frames[frame_index]}")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="upper right")

    animation = FuncAnimation(
        fig, update, frames=len(result.frames), interval=1000.0 / fps, repeat=False
    )

    if save_path:
        if str(save_path).lower().endswith(".gif"):
            animation.save(save_path, writer=_GifWriter(fps, loop=loop))
        else:
            animation.save(save_path, fps=fps)

    if show:
        plt.show()

    return animation

