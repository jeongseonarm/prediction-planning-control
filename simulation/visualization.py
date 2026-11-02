"""Animation of a finished simulation on the inD background image."""

import math

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from simulation.visualization_helpers.draw_trajectory import (
    Draw_trajectory,
    GT_COLOR,
    PRED_COLOR,
)
from simulation.visualization_helpers.draw_vehicle import Draw_vehicle

VEHICLE_COLOR = "#FFAB40"
EGO_COLOR = "#FF1744"
GLOBAL_PATH_COLOR = "#FFFFFF"
PLANNED_PATH_COLOR = "#00E676"

_GIF_FIXED_COLORS = [
    (32, 32, 32),
    (255, 255, 255),
    (255, 171, 64),
    (255, 23, 68),
    (240, 0, 155),
    (53, 213, 255),
    (255, 215, 64),
    (0, 230, 118),
]


def _legend_handles():
    """Static handles keep the legend complete even when tracks lack GT."""
    return [
        Patch(facecolor=VEHICLE_COLOR, edgecolor="#202020", alpha=0.75,
              label="Other vehicle"),
        Line2D([0], [0], color=PRED_COLOR, linewidth=1.4,
               linestyle=(0, (6, 2.5)),
               label="Predicted trajectory"),
        Line2D([0], [0], color=GT_COLOR, linewidth=1.1,
               linestyle="-",
               label="Ground truth"),
        Line2D([0], [0], color=GLOBAL_PATH_COLOR, linewidth=2,
               linestyle="--", path_effects=[
                   path_effects.Stroke(linewidth=4, foreground="#202020"),
                   path_effects.Normal(),
               ], label="Global path"),
        Line2D([0], [0], color=PLANNED_PATH_COLOR, linewidth=2.5,
               path_effects=[
                   path_effects.Stroke(linewidth=4.5, foreground="#202020"),
                   path_effects.Normal(),
               ], label="Ego planned path"),
        Patch(facecolor=EGO_COLOR, edgecolor="#202020", label="Ego vehicle"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor="#202020", markersize=7, label="Ego start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#FFD740",
               markeredgecolor="#202020", markersize=10, label="Ego goal"),
    ]


def _shared_gif_palette(frames):
    """Build one palette for the whole GIF so flat line colors do not flicker."""
    thumb_width, thumb_height = 160, 108
    columns = max(1, int(math.ceil(math.sqrt(len(frames)))))
    rows = int(math.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height))

    for index, frame in enumerate(frames):
        thumb = frame.convert("RGB")
        thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.BILINEAR)
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(thumb, (x, y))

    adaptive = sheet.quantize(
        colors=256 - len(_GIF_FIXED_COLORS), method=Image.Quantize.MEDIANCUT
    )
    adaptive_palette = adaptive.getpalette() or [0] * 768
    palette = [0] * 768

    for index, color in enumerate(_GIF_FIXED_COLORS):
        palette[index * 3:index * 3 + 3] = color

    offset = len(_GIF_FIXED_COLORS) * 3
    palette[offset:] = adaptive_palette[:768 - offset]

    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(palette)
    return palette_image


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
        if not self._frames:
            return

        palette = _shared_gif_palette(self._frames)
        # Replace each captured RGBA frame in place to avoid a second full-size
        # frame list while encoding longer recordings.
        for index, frame in enumerate(self._frames):
            self._frames[index] = frame.convert("RGB").quantize(
                palette=palette, dither=Image.Dither.NONE
            )
        frames = self._frames
        options = {"loop": 0} if self._loop else {}
        frames[0].save(
            self.outfile,
            save_all=True,
            append_images=frames[1:],
            duration=int(round(1000 / self.fps)),
            disposal=2,
            optimize=False,
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
    mask_rows, mask_cols = np.where(world.mask > 0)
    padding_px = int(round(4.0 * m2p))
    view_x = (
        max(0, int(mask_cols.min()) - padding_px),
        min(image.shape[1], int(mask_cols.max()) + padding_px),
    )
    view_y = (
        max(0, int(mask_rows.min()) - padding_px),
        min(image.shape[0], int(mask_rows.max()) + padding_px),
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame_index):
        ax.clear()
        ax.imshow(image)

        for track in result.tracks[frame_index]:
            track_data = track["track_data"]
            polygon = np.asarray(Draw_vehicle(track_data, m2p))

            if polygon.size > 0:
                ax.fill(
                    polygon[:, 0], polygon[:, 1],
                    facecolor=VEHICLE_COLOR, edgecolor="#202020",
                    linewidth=0.45, alpha=0.68, zorder=3,
                )

            Draw_trajectory(
                ax, track_data, track.get("gt_traj"), track.get("pred_traj"), m2p
            )

        global_line, = ax.plot(
            global_path_px[:, 0], global_path_px[:, 1],
            "--", color=GLOBAL_PATH_COLOR, linewidth=2.0, zorder=4,
        )
        global_line.set_path_effects([
            path_effects.Stroke(linewidth=4.0, foreground="#202020"),
            path_effects.Normal(),
        ])

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
            facecolor=EGO_COLOR, edgecolor="#202020", linewidth=1.0,
            alpha=0.96, zorder=8,
        )

        planned_px = _to_pixels(
            result.planned_xy[min(frame_index, len(result.planned_xy) - 1)], m2p
        )
        planned_line, = ax.plot(
            planned_px[:, 0], planned_px[:, 1],
            color=PLANNED_PATH_COLOR, linewidth=2.5, zorder=7,
        )
        planned_line.set_path_effects([
            path_effects.Stroke(linewidth=4.5, foreground="#202020"),
            path_effects.Normal(),
        ])

        ax.scatter(
            start_px[0], start_px[1], s=100, marker="s", color="white",
            edgecolors="#202020", linewidths=1.5, zorder=9,
        )
        ax.scatter(
            goal_px[0], goal_px[1], s=180, marker="*", color="#FFD740",
            edgecolors="#202020", linewidths=1.5, zorder=9,
        )

        ax.set_title(f"inD Environment | Frame {result.frames[frame_index]}")
        ax.set_xlim(*view_x)
        ax.set_ylim(view_y[1], view_y[0])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(
            handles=_legend_handles(), loc="upper right", ncol=2,
            fontsize=8, framealpha=0.94, facecolor="white", edgecolor="#444444",
            borderpad=0.55, labelspacing=0.45, handlelength=1.8,
            columnspacing=0.9,
        )

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

