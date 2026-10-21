import numpy as np


class ImageMaskCollisionChecker:

    def __init__(
        self,
        binary_mask: np.ndarray,
        m2p: float,
        veh_len: float = 4.5,
        veh_width: float = 1.8,
        margin: float = 0.1,  # safety margin around the vehicle body [m]
    ):
        self.mask = binary_mask  # 1: drivable, 0: blocked
        self.m2p = m2p
        self.rows, self.cols = binary_mask.shape

        eff_len = veh_len + margin
        eff_width = veh_width + margin

        # sample the vehicle body with a 7x3 grid of points
        xs = np.linspace(-eff_len / 2.0, eff_len / 2.0, num=7)
        ys = np.linspace(-eff_width / 2.0, eff_width / 2.0, num=3)
        gx, gy = np.meshgrid(xs, ys)
        self.local_pts = np.vstack([gx.ravel(), gy.ravel()]).T

    def check(self, x: float, y: float, yaw: float) -> bool:
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)

        wx = x + (
            self.local_pts[:, 0] * cos_t - self.local_pts[:, 1] * sin_t
        )
        wy = y + (
            self.local_pts[:, 0] * sin_t + self.local_pts[:, 1] * cos_t
        )

        cols = np.round(wx * self.m2p).astype(int)
        rows = np.round(-wy * self.m2p).astype(int)

        # outside the map
        if (
            np.any(cols < 0)
            or np.any(cols >= self.cols)
            or np.any(rows < 0)
            or np.any(rows >= self.rows)
        ):
            return True

        # any point on a blocked cell (0) is a collision
        if np.any(self.mask[rows, cols] == 0):
            return True

        return False

    def check_many(self, xs, ys, yaws) -> bool:
        """True if ANY of the poses collides (same rule as check, vectorized)."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        yaws = np.asarray(yaws, dtype=float)

        cos_t = np.cos(yaws)[:, None]
        sin_t = np.sin(yaws)[:, None]
        px = self.local_pts[None, :, 0]
        py = self.local_pts[None, :, 1]

        wx = xs[:, None] + (px * cos_t - py * sin_t)
        wy = ys[:, None] + (px * sin_t + py * cos_t)

        cols = np.round(wx * self.m2p).astype(int)
        rows = np.round(-wy * self.m2p).astype(int)

        if (
            np.any(cols < 0)
            or np.any(cols >= self.cols)
            or np.any(rows < 0)
            or np.any(rows >= self.rows)
        ):
            return True

        return bool(np.any(self.mask[rows, cols] == 0))
