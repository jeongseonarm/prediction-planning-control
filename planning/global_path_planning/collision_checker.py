import numpy as np


class ImageMaskCollisionChecker:

    def __init__(
        self,
        binary_mask: np.ndarray,
        m2p: float,
        veh_len: float = 4.5,
        veh_width: float = 1.8,
        margin: float = 0.1,  # 차체 외곽 안전 마진 (0.3 -> 0.1로 완화)
    ):
        self.mask = binary_mask  # 1: 도로, 0: 장애물
        self.m2p = m2p
        self.rows, self.cols = binary_mask.shape

        eff_len = veh_len + margin
        eff_width = veh_width + margin

        # 차체 내부 및 테두리를 21개 점으로 샘플링 (7x3 그리드)
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

        # 맵 경계 초과 검사
        if (
            np.any(cols < 0)
            or np.any(cols >= self.cols)
            or np.any(rows < 0)
            or np.any(rows >= self.rows)
        ):
            return True

        # 0(장애물/인도) 영역에 걸치면 충돌(True)
        if np.any(self.mask[rows, cols] == 0):
            return True

        return False