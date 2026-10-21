"""Map, collision checker and global path of a scenario."""

from dataclasses import dataclass

import cv2
import numpy as np

from planning.global_path_planning.collision_checker import ImageMaskCollisionChecker
from planning.global_path_planning.hybrid_a_star import hybrid_a_star


@dataclass
class World:
    mask: np.ndarray
    map_bounds: tuple               # (x_min, x_max, y_min, y_max) [m]
    collision_checker: ImageMaskCollisionChecker
    global_path: np.ndarray         # (N, 3) x, y, yaw
    background_rgb: np.ndarray = None


def load_background(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Background image not found: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask_img is None:
        raise FileNotFoundError(f"Mask file not found: {path}")

    _, mask = cv2.threshold(mask_img, 0, 1, cv2.THRESH_BINARY)
    return mask


def build_world(cfg, with_background=True):
    """Load the map and plan the global path with Hybrid A*."""
    background = load_background(cfg.background_path) if with_background else None
    mask = load_mask(cfg.mask_path)

    img_h, img_w = mask.shape
    map_bounds = (0.0, img_w / cfg.m2p, -img_h / cfg.m2p, 0.0)

    checker = ImageMaskCollisionChecker(
        binary_mask=mask,
        m2p=cfg.m2p,
        veh_len=cfg.ego_length,
        veh_width=cfg.ego_width,
        margin=cfg.map_margin,
    )

    start = list(cfg.ego_start)
    goal = list(cfg.ego_goal)

    start_collision = checker.check(*start)
    goal_collision = checker.check(*goal)
    print("Start Point Collision:", start_collision)
    print("Goal Point Collision:", goal_collision)

    if start_collision or goal_collision:
        raise RuntimeError("The start or goal pose is outside the drivable mask.")

    print("Planning global path via Hybrid A*...")

    global_path, _ = hybrid_a_star(
        start=start,
        goal=goal,
        map_bounds=map_bounds,
        is_collision_fn=checker.check,
        wheelbase=cfg.wheelbase,
        max_steer=np.deg2rad(cfg.max_steer_deg),
        step_size=cfg.astar_step_size,
        xy_res=cfg.astar_xy_res,
        yaw_res=np.deg2rad(cfg.astar_yaw_res_deg),
        steer_samples=cfg.astar_steer_samples,
    )

    if global_path is None:
        raise RuntimeError("Hybrid A* failed to find a valid global path.")

    return World(
        mask=mask,
        map_bounds=map_bounds,
        collision_checker=checker,
        global_path=np.asarray(global_path, dtype=np.float64),
        background_rgb=background,
    )
