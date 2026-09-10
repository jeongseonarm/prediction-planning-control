import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# ============================================================
# Project Path
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_DIR = os.path.join(ROOT_DIR, "control")
PLANNING_DIR = os.path.join(ROOT_DIR, "planning")
PREDICTION_DIR = os.path.join(ROOT_DIR, "trajectory_prediction")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CONTROL_DIR)
sys.path.insert(0, PLANNING_DIR)
sys.path.insert(0, PREDICTION_DIR)


# ============================================================
# Import
# ============================================================

from controllers.mpc import MPCController
from controllers.pure_pursuit import PurePursuitController
from global_path_planning.collision_checker import (
    ImageMaskCollisionChecker,
)
from global_path_planning.hybrid_a_star import hybrid_a_star
from models.kinematic_bicycle import KinematicBicycle
from Models.Dual_transformer import Dual_transformer
from Utils.Average_deviation import Average_deviation
from Utils.Base import Read_config, reverse_standard_scale
from Utils.Draw_trajectory import Draw_trajectory
from Utils.Draw_vehicle import Draw_vehicle
from Utils.Make_tensor import Make_tensor

# ============================================================
# Configuration
# ============================================================

DATA_NUM = 18

DATA_PATH = os.path.join(
    PREDICTION_DIR, "Data", "18_tracks_del_vru_static_maneuver.csv"
)

BACKGROUND_PATH = os.path.join(
    PREDICTION_DIR,
    "Data",
    "inD_backgroud_images",
    f"{DATA_NUM:02d}_background.png",
)

MASK_PATH = os.path.join(
    ROOT_DIR,
    "map",
    "drivable_mask_images",
    f"{DATA_NUM:02d}_mask.png",
)

LOG_DIR = os.path.join(PREDICTION_DIR, "log")

M2P = 10.2


# ============================================================
# Ego Vehicle
# ============================================================

# State for planning = [x, y, theta]
EGO_START = [20.0, -37.0, np.deg2rad(25.0)]
EGO_GOAL = [50.0, -42.0, np.deg2rad(-70.0)]

# Initial velocity for MPC state = [x, y, theta, v]
INITIAL_SPEED = 5.0

EGO_LENGTH = 2.5
EGO_WIDTH = 1.8
WHEELBASE = 2.0

DT = 0.04

MPC_HORIZON = 10


# ============================================================
# Velocity Profile
# ============================================================

TARGET_SPEED = 5.0


def calc_speed_profile(global_path, target_speed):

    speed_profile = np.full(
        len(global_path),
        target_speed,
        dtype=np.float64,
    )

    speed_profile[-1] = 0.0

    return speed_profile


# ============================================================
# Visualization / Evaluation Frames
# ============================================================

INPUT_FRAME = 25
OUTPUT_FRAME = 75

PREDICTION_START_FRAME = 200
PREDICTION_END_FRAME = 500

VEHICLE_COLOR = (1.0, 0.6, 0.2)
EGO_COLOR = "red"


# ============================================================
# Main
# ============================================================


def main():

    # ========================================================
    # Load Data
    # ========================================================

    Data = pd.read_csv(DATA_PATH)
    Data = Data.sort_values(["frame", "trackId"]).reset_index(drop=True)

    frames = sorted(Data["frame"].unique())

    frames = [
        frame
        for frame in frames
        if PREDICTION_START_FRAME <= frame < PREDICTION_END_FRAME
    ]

    # ========================================================
    # Load Background
    # ========================================================

    IMG = cv2.imread(BACKGROUND_PATH)

    if IMG is None:
        raise FileNotFoundError(
            f"Background image not found: {BACKGROUND_PATH}"
        )

    IMG_RGB = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)

    print("Background:", IMG_RGB.shape)

    # ========================================================
    # Load Drivable Mask
    # ========================================================

    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    if mask_img is None:
        raise FileNotFoundError(f"Mask file not found: {MASK_PATH}")

    _, mask = cv2.threshold(mask_img, 0, 1, cv2.THRESH_BINARY)

    img_h, img_w = mask.shape

    map_bounds = (
        0.0,
        img_w / M2P,
        -img_h / M2P,
        0.0,
    )

    # ========================================================
    # Create Collision Checker
    # ========================================================

    collision_checker = ImageMaskCollisionChecker(
        binary_mask=mask,
        m2p=M2P,
        veh_len=EGO_LENGTH,
        veh_width=EGO_WIDTH,
        margin=0.1,
    )

    # ========================================================
    # Check Start / Goal
    # ========================================================

    hybrid_start = [
        EGO_START[0],
        EGO_START[1],
        EGO_START[2],
    ]

    hybrid_goal = [
        EGO_GOAL[0],
        EGO_GOAL[1],
        EGO_GOAL[2],
    ]

    start_collision = collision_checker.check(
        hybrid_start[0],
        hybrid_start[1],
        hybrid_start[2],
    )

    goal_collision = collision_checker.check(
        hybrid_goal[0],
        hybrid_goal[1],
        hybrid_goal[2],
    )

    print("Start Point Collision:", start_collision)
    print("Goal Point Collision:", goal_collision)

    if start_collision or goal_collision:
        raise RuntimeError(
            "Start 또는 Goal 위치가 drivable mask 밖에 있습니다."
        )

    # ========================================================
    # Generate Global Path using Hybrid A*
    # ========================================================

    print("Planning global path via Hybrid A*...")

    global_path, path_cost = hybrid_a_star(
        start=hybrid_start,
        goal=hybrid_goal,
        map_bounds=map_bounds,
        is_collision_fn=collision_checker.check,
        wheelbase=WHEELBASE,
        max_steer=np.deg2rad(35.0),
        step_size=0.5,
        xy_res=0.5,
        yaw_res=np.deg2rad(10.0),
        steer_samples=15,
    )

    if global_path is None:
        raise RuntimeError(
            "Hybrid A* failed to find a valid global path."
        )

    global_path = np.asarray(global_path, dtype=np.float64)

    # ========================================================
    # Global Path Preprocessing & Cumulative Distance
    # ========================================================

    global_heading = np.unwrap(global_path[:, 2])

    dx = np.diff(global_path[:, 0])
    dy = np.diff(global_path[:, 1])
    ds = np.hypot(dx, dy)
    s_path = np.insert(np.cumsum(ds), 0, 0.0) # 0,0.0 -> 0번째 index에 0값 삽입. ds0 = 0을 추가.

    # ========================================================
    # Speed Profile Generation
    # ========================================================

    speed_profile = calc_speed_profile(
        global_path,
        TARGET_SPEED,
    )

    print("Target Speed:", TARGET_SPEED)
    print("Final Speed:", speed_profile[-1])

    # ========================================================
    # Create Ego Vehicle & MPC
    # ========================================================

    ego_vehicle = KinematicBicycle(
        wheelbase=WHEELBASE,
        dt=DT,
    )

    ego_controller = MPCController(
        model=ego_vehicle,
        horizon=MPC_HORIZON,
    )

    # ego_controller = PurePursuitController(
    #     wheelbase=WHEELBASE,
    #     k_lookahead=0.5,
    #     min_lookahead=2.0,
    #     kp_speed=TARGET_SPEED
    # )

    # ========================================================
    # Simulate Ego
    # ========================================================

    # MPC State = [x, y, theta, v]
    ego_state = np.array(
        [
            EGO_START[0],
            EGO_START[1],
            EGO_START[2],
            INITIAL_SPEED,
        ],
        dtype=np.float64,
    )

    ego_trajectory = []

    previous_index = 0

    for _ in range(len(frames)):

        # ----------------------------------------------------
        # Save Current State
        # ----------------------------------------------------

        ego_trajectory.append(ego_state.copy())

        # ----------------------------------------------------
        # Find Nearest Global Path Point
        # ----------------------------------------------------

        distance_to_path = np.sqrt(
            (global_path[:, 0] - ego_state[0]) ** 2
            + (global_path[:, 1] - ego_state[1]) ** 2
        )

        nearest_index = np.argmin(distance_to_path)
        nearest_index = max(nearest_index, previous_index)

        previous_index = nearest_index

        # ----------------------------------------------------
        # Construct MPC Reference Trajectory via 1D Interpolation
        # ----------------------------------------------------

        s_0 = s_path[nearest_index]
        target_v = speed_profile[nearest_index]

        t_steps = np.linspace(0, MPC_HORIZON * DT, MPC_HORIZON + 1)
        s_targets = np.clip(s_0 + target_v * t_steps, s_path[0], s_path[-1]) 

        ref_x = np.interp(s_targets, s_path, global_path[:, 0])
        ref_y = np.interp(s_targets, s_path, global_path[:, 1])
        ref_yaw = np.interp(s_targets, s_path, global_heading)
        ref_v = np.interp(s_targets, s_path, speed_profile)

        reference = np.column_stack([ref_x, ref_y, ref_yaw, ref_v])

        # ----------------------------------------------------
        # MPC Control
        # ----------------------------------------------------

        control_input = ego_controller.compute_control(
            ego_state,
            reference,
        )

        # ----------------------------------------------------
        # Vehicle Update
        # ----------------------------------------------------

        ego_state = ego_vehicle.update(
            ego_state,
            control_input,
        )

        # ----------------------------------------------------
        # Goal Check
        # ----------------------------------------------------

        position_error = np.linalg.norm(
            ego_state[:2] - np.asarray(EGO_GOAL[:2])
        )

        yaw_error = np.arctan2(
            np.sin(ego_state[2] - EGO_GOAL[2]),
            np.cos(ego_state[2] - EGO_GOAL[2]),
        )

        if position_error < 1.0 and abs(yaw_error) < np.deg2rad(10.0):

            ego_trajectory.append(ego_state.copy())

            print("Goal reached.")
            print("Position Error:", position_error)
            print("Yaw Error [deg]:", np.rad2deg(yaw_error))

            break

    ego_trajectory = np.asarray(ego_trajectory)

    print("Ego trajectory:", ego_trajectory.shape)

    # ========================================================
    # Inference Results
    # ========================================================

    # --------------------------------------------------------
    # Load Prediction Configuration
    # --------------------------------------------------------

    config_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))
    Config_dict = {}
    for file in config_files:
        Config_dict.update(Read_config(file))
    Config_dict["Device"] = "cpu"

    Device = torch.device("cpu")

    TEST_MODEL = Dual_transformer(Config_dict).to(Device)

    weight_dir = os.path.join(LOG_DIR, "pt")
    weight_files = glob.glob(os.path.join(weight_dir, "*.pt"))

    if len(weight_files) == 0:
        raise FileNotFoundError(
            f"No model weight file found in: {weight_dir}"
        )

    latest_weight = max(
        weight_files,
        key=os.path.getmtime,
    )

    TEST_MODEL.load_state_dict(
        torch.load(
            latest_weight,
            map_location=Device,
        )
    )

    TEST_MODEL.eval()

    # ========================================================
    # Prediction Inference
    # ========================================================

    results = []

    for frame in tqdm(frames, desc="Trajectory Prediction"):

        Original_input = Data[Data["frame"] == frame]

        Assumption_input = Data[
            (frame >= Data["frame"])
            & (Data["frame"] > frame - INPUT_FRAME)
        ]

        Unique_track_id = Original_input["trackId"].unique()

        GT = Data[
            (frame + 1 <= Data["frame"])
            & (Data["frame"] <= frame + OUTPUT_FRAME)
        ]

        frame_result = []

        for track_id in Unique_track_id:

            Track_data = Assumption_input[
                Assumption_input["trackId"] == track_id
            ].copy()

            GT_data = GT[GT["trackId"] == track_id].copy()

            GT_traj = GT_data[["xCenter", "yCenter"]].to_numpy()

            Track_pts = Draw_vehicle(
                Track_data,
                M2P,
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if len(Track_data) == INPUT_FRAME and len(GT_data) == OUTPUT_FRAME:

                Tensor_data = Make_tensor(
                    Track_data,
                    Config_dict,
                )

                with torch.no_grad():

                    Trajectory, *_ = TEST_MODEL(
                        Tensor_data.unsqueeze(0),
                        Tensor_data.unsqueeze(0),
                        Tensor_data.unsqueeze(0),
                    )

                Trajectory = Trajectory[
                    :,
                    :OUTPUT_FRAME,
                    :2,
                ]

                Trajectory[:, :, 0] = reverse_standard_scale(
                    Trajectory[:, :, 0],
                    Config_dict["mean"][0],
                    Config_dict["std"][0],
                )

                Trajectory[:, :, 1] = reverse_standard_scale(
                    Trajectory[:, :, 1],
                    Config_dict["mean"][1],
                    Config_dict["std"][1],
                )

                Pred_traj = Trajectory.squeeze(0).detach().cpu().numpy()

                AD = Average_deviation(
                    GT_traj[:, 0],
                    GT_traj[:, 1],
                    Pred_traj[:, 0],
                    Pred_traj[:, 1],
                )

                frame_result.append(
                    {
                        "id": track_id,
                        "track_data": Track_data.copy(),
                        "gt_traj": GT_traj.copy(),
                        "pred_traj": Pred_traj.copy(),
                        "track_pts": Track_pts.copy(),
                        "AD": AD,
                    }
                )

            else:

                frame_result.append(
                    {
                        "id": track_id,
                        "track_data": Track_data.copy(),
                        "gt_traj": None,
                        "pred_traj": None,
                        "track_pts": Track_pts.copy(),
                        "AD": None,
                    }
                )

        results.append(
            {
                "frame": frame,
                "tracks": frame_result,
            }
        )

    print(
        "\nInference complete.\n" "Total frames:",
        len(results),
    )

    # ========================================================
    # Visualization
    # ========================================================

    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame_index):

        ax.clear()

        frame_result = results[frame_index]

        frame = frame_result["frame"]
        tracks = frame_result["tracks"]

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        ax.imshow(IMG_RGB)

        # ----------------------------------------------------
        # Surrounding Vehicles
        # ----------------------------------------------------

        for track in tracks:

            Track_data = track["track_data"]
            GT_traj = track["gt_traj"]
            Pred_traj = track["pred_traj"]
            Track_pts = np.asarray(track["track_pts"])
            AD = track["AD"]

            if Track_pts.size > 0:

                ax.fill(
                    Track_pts[:, 0],
                    Track_pts[:, 1],
                    color=VEHICLE_COLOR,
                    alpha=0.5,
                )

            if GT_traj is not None and Pred_traj is not None:

                Draw_trajectory(
                    ax,
                    Track_data,
                    GT_traj,
                    Pred_traj,
                    M2P
                )

        # ----------------------------------------------------
        # Global Path
        # ----------------------------------------------------

        path = global_path.copy()

        path[:, 0] *= M2P
        path[:, 1] *= -M2P

        ax.plot(
            path[:, 0],
            path[:, 1],
            "--",
            color="yellow",
            linewidth=2.5,
            label="Global Path",
        )

        # ----------------------------------------------------
        # Ego Vehicle Position Index
        # ----------------------------------------------------

        ego_index = min(
            frame_index,
            len(ego_trajectory) - 1,
        )

        # ----------------------------------------------------
        # Cumulative Ego Trajectory
        # ----------------------------------------------------

        ego_path = ego_trajectory[
            : ego_index + 1,
            :2,
        ].copy()

        if len(ego_path) > 0:

            ego_path[:, 0] *= M2P
            ego_path[:, 1] *= -M2P

            ax.plot(
                ego_path[:, 0],
                ego_path[:, 1],
                color="red",
                linewidth=2,
                label="Ego Trajectory",
            )

        # ----------------------------------------------------
        # Ego Vehicle
        # ----------------------------------------------------

        state = ego_trajectory[ego_index]

        ego_data = pd.DataFrame(
            {
                "xCenter": [state[0]],
                "yCenter": [state[1]],
                "heading": [np.rad2deg(state[2])],
                "length": [EGO_LENGTH],
                "width": [EGO_WIDTH],
            }
        )

        ego_polygon = Draw_vehicle(
            ego_data,
            M2P,
        )

        ax.fill(
            ego_polygon[:, 0],
            ego_polygon[:, 1],
            color=EGO_COLOR,
            alpha=0.9,
            label="Ego Vehicle",
        )

        # ----------------------------------------------------
        # Ego Start
        # ----------------------------------------------------

        start = np.asarray(
            EGO_START[:2],
            dtype=np.float64,
        )

        start[0] *= M2P
        start[1] *= -M2P

        ax.scatter(
            start[0],
            start[1],
            s=100,
            marker="s",
            color="white",
            edgecolors="black",
            linewidths=1.5,
            label="Ego Start",
        )

        # ----------------------------------------------------
        # Ego Goal
        # ----------------------------------------------------

        goal = np.asarray(
            EGO_GOAL[:2],
            dtype=np.float64,
        )

        goal[0] *= M2P
        goal[1] *= -M2P

        ax.scatter(
            goal[0],
            goal[1],
            s=180,
            marker="*",
            color="yellow",
            edgecolors="black",
            linewidths=1.5,
            label="Ego Goal",
        )

        # ----------------------------------------------------
        # Plot Settings
        # ----------------------------------------------------

        ax.set_title(f"inD Environment | Frame {frame}")

        ax.set_xlim(
            0,
            IMG_RGB.shape[1],
        )

        ax.set_ylim(
            IMG_RGB.shape[0],
            0,
        )

        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="upper right")

    # ========================================================
    # Animation
    # ========================================================

    ani = FuncAnimation(
        fig,
        update,
        frames=len(results),
        interval=25,
        repeat=False,
    )

    plt.show()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()