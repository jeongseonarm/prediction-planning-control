import numpy as np


# ============================================================
# Fixed Colors
# ============================================================

VEHICLE_COLOR = (1.0, 0.6, 0.2)
INPUT_COLOR = (0.1, 0.85, 0.1)
PRED_COLOR = (0.1, 0.1, 0.85)
GT_COLOR = (0.85, 0.1, 0.1)


# ============================================================
# Draw Trajectory
# ============================================================

def Draw_trajectory(ax, Track_data, GT_traj, Pred_traj, M2P):

    Input_traj = Track_data[['xCenter', 'yCenter']].to_numpy()

    Input_img = Input_traj * M2P
    Input_img[:, 1] = -Input_img[:, 1]

    Pred_img = Pred_traj * M2P
    Pred_img[:, 1] = -Pred_img[:, 1]

    GT_img = GT_traj * M2P
    GT_img[:, 1] = -GT_img[:, 1]


    # ========================================================
    # Historical trajectory
    # ========================================================

    ax.plot(
        Input_img[:, 0],
        Input_img[:, 1],
        color=INPUT_COLOR,
        linewidth= 1.5
    )


    # ========================================================
    # Prediction
    # ========================================================

    ax.plot(
        Pred_img[:, 0],
        Pred_img[:, 1],
        color=PRED_COLOR,
        linewidth= 1.5
    )


    # ========================================================
    # Ground Truth
    # ========================================================

    ax.plot(
        GT_img[:, 0],
        GT_img[:, 1],
        color=GT_COLOR,
        linewidth= 1.5
    )


    # ========================================================
    # Current position
    # ========================================================

    ax.scatter(
        Input_img[-1, 0],
        Input_img[-1, 1],
        color= INPUT_COLOR,
        s=10
    )


    # ========================================================
    # Prediction endpoint
    # ========================================================

    ax.scatter(
        Pred_img[-1, 0],
        Pred_img[-1, 1],
        color=PRED_COLOR,
        s=10
    )


    # ========================================================
    # Ground Truth endpoint
    # ========================================================

    ax.scatter(
        GT_img[-1, 0],
        GT_img[-1, 1],
        color=GT_COLOR,
        s=10
    )
