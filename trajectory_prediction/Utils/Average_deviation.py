import numpy as np
import torch

def Average_deviation(GT_trajectory_x, GT_trajectory_y, Pred_trajectory_x, Pred_trajectory_y):
    # Check if the input is a tensor or numpy array
    if isinstance(GT_trajectory_x, torch.Tensor):
        GT_trajectory_x = GT_trajectory_x.detach().cpu().numpy()
        GT_trajectory_y = GT_trajectory_y.detach().cpu().numpy()
        Pred_trajectory_x = Pred_trajectory_x.detach().cpu().numpy()
        Pred_trajectory_y = Pred_trajectory_y.detach().cpu().numpy()
    
    # Calculate R (GT trajectory)
    R_x = GT_trajectory_x[1:] - GT_trajectory_x[:-1]  # R's x component
    R_y = GT_trajectory_y[1:] - GT_trajectory_y[:-1]  # R's y component

    R_magnitude = np.sqrt(R_x**2 + R_y**2)
    epsilon = 1e-8
    R_magnitude = R_magnitude + epsilon  # Add a small value to avoid division by zero

    # Calculate unit vectors
    R_x_unit = R_x / R_magnitude
    R_y_unit = R_y / R_magnitude

    # Calculate the difference between the predicted and ground truth values
    diff_x = Pred_trajectory_x - GT_trajectory_x
    diff_y = Pred_trajectory_y - GT_trajectory_y
    diff_x = diff_x[:-1]
    diff_y = diff_y[:-1]

    # Calculate Lateral Deviation
    Deviation_lateral = abs(-R_y_unit * diff_x + R_x_unit * diff_y)
    Average_Deviation_lateral = np.mean(Deviation_lateral)

    return Average_Deviation_lateral

# Reference : Zhang, Q., Hu, S., Sun, J., Chen, Q. A., & Mao, Z. M. (2022). On adversarial robustness of trajectory prediction for autonomous vehicles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 15159-15168).
