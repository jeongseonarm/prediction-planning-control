"""
Scenario configuration: every number that main.py used to hard-code.

Positions are in metres in the inD frame (x right, y up, origin at the top-left
of the background image, so image rows are -y * m2p).
"""

import os
from dataclasses import dataclass

import numpy as np

from planning.local_path_planning.frenet_local_planner import PlannerConfig

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ScenarioConfig:
    # ---- data ---------------------------------------------------------
    data_num: int = 18
    m2p: float = 10.2                       # metres to pixels of the background image
    prediction_dir: str = os.path.join(ROOT_DIR, "trajectory_prediction")
    mask_dir: str = os.path.join(ROOT_DIR, "map", "drivable_mask_images")

    # ---- ego vehicle --------------------------------------------------
    ego_start: tuple = (80.0, -25.0, float(np.deg2rad(-168.69)))   # x, y, yaw
    ego_goal: tuple = (20.0, -37.0, float(np.deg2rad(-168.69)))
    initial_speed: float = 5.0              # [m/s]
    ego_length: float = 2.5
    ego_width: float = 1.8
    wheelbase: float = 2.0
    goal_position_tolerance: float = 1.0    # [m]
    goal_yaw_tolerance: float = float(np.deg2rad(10.0))

    # ---- runtime timing ----------------------------------------------
    # `dt` remains the MPC/control sampling interval for compatibility.
    dt: float = 0.04                        # controller period [s]
    simulation_period: float = 0.04         # vehicle model update period [s]
    prediction_period: float = 0.04         # independent prediction worker [s] (25 Hz)
    planner_period: float = 0.10            # independent local-planner worker [s]
    max_prediction_age: float = 0.25        # discard older traffic predictions [s]
    max_plan_age: float = 0.25              # reject an old local plan [s]
    max_control_age: float = 0.12           # stale command triggers braking [s]
    emergency_brake_decel: float = 3.0      # stale-data fallback [m/s^2]
    execution_mode: str = "multirate"       # "multirate" or legacy "sequential"

    # ---- control ------------------------------------------------------
    mpc_horizon: int = 25                   # steps
    controller: str = "ltv"                 # "ltv" (OSQP) or "nmpc" (SLSQP, reference)

    # ---- global path (Hybrid A*) ---------------------------------------
    map_margin: float = 0.1                 # collision-checker body margin [m]
    max_steer_deg: float = 30.0
    astar_step_size: float = 0.75           # distance travelled per expansion [m]
    astar_motion_resolution: float = 0.1    # integration / collision-check spacing [m]
    astar_xy_res: float = 0.5
    astar_yaw_res_deg: float = 10.0
    astar_steer_samples: int = 15
    astar_steer_change_weight: float = 0.1  # cost weight per radian of steer change

    # ---- prediction ---------------------------------------------------
    input_frame: int = 25
    output_frame: int = 75
    frame_dt: float = 0.04                  # time between two dataset frames [s]
    start_frame: int = 300
    end_frame: int = 500

    # ---- local planner ------------------------------------------------
    target_speed_kmh: float = 10.0
    max_speed_kmh: float = 30.0
    max_accel: float = 3.0
    safety_margin: float = 0.3
    # d < 0: right of the global path (own lane), d > 0: left (oncoming lane).
    # The drivable mask is very wide (sidewalks included), so the lane side is
    # limited here, not by the mask.
    lateral_offsets: tuple = tuple(np.linspace(-3.0, 1.0, 9))
    target_speed_step: float = 1.0
    n_speed_samples: int = 5

    def __post_init__(self):
        for name in (
            "dt", "simulation_period", "frame_dt", "prediction_period", "planner_period",
            "max_prediction_age", "max_plan_age", "max_control_age",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.execution_mode not in ("multirate", "sequential"):
            raise ValueError("execution_mode must be 'multirate' or 'sequential'.")
        if (
            not np.isfinite(self.emergency_brake_decel)
            or self.emergency_brake_decel <= 0
        ):
            raise ValueError("emergency_brake_decel must be finite and positive.")

    # ---- derived paths --------------------------------------------------
    @property
    def data_path(self):
        return os.path.join(
            self.prediction_dir, "Data", "18_tracks_del_vru_static_maneuver.csv"
        )

    @property
    def log_dir(self):
        return os.path.join(self.prediction_dir, "log")

    @property
    def background_path(self):
        return os.path.join(
            self.prediction_dir, "Data", "inD_backgroud_images",
            f"{self.data_num:02d}_background.png",
        )

    @property
    def mask_path(self):
        return os.path.join(self.mask_dir, f"{self.data_num:02d}_mask.png")

    def planner_config(self):
        return PlannerConfig(
            wheelbase=self.wheelbase,
            prediction_dt=self.frame_dt,
            prediction_t0=self.frame_dt,
            ego_length=self.ego_length,
            ego_width=self.ego_width,
            target_speed=self.target_speed_kmh / 3.6,
            max_speed=self.max_speed_kmh / 3.6,
            max_accel=self.max_accel,
            safety_margin=self.safety_margin,
            lateral_offsets=tuple(self.lateral_offsets),
            target_speed_step=self.target_speed_step,
            n_speed_samples=self.n_speed_samples,
            # the MPC only needs mpc_horizon * dt seconds of reference
            min_reuse_points=int(np.ceil(self.mpc_horizon * self.dt / 0.1)) + 1,
        )

