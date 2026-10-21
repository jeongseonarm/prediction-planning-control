"""Closed loop: prediction -> local planning -> MPC -> vehicle model."""

from dataclasses import dataclass, field

import numpy as np

from control.models.kinematic_bicycle import KinematicBicycle
from planning.local_path_planning.frenet_local_planner import (
    FrenetLocalPlanner,
    resample_reference,
)


@dataclass
class SimulationResult:
    frames: list = field(default_factory=list)          # simulated dataset frames
    tracks: list = field(default_factory=list)          # per frame: predictor output
    ego_states: np.ndarray = None                       # (K, 6) x, y, yaw, v, a, steer
    planned_xy: list = field(default_factory=list)      # per frame: (n, 2) planned path
    planner_status: list = field(default_factory=list)  # per frame: "ok", "stop", ...
    goal_reached: bool = False


def build_controller(cfg, vehicle, a_limit):
    """Controller named by cfg.controller, with the planner's acceleration limit."""
    # imported on use: only the LTV-MPC needs osqp
    if cfg.controller == "ltv":
        from control.controllers.ltv_mpc import LTVMPCController

        controller = LTVMPCController(model=vehicle, horizon=cfg.mpc_horizon)
    elif cfg.controller == "nmpc":
        from control.controllers.nmpc_slsqp import MPCController

        controller = MPCController(model=vehicle, horizon=cfg.mpc_horizon)
    else:
        raise ValueError(f"Unknown controller '{cfg.controller}' (use 'ltv' or 'nmpc').")

    # the MPC must be able to follow what the planner may ask for
    controller.a_min = -a_limit
    controller.a_max = a_limit
    return controller


def build_planner(cfg, world):
    planner_config = cfg.planner_config()
    checker = world.collision_checker

    planner = FrenetLocalPlanner(
        config=planner_config,
        is_collision_fn=checker.check,
        is_collision_many_fn=checker.check_many,
    )
    planner.set_waypoints(world.global_path)
    return planner, planner_config


def _plan_inputs(tracks):
    """Predicted paths, sizes and headings of the tracks that have a path."""
    used = [item for item in tracks if item["plan_traj"] is not None]
    return (
        [item["plan_traj"] for item in used],
        [item["size"] for item in used],
        [item["heading"] for item in used],
    )


def run_simulation(cfg, world, predictor, keep_tracks=True, progress=True):
    """
    Drive the ego vehicle from cfg.ego_start towards cfg.ego_goal.

    Per frame: predict the surrounding vehicles, plan a Frenet trajectory,
    resample it to the MPC step, compute one control and advance the model.
    The planner, controller and vehicle are created once so that warm starts
    and the previous planned trajectory carry over between frames.
    """
    planner, planner_config = build_planner(cfg, world)

    vehicle = KinematicBicycle(wheelbase=cfg.wheelbase, dt=cfg.dt)
    controller = build_controller(cfg, vehicle, planner_config.max_accel)

    frames = predictor.frames(cfg.start_frame, cfg.end_frame)

    if progress:
        from tqdm import tqdm
        frames = tqdm(frames, desc="Navigation Ego Car")

    goal_xy = np.asarray(cfg.ego_goal[:2])
    goal_yaw = cfg.ego_goal[2]

    ego_state = np.array([*cfg.ego_start, cfg.initial_speed, 0.0, 0.0])
    ego_states = []
    result = SimulationResult()

    for frame in frames:
        tracks = predictor.predict(frame)

        result.frames.append(frame)
        if keep_tracks:
            result.tracks.append(tracks)

        predictions, sizes, headings = _plan_inputs(tracks)
        planned = planner.plan(ego_state, predictions, sizes, headings)

        result.planned_xy.append(np.column_stack((planned["x"], planned["y"])))
        result.planner_status.append(planned["status"])
        ego_states.append(ego_state.copy())

        # sample the plan at the MPC time step: reference[k] <-> t = k * dt
        reference = resample_reference(planned, cfg.dt, cfg.mpc_horizon + 1)

        # MPC state = [x, y, theta, v]
        state_mpc = ego_state[:4]
        control = controller.compute_control(state_mpc, reference)
        state_mpc = vehicle.update(state_mpc, control)
        ego_state = np.concatenate((state_mpc, control))

        position_error = np.linalg.norm(ego_state[:2] - goal_xy)
        yaw_error = np.arctan2(
            np.sin(ego_state[2] - goal_yaw), np.cos(ego_state[2] - goal_yaw)
        )

        if (
            position_error < cfg.goal_position_tolerance
            and abs(yaw_error) < cfg.goal_yaw_tolerance
        ):
            ego_states.append(ego_state.copy())
            result.goal_reached = True

            print("Goal reached.")
            print("Position Error:", position_error)
            print("Yaw Error [deg]:", np.rad2deg(yaw_error))
            break

    result.ego_states = np.asarray(ego_states)
    return result
