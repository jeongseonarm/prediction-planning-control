"""Multi-rate, latest-value runtime for the inD closed-loop simulation.

Prediction, local planning, and control run as independent periodic workers.
Each worker consumes the newest available input and drops obsolete work instead
of building a backlog. The inD file acts as a replayed sensor feed; this module
does not connect to physical sensors or actuators.
"""

import asyncio
import math
import threading
import time

import numpy as np

from control.models.kinematic_bicycle import KinematicBicycle
from planning.local_path_planning.frenet_local_planner import resample_reference
from .runner import SimulationResult, _plan_inputs, build_controller, build_planner


class _LatestValue:
    """Thread-safe latest-value slot with its simulation-time timestamp."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._sim_time = None
        self._sequence = 0

    def publish(self, value, sim_time):
        with self._lock:
            self._value = value
            self._sim_time = float(sim_time)
            self._sequence += 1

    def read(self):
        with self._lock:
            return self._value, self._sim_time, self._sequence


class _RuntimeState:
    def __init__(self):
        self.clock = _LatestValue()
        self.frame = _LatestValue()
        self.ego = _LatestValue()
        self.prediction = _LatestValue()
        self.plan = _LatestValue()
        self.control = _LatestValue()


def _shift_predictions(tracks, age, frame_dt):
    """Move predicted traffic paths forward to the planner's current time."""
    shifted = []
    for item in tracks:
        copied = dict(item)
        path = item.get("plan_traj")
        if path is not None and len(path):
            path = np.asarray(path, dtype=float)
            source_t = np.arange(1, len(path) + 1, dtype=float) * frame_dt
            query_t = age + source_t
            copied["plan_traj"] = np.column_stack(
                [np.interp(query_t, source_t, path[:, axis]) for axis in range(2)]
            )
        shifted.append(copied)
    return shifted


def _brake_control(cfg, ego_state):
    """Conservative command used until fresh planner/controller data exists."""
    steer = float(ego_state[5]) if len(ego_state) > 5 else 0.0
    return np.array([-min(cfg.emergency_brake_decel, cfg.max_accel), steer])


def _make_metrics():
    return {
        name: {
            "calls": 0,
            "errors": 0,
            "stale_input_skips": 0,
            "discarded_results": 0,
            "deadline_misses": 0,
            "last_compute_ms": None,
            "max_compute_ms": 0.0,
            "last_error": None,
        }
        for name in ("prediction", "planner", "controller")
    }


async def run_multirate_simulation(
    cfg, world, predictor, keep_tracks=True, progress=True
):
    """Run dataset replay, vehicle simulation and independent async workers.

    The vehicle model advances at ``cfg.simulation_period``. The MPC keeps its
    existing ``cfg.dt`` model step; prediction and planning use their own
    periods. Slow workers run in threads, while the plant holds the newest
    control command between updates (zero-order hold).
    """
    frames = predictor.frames(cfg.start_frame, cfg.end_frame)
    if not frames:
        raise ValueError("No dataset frames in the requested start/end range.")

    frame_times = np.asarray(
        [(frame - frames[0]) * cfg.frame_dt for frame in frames], dtype=float
    )
    duration = frame_times[-1] + cfg.frame_dt
    n_steps = max(1, int(math.ceil(duration / cfg.simulation_period)))

    planner, planner_config = build_planner(cfg, world)
    control_model = KinematicBicycle(wheelbase=cfg.wheelbase, dt=cfg.dt)
    vehicle = KinematicBicycle(
        wheelbase=cfg.wheelbase, dt=cfg.simulation_period
    )
    controller = build_controller(cfg, control_model, planner_config.max_accel)

    state = _RuntimeState()
    ego_state = np.array([*cfg.ego_start, cfg.initial_speed, 0.0, 0.0], dtype=float)
    state.clock.publish(0.0, 0.0)
    state.frame.publish(frames[0], 0.0)
    state.ego.publish(ego_state.copy(), 0.0)
    state.control.publish(
        {"u": _brake_control(cfg, ego_state), "source": "startup"}, 0.0
    )

    result = SimulationResult()
    ego_history = []
    metrics = _make_metrics()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    wall_origin = loop.time()
    goal_xy = np.asarray(cfg.ego_goal[:2], dtype=float)
    goal_yaw = float(cfg.ego_goal[2])
    frame_index = 0
    logged_frame_sequence = 0
    last_prediction_input_sequence = 0

    async def wait_until(deadline):
        delay = deadline - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

    async def periodic_worker(name, period, callback):
        deadline = wall_origin
        while not stop.is_set():
            await wait_until(deadline)
            if stop.is_set():
                break
            started = time.perf_counter()
            try:
                await callback()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep other workers and the plant alive
                metrics[name]["errors"] += 1
                metrics[name]["last_error"] = f"{type(exc).__name__}: {exc}"
            elapsed = (time.perf_counter() - started) * 1000.0
            metrics[name]["last_compute_ms"] = elapsed
            metrics[name]["max_compute_ms"] = max(
                metrics[name]["max_compute_ms"], elapsed
            )
            deadline += period
            now = loop.time()
            if deadline <= now:
                missed = int((now - deadline) // period) + 1
                metrics[name]["deadline_misses"] += missed
                deadline += missed * period

    async def predict_once():
        nonlocal last_prediction_input_sequence
        frame, frame_time, sequence = state.frame.read()
        if frame is None:
            return
        if sequence == last_prediction_input_sequence:
            return
        last_prediction_input_sequence = sequence
        metrics["prediction"]["calls"] += 1
        tracks = await asyncio.to_thread(predictor.predict, frame)
        current_time, _, _ = state.clock.read()
        if current_time - frame_time > cfg.max_prediction_age:
            metrics["prediction"]["discarded_results"] += 1
            return
        state.prediction.publish(tracks, frame_time)

    async def plan_once():
        ego, ego_time, _ = state.ego.read()
        tracks, prediction_time, _ = state.prediction.read()
        current_time, _, _ = state.clock.read()
        if ego is None or tracks is None or prediction_time is None:
            metrics["planner"]["stale_input_skips"] += 1
            return
        prediction_age = ego_time - prediction_time
        if prediction_age < -1e-6 or prediction_age > cfg.max_prediction_age:
            metrics["planner"]["stale_input_skips"] += 1
            return

        aligned_tracks = _shift_predictions(
            tracks, max(0.0, prediction_age), cfg.frame_dt
        )
        predictions, sizes, headings = _plan_inputs(aligned_tracks)
        metrics["planner"]["calls"] += 1
        planned = await asyncio.to_thread(
            planner.plan, ego.copy(), predictions, sizes, headings
        )
        current_time, _, _ = state.clock.read()
        if current_time - ego_time > cfg.max_plan_age:
            metrics["planner"]["discarded_results"] += 1
            return
        state.plan.publish(
            {"trajectory": planned, "source_time": ego_time}, current_time
        )

    async def control_once():
        ego, ego_time, _ = state.ego.read()
        plan, _, _ = state.plan.read()
        current_time, _, _ = state.clock.read()
        if ego is None:
            return

        if (
            plan is None
            or current_time - plan["source_time"] > cfg.max_plan_age
            or current_time - ego_time > cfg.max_control_age
        ):
            state.control.publish(
                {"u": _brake_control(cfg, ego), "source": "stale-plan"},
                current_time,
            )
            metrics["controller"]["stale_input_skips"] += 1
            return

        trajectory = plan["trajectory"]
        elapsed = max(0.0, current_time - plan["source_time"])
        reference = resample_reference(
            trajectory,
            cfg.dt,
            cfg.mpc_horizon + 1,
            start_time=elapsed,
        )
        metrics["controller"]["calls"] += 1
        control = await asyncio.to_thread(
            controller.compute_control, ego[:4].copy(), reference
        )

        current_time, _, _ = state.clock.read()
        if (
            current_time - ego_time > cfg.max_control_age
            or current_time - plan["source_time"] > cfg.max_plan_age
        ):
            metrics["controller"]["discarded_results"] += 1
            control = _brake_control(cfg, ego)
            source = "stale-result-brake"
        else:
            source = "mpc"
        state.control.publish({"u": np.asarray(control), "source": source}, current_time)

    workers = [
        asyncio.create_task(
            periodic_worker("prediction", cfg.prediction_period, predict_once),
            name="prediction-worker",
        ),
        asyncio.create_task(
            periodic_worker("planner", cfg.planner_period, plan_once),
            name="planner-worker",
        ),
        asyncio.create_task(
            periodic_worker("controller", cfg.dt, control_once),
            name="controller-worker",
        ),
    ]

    try:
        iterator = range(n_steps)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Multirate closed-loop simulation")

        for tick in iterator:
            sim_time = tick * cfg.simulation_period
            if sim_time >= duration:
                break
            await wait_until(wall_origin + sim_time)
            state.clock.publish(sim_time, sim_time)

            while (
                frame_index + 1 < len(frames)
                and frame_times[frame_index + 1] <= sim_time + 1e-9
            ):
                frame_index += 1
                state.frame.publish(frames[frame_index], frame_times[frame_index])

            frame_value, _, frame_sequence = state.frame.read()
            current_ego, _, _ = state.ego.read()
            if frame_sequence != logged_frame_sequence:
                logged_frame_sequence = frame_sequence
                result.frames.append(frame_value)
                ego_history.append(current_ego.copy())
                current_plan, _, _ = state.plan.read()
                if (
                    current_plan is not None
                    and sim_time - current_plan["source_time"] <= cfg.max_plan_age
                ):
                    trajectory = current_plan["trajectory"]
                    result.planned_xy.append(
                        np.column_stack((trajectory["x"], trajectory["y"]))
                    )
                    result.planner_status.append(trajectory["status"])
                else:
                    result.planned_xy.append(
                        np.column_stack((world.global_path[:, 0], world.global_path[:, 1]))
                    )
                    result.planner_status.append("waiting" if current_plan is None else "stale")
                if keep_tracks:
                    observed_tracks, prediction_time, _ = state.prediction.read()
                    if (
                        prediction_time is None
                        or sim_time - prediction_time > cfg.max_prediction_age
                    ):
                        observed_tracks = []
                    result.tracks.append(observed_tracks or [])

            if tick + 1 >= n_steps or sim_time + cfg.simulation_period >= duration:
                break

            await wait_until(wall_origin + sim_time + cfg.simulation_period)
            next_time = sim_time + cfg.simulation_period
            state.clock.publish(next_time, next_time)

            command, command_time, _ = state.control.read()
            ego_now, _, _ = state.ego.read()
            if command is None or next_time - command_time > cfg.max_control_age:
                control = _brake_control(cfg, ego_now)
                metrics["controller"]["stale_input_skips"] += 1
            else:
                control = command["u"]

            next_state = vehicle.update(ego_now[:4], control)
            ego_state = np.concatenate((next_state, np.asarray(control, dtype=float)))
            state.ego.publish(ego_state.copy(), next_time)

            position_error = np.linalg.norm(ego_state[:2] - goal_xy)
            yaw_error = np.arctan2(
                np.sin(ego_state[2] - goal_yaw),
                np.cos(ego_state[2] - goal_yaw),
            )
            if (
                position_error < cfg.goal_position_tolerance
                and abs(yaw_error) < cfg.goal_yaw_tolerance
            ):
                result.goal_reached = True
                print("Goal reached.")
                print("Position Error:", position_error)
                print("Yaw Error [deg]:", np.rad2deg(yaw_error))
                break

        result.ego_states = np.asarray(ego_history, dtype=float)
        # Fill visualization with the latest asynchronous result at return.
        latest_plan, _, _ = state.plan.read()
        if latest_plan and result.planned_xy:
            result.planned_xy[-1] = np.column_stack(
                (latest_plan["trajectory"]["x"], latest_plan["trajectory"]["y"])
            )
            result.planner_status[-1] = latest_plan["trajectory"]["status"]
        result.runtime_diagnostics = metrics
        return result
    finally:
        stop.set()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

