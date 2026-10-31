import heapq
import math
import numpy as np


class Node:

    def __init__(
        self,
        x_idx: int,
        y_idx: int,
        yaw_idx: int,
        state: np.ndarray,
        g_cost: float,
        f_cost: float,
        parent_key=None,
        steer: float = 0.0,
        steer_idx: int = -1,
    ):
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.yaw_idx = yaw_idx
        self.state = np.array(state)  # [x, y, theta]
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.parent_key = parent_key
        self.steer = steer
        self.steer_idx = steer_idx

    @property
    def key(self):
        return (self.x_idx, self.y_idx, self.yaw_idx, self.steer_idx)


def calc_index(state, map_bounds, xy_res, yaw_res, yaw_bins):
    x, y, theta = state[:3]
    x_min, _, y_min, _ = map_bounds

    x_idx = int(np.round((x - x_min) / xy_res))
    y_idx = int(np.round((y - y_min) / xy_res))

    norm_yaw = (theta + np.pi * 2) % (np.pi * 2)
    yaw_idx = int(np.round(norm_yaw / yaw_res)) % yaw_bins

    return x_idx, y_idx, yaw_idx


def heuristic(state, goal):
    """Euclidean lower bound; vehicle feasibility is handled by motion primitives."""
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    return np.hypot(dx, dy)


def hybrid_a_star(
    start: list,
    goal: list,
    map_bounds: tuple,
    is_collision_fn,
    wheelbase: float = 2.7,
    max_steer: float = np.deg2rad(30.0),
    step_size: float = 1.5,
    xy_res: float = 1.0,
    yaw_res: float = np.deg2rad(15.0),
    steer_samples: int = 5,
    motion_resolution: float = 0.1,
    steer_change_weight: float = 0.1,
):
    """
    Args:
        start: [x, y, yaw] (m, rad)
        goal: [x, y, yaw] (m, rad)
        map_bounds: (x_min, x_max, y_min, y_max)
        is_collision_fn: callback fn(x, y, yaw) -> bool, True on collision
        wheelbase: vehicle wheelbase (m)
        max_steer: maximum steering angle (rad)
        step_size: travelled distance per expansion (m)
        xy_res: position grid resolution (m)
        yaw_res: heading grid resolution (rad)
        steer_samples: number of steering branches
        motion_resolution: maximum distance between motion/collision samples (m)
        steer_change_weight: cost weight for steering change (cost/rad)
    """
    if wheelbase <= 0.0:
        raise ValueError("wheelbase must be positive")
    if step_size <= 0.0 or motion_resolution <= 0.0:
        raise ValueError("step_size and motion_resolution must be positive")
    if xy_res <= 0.0 or yaw_res <= 0.0:
        raise ValueError("xy_res and yaw_res must be positive")
    if steer_samples < 2:
        raise ValueError("steer_samples must be at least 2")
    if steer_change_weight < 0.0:
        raise ValueError("steer_change_weight must be non-negative")

    x_min, x_max, y_min, y_max = map_bounds
    yaw_bins = max(1, int(np.round(2 * np.pi / yaw_res)))

    start_state = np.array(start[:3], dtype=float)
    goal_state = np.array(goal[:3], dtype=float)

    start_state[2] = (start_state[2] + np.pi) % (2 * np.pi) - np.pi
    goal_state[2] = (goal_state[2] + np.pi) % (2 * np.pi) - np.pi

    s_idx = calc_index(start_state, map_bounds, xy_res, yaw_res, yaw_bins)

    start_node = Node(
        *s_idx,
        state=start_state,
        g_cost=0.0,
        f_cost=heuristic(start_state, goal_state),
        steer_idx=-1,  # no incoming steering command at the initial state
    )

    open_set = {start_node.key: start_node}
    closed_set = set()
    queue = []
    heapq.heappush(queue, (start_node.f_cost, start_node.g_cost, start_node.key))

    steer_inputs = np.linspace(-max_steer, max_steer, steer_samples)
    goal_node = None

    while queue:
        _, queued_g_cost, curr_key = heapq.heappop(queue)
        curr_node = open_set.get(curr_key)

        # Ignore obsolete heap entries left behind after a cheaper route was found.
        if curr_node is None or queued_g_cost > curr_node.g_cost + 1e-12:
            continue
        if curr_key in closed_set:
            continue
        closed_set.add(curr_key)

        dist = np.hypot(
            curr_node.state[0] - goal_state[0],
            curr_node.state[1] - goal_state[1],
        )
        yaw_err = abs(
            (curr_node.state[2] - goal_state[2] + np.pi) % (2 * np.pi) - np.pi
        )

        if dist < (xy_res * 1.2) and yaw_err < (yaw_res * 1.2):
            goal_node = curr_node
            break

        # Integrate each steering primitive in short increments. Check the
        # map bounds and vehicle collision at every intermediate pose.
        num_steps = int(math.ceil(step_size / motion_resolution))
        ds = step_size / num_steps

        for steer_idx, steer in enumerate(steer_inputs):
            x, y, yaw = curr_node.state
            collision = False

            for _ in range(num_steps):
                x += ds * np.cos(yaw)
                y += ds * np.sin(yaw)
                yaw += (ds / wheelbase) * np.tan(steer)
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    collision = True
                    break
                if is_collision_fn(x, y, yaw):
                    collision = True
                    break

            if collision:
                continue

            new_state = np.array([x, y, yaw])
            n_idx = calc_index(new_state, map_bounds, xy_res, yaw_res, yaw_bins)
            # Steering history affects the next transition cost, so it is
            # part of the discretized search state as well.
            n_key = (*n_idx, steer_idx)

            if n_key in closed_set:
                continue

            distance_cost = step_size
            steering_cost = abs(steer) * 0.1
            steering_change_cost = (
                abs(steer - curr_node.steer) * steer_change_weight
            )
            g_cost = (
                curr_node.g_cost
                + distance_cost
                + steering_cost
                + steering_change_cost
            )
            f_cost = g_cost + heuristic(new_state, goal_state)

            if n_key not in open_set or g_cost < open_set[n_key].g_cost:
                node = Node(
                    *n_idx,
                    state=new_state,
                    g_cost=g_cost,
                    f_cost=f_cost,
                    parent_key=curr_key,
                    steer=steer,
                    steer_idx=steer_idx,
                )
                open_set[n_key] = node
                heapq.heappush(queue, (f_cost, g_cost, n_key))

    if goal_node is None:
        return None, np.inf

    # Path Backtracking
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr.state.tolist())
        curr = open_set.get(curr.parent_key, None)

    path.reverse()
    return path, goal_node.g_cost

