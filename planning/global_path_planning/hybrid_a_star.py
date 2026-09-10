import heapq
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
    ):
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.yaw_idx = yaw_idx
        self.state = np.array(state)  # [x, y, theta]
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.parent_key = parent_key
        self.steer = steer

    @property
    def key(self):
        return (self.x_idx, self.y_idx, self.yaw_idx)


def calc_index(state, map_bounds, xy_res, yaw_res, yaw_bins):
    x, y, theta = state[:3]
    x_min, _, y_min, _ = map_bounds

    x_idx = int(np.round((x - x_min) / xy_res))
    y_idx = int(np.round((y - y_min) / xy_res))

    norm_yaw = (theta + np.pi * 2) % (np.pi * 2)
    yaw_idx = int(np.round(norm_yaw / yaw_res)) % yaw_bins

    return x_idx, y_idx, yaw_idx


def heuristic(state, goal):
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    dyaw = abs((state[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi)
    return np.hypot(dx, dy) + dyaw * 1.5


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
):
    """
    Args:
        start: [x, y, yaw] (m, rad)
        goal: [x, y, yaw] (m, rad)
        map_bounds: (x_min, x_max, y_min, y_max)
        is_collision_fn: 콜백 함수, signature: fn(x, y, yaw) -> bool
        wheelbase: 차량 축거 (m)
        max_steer: 최대 조향각 (rad)
        step_size: 1 스텝당 이동 전진 거리 (m)
        xy_res: 위치 격자 버킷 해상도 (m)
        yaw_res: 헤딩 격자 버킷 해상도 (rad)
        steer_samples: 조향 분기 개수
    """
    x_min, x_max, y_min, y_max = map_bounds
    yaw_bins = int(np.round(2 * np.pi / yaw_res))

    start_state = np.array(start[:3])
    goal_state = np.array(goal[:3])

    s_idx = calc_index(start_state, map_bounds, xy_res, yaw_res, yaw_bins)

    start_node = Node(
        *s_idx,
        state=start_state,
        g_cost=0.0,
        f_cost=heuristic(start_state, goal_state),
    )

    open_set = {start_node.key: start_node}
    closed_set = set()
    queue = []
    heapq.heappush(queue, (start_node.f_cost, start_node.key))

    steer_inputs = np.linspace(-max_steer, max_steer, steer_samples)
    goal_node = None

    while queue:
        _, curr_key = heapq.heappop(queue)
        curr_node = open_set[curr_key]

        if curr_key in closed_set:
            continue
        closed_set.add(curr_key)

        # Goal Check
        dist = np.hypot(
            curr_node.state[0] - goal_state[0],
            curr_node.state[1] - goal_state[1],
        )
        yaw_err = abs(curr_node.state[2] - goal_state[2])

        if dist < (xy_res * 1.2) and yaw_err < (yaw_res * 1.2):
            goal_node = curr_node
            break

        # Node Expansion
        for steer in steer_inputs:
            x, y, yaw = curr_node.state
            new_x = x + step_size * np.cos(yaw)
            new_y = y + step_size * np.sin(yaw)
            new_yaw = yaw + (step_size / wheelbase) * np.tan(steer)
            new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi

            new_state = np.array([new_x, new_y, new_yaw])

            # 1. Bounds Check
            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                continue

            # 2. Collision Check 
            if is_collision_fn(new_x, new_y, new_yaw):
                continue

            n_idx = calc_index(
                new_state, map_bounds, xy_res, yaw_res, yaw_bins
            )
            n_key = n_idx

            if n_key in closed_set:
                continue

            steer_penalty = abs(steer) * 0.1
            g_cost = curr_node.g_cost + step_size + steer_penalty
            f_cost = g_cost + heuristic(new_state, goal_state)

            if n_key not in open_set or g_cost < open_set[n_key].g_cost:
                node = Node(
                    *n_idx,
                    state=new_state,
                    g_cost=g_cost,
                    f_cost=f_cost,
                    parent_key=curr_key,
                    steer=steer,
                )
                open_set[n_key] = node
                heapq.heappush(queue, (f_cost, n_key))

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