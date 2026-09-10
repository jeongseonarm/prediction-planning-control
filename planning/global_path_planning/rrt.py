import numpy as np


def rrt(
    map_generator,
    start,
    goal,
    max_iterations=5000,
    step_size=2.0,
    goal_sample_rate=0.1
):
    # ========================================================
    # Start / Goal
    # ========================================================
    start_x, start_y = start
    goal_x, goal_y = goal

    # ========================================================
    # Start / Goal Collision Check
    # ========================================================
    if map_generator.is_collision(start_x, start_y):
        return None, np.inf

    if map_generator.is_collision(goal_x, goal_y):
        return None, np.inf

    # ========================================================
    # Tree
    #
    # Node format:
    # [x, y, parent_index]
    # ========================================================
    tree = [(start_x, start_y, -1)]

    # ========================================================
    # RRT
    # ========================================================
    for _ in range(max_iterations):
        # ----------------------------------------------------
        # Random Sampling
        # ----------------------------------------------------
        if np.random.rand() < goal_sample_rate:
            sample_x = goal_x
            sample_y = goal_y
        else:
            sample_x = np.random.uniform(0, map_generator.width)
            sample_y = np.random.uniform(0, map_generator.height)

        # ----------------------------------------------------
        # Nearest Node
        # ----------------------------------------------------
        nearest_index = 0
        nearest_distance = np.inf

        for i, node in enumerate(tree):
            dx = sample_x - node[0]
            dy = sample_y - node[1]
            current_distance = dx * dx + dy * dy

            if current_distance < nearest_distance:
                nearest_distance = current_distance
                nearest_index = i

        nearest_x = tree[nearest_index][0]
        nearest_y = tree[nearest_index][1]

        # ----------------------------------------------------
        # Steer
        # ----------------------------------------------------
        dx = sample_x - nearest_x
        dy = sample_y - nearest_y
        distance = np.hypot(dx, dy)

        if distance == 0:
            continue

        if distance <= step_size:
            new_x = sample_x
            new_y = sample_y
        else:
            new_x = nearest_x + step_size * dx / distance
            new_y = nearest_y + step_size * dy / distance

        # ----------------------------------------------------
        # Collision Check
        # ----------------------------------------------------
        if map_generator.is_collision_line(nearest_x, nearest_y, new_x, new_y):
            continue

        # ----------------------------------------------------
        # Add Node
        # ----------------------------------------------------
        tree.append((new_x, new_y, nearest_index))
        new_index = len(tree) - 1

        # ----------------------------------------------------
        # Goal Connection
        # ----------------------------------------------------
        goal_distance = np.hypot(goal_x - new_x, goal_y - new_y)

        if goal_distance <= step_size:
            if not map_generator.is_collision_line(new_x, new_y, goal_x, goal_y):
                tree.append((goal_x, goal_y, new_index))
                goal_index = len(tree) - 1
                break
    else:
        # ====================================================
        # Failed
        # ====================================================
        return None, np.inf

    # ========================================================
    # Path Reconstruction
    # ========================================================
    path = []
    current_index = goal_index

    while current_index != -1:
        node = tree[current_index]
        path.append((node[0], node[1]))
        current_index = node[2]

    path.reverse()

    # ========================================================
    # Path Length
    # ========================================================
    path_length = 0.0

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        path_length += np.hypot(x2 - x1, y2 - y1)

    return path, path_length