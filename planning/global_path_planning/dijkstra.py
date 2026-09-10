import heapq
import numpy as np


def dijkstra(grid, start, goal):
    rows, cols = grid.shape

    # ========================================================
    # Distance
    # ========================================================
    distance = np.full((rows, cols), np.inf)
    distance[start[1], start[0]] = 0

    # ========================================================
    # Parent
    # ========================================================
    parent = np.full((rows, cols, 2), -1, dtype=int)

    # ========================================================
    # Visited
    # ========================================================
    visited = np.zeros((rows, cols), dtype=bool)

    # ========================================================
    # Priority Queue
    # ========================================================
    queue = []
    heapq.heappush(queue, (0, start[0], start[1]))

    # ========================================================
    # Dijkstra
    # ========================================================
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        current_distance, x, y = heapq.heappop(queue)

        if visited[y, x]:
            continue

        visited[y, x] = True

        if (x, y) == goal:
            break

        # Neighbor 탐색
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= cols:
                continue

            if ny < 0 or ny >= rows:
                continue

            if grid[ny, nx] == 1:
                continue

            if visited[ny, nx]:
                continue

            # Distance Update
            new_distance = current_distance + 1

            if new_distance < distance[ny, nx]:
                distance[ny, nx] = new_distance
                parent[ny, nx] = [x, y]
                heapq.heappush(queue, (new_distance, nx, ny))

    # ========================================================
    # Goal Check
    # ========================================================
    if not visited[goal[1], goal[0]]:
        return None, np.inf

    # ========================================================
    # Path Reconstruction
    # ========================================================
    path = []
    current = goal

    while current != start:
        path.append(current)

        px = parent[current[1], current[0], 0]
        py = parent[current[1], current[0], 1]

        if px == -1:
            return None, np.inf

        current = (int(px), int(py))

    path.append(start)
    path.reverse()

    # ========================================================
    # Path Length
    # ========================================================
    path_length = distance[goal[1], goal[0]]

    return path, path_length