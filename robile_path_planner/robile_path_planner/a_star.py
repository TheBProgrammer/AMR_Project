import numpy as np
import heapq
import cv2

class AStarPlanner:
    def __init__(self, grid_map, resolution=0.05, origin=(0.0, 0.0)):
        self.grid_map = grid_map
        self.rows, self.cols = grid_map.shape
        self.resolution = resolution
        self.origin = origin

        # Distance map from obstacles (higher = safer)
        self.distance_map = self.compute_clearance_map(grid_map)

    def compute_clearance_map(self, grid_map):
        """
        Uses distance transform to compute how far each free cell is from an obstacle.
        """
        # Obstacles are 1, free space is 0
        binary_obstacles = np.where(grid_map == 0, 0, 1).astype(np.uint8)
        dist_transform = cv2.distanceTransform(binary_obstacles, cv2.DIST_L2, 5)

        # Normalize so values are [0,1], then scale: higher = more clear space
        max_dist = np.max(dist_transform)
        return dist_transform / (max_dist + 1e-5)

    def heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node):
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid_map[nr, nc] == 0:
                    move_cost = np.hypot(dr, dc)
                    neighbors.append(((nr, nc), move_cost))
        return neighbors

    def grid_to_world(self, grid_coord):
        x = grid_coord[1] * self.resolution + self.origin[0]
        y = grid_coord[0] * self.resolution + self.origin[1]
        return (x, y)

    def world_to_grid(self, world_coord):
        col = int((world_coord[0] - self.origin[0]) / self.resolution)
        row = int((world_coord[1] - self.origin[1]) / self.resolution)
        return (row, col)

    def plan(self, start_world, goal_world):
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)

        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols):
            raise ValueError(f"Goal position {goal} is out of bounds")
        if self.grid_map[start] != 0:
            raise ValueError(f"Start position {start} is in an obstacle")
        if self.grid_map[goal] != 0:
            raise ValueError(f"Goal position {goal} is in an obstacle")

        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start, None))
        came_from = {}
        cost_so_far = {start: 0}

        while open_set:
            _, cost, current, parent = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return [self.grid_to_world(p) for p in path]

            for neighbor, move_cost in self.get_neighbors(current):
                # Safety boost from distance map (higher = safer)
                safety_weight = 1.0 - self.distance_map[neighbor[0], neighbor[1]]
                weighted_cost = move_cost * (1.0 + 8.0 * safety_weight)  # penalty when close to obstacles

                new_cost = cost + weighted_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, current))
                    came_from[neighbor] = current
        return None