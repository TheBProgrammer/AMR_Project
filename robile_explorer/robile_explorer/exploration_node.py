"""
ROS 2 Node for autonomous frontier-based exploration:
    - Detects unknown frontiers in occupancy map
    - Clusters them via DBSCAN
    - Selects the most informative cluster
    - Projects a reachable free-space goal
    - Publishes PoseStamped goal
    - Publishes visual markers
    - Retries alternate clusters when goal projection fails
    - Stops when map is fully explored

Strategy Type	        alpha	beta	    Outcome
Balanced exploration 	1.5	    1.0	    Reasonably informative & reachable
Greedy for info	        3.0	    0.5	    Picks big info frontiers, even if far
Conservative explorer	1.0	    2.0	    Stays close to home, works well in small environments
Random-explorer-lite	1.0	    0.0	    Ignores distance, can help jump-start
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from sklearn.cluster import DBSCAN
from collections import deque
from tf_transformations import quaternion_from_euler
import cv2


class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        self.declare_parameter('alpha', 1.5)
        self.declare_parameter('beta', 1.0)
        self.declare_parameter('explored_ratio', 0.99)
        self.declare_parameter('inflation_radius_m', 0.8)

        self.pose = None
        self.map = None
        self.grid = None
        self.inflated_map = None
        self.goal_reached = True
        self.failed_centroids = []

        self.save_map_client = self.create_client(Empty, '/slam_toolbox/save_map')

        self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/frontier_clusters', 10)
        self.projection_pub = self.create_publisher(MarkerArray, '/centroid_projection_markers', 10)

        self.timer = self.create_timer(1.0, self.explore_callback)

    def pose_callback(self, msg):
        self.pose = msg.pose.pose

    def goal_reached_callback(self, msg: Bool):
        self.goal_reached = msg.data

    # === Inflated Map ===
    def map_callback(self, msg):
        try:
            self.map = msg
            width, height = msg.info.width, msg.info.height
            resolution = msg.info.resolution
            grid = np.array(msg.data, dtype=np.int8).reshape((height, width))

            # Inflate obstacles
            binary_map = np.zeros_like(grid, dtype=np.uint8)
            binary_map[grid == 100] = 1  # real obstacles only
            inflation_radius = self.get_parameter('inflation_radius_m').get_parameter_value().double_value
            kernel_radius = max(1, int(inflation_radius / resolution))
            kernel = np.ones((2 * kernel_radius + 1, 2 * kernel_radius + 1), dtype=np.uint8)
            self.inflated_map = cv2.dilate(binary_map, kernel)

            self.grid = grid
        except Exception as e:
            self.get_logger().error(f"Map error: {e}")

    # === Exploration Loop ===
    def explore_callback(self):
        if self.pose is None or self.grid is None:
            self.get_logger().info("Waiting for map and pose...")
            return

        if not self.goal_reached:
            self.get_logger().info("Waiting for previous goal...")
            return

        # Convert robot position to grid
        width, height = self.grid.shape[1], self.grid.shape[0]
        resolution = self.map.info.resolution
        origin = self.map.info.origin
        rx = int((self.pose.position.x - origin.position.x) / resolution)
        ry = int((self.pose.position.y - origin.position.y) / resolution)
        rx, ry = np.clip(rx, 0, width - 1), np.clip(ry, 0, height - 1)

        # === Compute reachable explored ratio (flood fill) ===
        ratio = self.compute_explored_ratio(self.grid, (rx, ry))
        
        explored_ratio_param = self.get_parameter('explored_ratio').get_parameter_value().double_value
        if ratio >= explored_ratio_param:
            self.get_logger().info(f"Fully explored ({ratio * 100:.2f}%) — saving map.")
            self.save_map()
            rclpy.shutdown()
            return

        # === Frontier detection ===
        frontiers = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if self.grid[y, x] == -1 and np.any(self.grid[y-1:y+2, x-1:x+2] == 0):
                    frontiers.append((x, y))

        if not frontiers:
            self.get_logger().warn("No frontiers found.")
            return

        frontier_points = np.array(frontiers)
        clustering = DBSCAN(eps=3, min_samples=5).fit(frontier_points)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(frontier_points[i])

        self.publish_markers(clusters, resolution, origin)

        alpha = self.get_parameter('alpha').get_parameter_value().double_value
        beta = self.get_parameter('beta').get_parameter_value().double_value

        evaluated = []
        max_info = max_dist = 1e-5
        for cluster in clusters.values():
            f_pts = [pt for pt in cluster if np.any(self.grid[pt[1]-1:pt[1]+2, pt[0]-1:pt[0]+2] == 0)]
            if not f_pts:
                continue
            cx, cy = np.mean(f_pts, axis=0)
            gcx, gcy = int(cx), int(cy)
            dist = np.linalg.norm([gcx - rx, gcy - ry])
            window = self.grid[max(0, gcy-5):gcy+6, max(0, gcx-5):gcx+6]
            info = np.count_nonzero(window == -1)

            max_info = max(max_info, info)
            max_dist = max(max_dist, dist)
            evaluated.append((cluster, (cx, cy), info, dist))

        if not evaluated:
            self.get_logger().warn("All clusters filtered.")
            return

        goal_found = False
        for cluster, (cx, cy), info, dist in sorted(
            evaluated, key=lambda t: -(alpha * t[2]/max_info - beta * t[3]/max_dist)
        ):
            gcx, gcy = int(round(cx)), int(round(cy))
            if any(np.linalg.norm(np.array([gcx, gcy]) - np.array(f)) < 5 for f in self.failed_centroids):
                continue  # This cluster failed before

            target = self.project_to_free_space(self.inflated_map, (gcx, gcy))
            if target:
                gx, gy = target
                wx = origin.position.x + gx * resolution
                wy = origin.position.y + gy * resolution

                # Quaternion from yaw toward target
                dx = wx - self.pose.position.x
                dy = wy - self.pose.position.y
                yaw = np.arctan2(dy, dx)
                qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)

                msg = PoseStamped()
                msg.header.frame_id = 'map'
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.position.x = wx
                msg.pose.position.y = wy
                msg.pose.orientation.x = qx
                msg.pose.orientation.y = qy
                msg.pose.orientation.z = qz
                msg.pose.orientation.w = qw
                self.goal_pub.publish(msg)
                self.pose_published = msg
                self.goal_reached = False
                self.get_logger().info(f"[Goal] Sent robot to ({wx:.2f}, {wy:.2f}) | Score = {info}")
                self.publish_projection_markers((cx, cy), (gx, gy), resolution, origin)
                goal_found = True
                break
            else:
                self.failed_centroids.append((gcx, gcy))
                if len(self.failed_centroids) > 30:
                    self.failed_centroids.pop(0)

        if not goal_found:
            self.get_logger().warn("No reachable goals from any cluster.")

    # === Explored Map using Flood Fill ===
    def compute_explored_ratio(self, grid, start_cell):
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        queue = deque([start_cell])
        explored = 0
        reachable = 0

        while queue:
            x, y = queue.popleft()
            if not (0 <= x < w and 0 <= y < h):
                continue
            if visited[y, x] or grid[y, x] == 100:
                continue

            visited[y, x] = True
            reachable += 1
            if grid[y, x] != -1:
                explored += 1

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                queue.append((x+dx, y+dy))

        total_explorable = np.count_nonzero(grid != 100)  # all non-wall areas (free + unknown)
        actual_explored = np.count_nonzero(grid == 0)     # actual free cells (explored)

        self.get_logger().info(f"Reachable: {reachable}, Free: {actual_explored}, Ratio: {actual_explored / max(reachable, 1):.2f}")

        return actual_explored / total_explorable
    
    # === Spiral Projection ===
    def project_to_free_space(self, inflated_map, start, max_steps=100):
        height, width = inflated_map.shape
        visited = set()
        queue = deque([start])
        while queue and max_steps > 0:
            x, y = queue.popleft()
            max_steps -= 1
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if 0 <= x < width and 0 <= y < height and inflated_map[y, x] == 0:
                return (x, y)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    queue.append((x + dx, y + dy))
        return None

    # === Visualisation ===
    def publish_markers(self, clusters, resolution, origin):
        marker_array = MarkerArray()
        for i, points in enumerate(clusters.values()):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontiers'
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = 0.05
            marker.color.r = float(np.random.rand())
            marker.color.g = float(np.random.rand())
            marker.color.b = float(np.random.rand())
            marker.color.a = 1.0
            for p in points:
                marker.points.append(Point(
                    x=origin.position.x + p[0] * resolution,
                    y=origin.position.y + p[1] * resolution,
                    z=0.05
                ))
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def publish_projection_markers(self, centroid, projected, resolution, origin):
        marker_array = MarkerArray()

        # Red = centroid, Green = projected
        for point, color, ns, mid in zip([centroid, projected], [(1, 0, 0), (0, 1, 0)], ["centroid", "goal"], [0, 1]):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = mid
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.25
            marker.pose.position.x = origin.position.x + point[0] * resolution
            marker.pose.position.y = origin.position.y + point[1] * resolution
            marker.pose.orientation.w = 1.0
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        self.projection_pub.publish(marker_array)

    def save_map(self):
        if self.save_map_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Saving final map...")
            self.save_map_client.call_async(Empty.Request())
        else:
            self.get_logger().warn("Map save service not available.")


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()