import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

import numpy as np
import cv2
import math  # Added this import

from robile_path_planner.a_star import AStarPlanner
from robile_path_planner.extract_waypoints import WaypointExtractor
from robile_path_planner.field_based_planner import Planner

class NavigationManagerNode(Node):
    def __init__(self):
        super().__init__('navigation_manager')
        self.get_logger().info("Navigation Manager Node is starting...")

        self.declare_parameter('localization_source', 'mcl')
        self.localization_source = self.get_parameter('localization_source').get_parameter_value().string_value

        self.map = None
        self.resolution = None
        self.origin = None
        self.goal_received = False
        self.start_received = False
        self.start_pose = None
        self.goal_pose = None
        self.goal_pose_orientation = None
        self.is_planning_now = False
        self.inflated_map = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        if self.localization_source == 'mcl':
            self.create_subscription(PoseStamped, '/mcl_pose', self.start_callback_mcl, 10)
            self.get_logger().info("Configured to use MCL for localization.")
        elif self.localization_source == 'amcl':
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.start_callback_amcl, qos)
            self.get_logger().info("Configured to use AMCL for localization.")
        elif self.localization_source == 'slam':
            self.create_subscription(PoseWithCovarianceStamped, '/pose', self.start_callback_slam, 10)
            self.get_logger().info("Configured to use SLAM Toolbox for localization.")
        elif self.localization_source == 'odom':
            self.create_subscription(Odometry, '/odom', self.start_callback_odom, 10)
            self.get_logger().info("Configured to use Odometry for localization.")

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)

        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.waypoint_pub = self.create_publisher(MarkerArray, '/waypoints', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.local_planner = Planner(self)
        self.local_planner.set_goal_reached_callback(self.waypoint_reached)

        self.current_waypoints = []
        self.current_waypoint_index = 0
        self.navigating = False

        self.create_timer(0.2, self.try_plan)
        self.create_timer(0.05, self.local_planner.planner_loop)

    def scan_callback(self, msg):
        self.local_planner.update_scan(msg)
        
    def map_callback(self, msg: OccupancyGrid):
        try:
            self.resolution = msg.info.resolution
            self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            width = msg.info.width
            height = msg.info.height
            data = np.array(msg.data).reshape((height, width))
            
            # self.get_logger().info(f"Map received. Dimensions: {width}x{height}, Resolution: {self.resolution:.3f} m/cell."))
            
            self.map = np.zeros_like(data, dtype=np.uint8)
            self.map[data == 100] = 1   # only known obstacles get inflated
            # self.map[data == -1] = 1  # unknown areas are not obstacles, useful for task 3

            robot_radius = 0.6
            robot_radius_cells = int(robot_radius / self.resolution)
            kernel = np.ones((2 * robot_radius_cells + 1, 2 * robot_radius_cells + 1), dtype=np.uint8)
            self.inflated_map = cv2.dilate(self.map, kernel, iterations=1)
            # self.get_logger().info("Map successfully loaded and inflated for path planning.")
            self.publish_costmap()

        except Exception as e:
            self.get_logger().error(f"Failed to process map: {e}")

    def publish_costmap(self):
        if self.inflated_map is not None:
            costmap_msg = OccupancyGrid()
            costmap_msg.header.frame_id = "map"
            costmap_msg.info.resolution = self.resolution
            costmap_msg.info.width = self.inflated_map.shape[1]
            costmap_msg.info.height = self.inflated_map.shape[0]
            costmap_msg.info.origin.position.x = self.origin[0]
            costmap_msg.info.origin.position.y = self.origin[1]
            
            flat_map = self.inflated_map.flatten()
            costmap_msg.data = [100 if x == 1 else 0 for x in flat_map]
            
            self.costmap_pub.publish(costmap_msg)
            # self.get_logger().info("Published inflated map as costmap.")
            
    def start_callback_amcl(self, msg: PoseWithCovarianceStamped):
        self.start_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if not self.start_received:
            self.start_received = True
            self.get_logger().info(f"Initial pose received from AMCL: ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f})")

    def start_callback_slam(self, msg: PoseWithCovarianceStamped):
        self.start_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if not self.start_received:
            self.start_received = True
            self.get_logger().info(f"Initial pose received from SLAM: ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f})")

    def start_callback_mcl(self, msg: PoseStamped):
        self.start_pose = (msg.pose.position.x, msg.pose.position.y)
        if not self.start_received:
            self.start_received = True
            self.get_logger().info(f"Initial pose received from MCL: ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f})")
    
    def start_callback_odom(self, msg: Odometry):
        try:
            # Transform from odom → map
            tf = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5))

            # Get current odometry position
            x_odom = msg.pose.pose.position.x
            y_odom = msg.pose.pose.position.y

            # Apply transform: new pose in /map
            x_map = tf.transform.translation.x + x_odom
            y_map = tf.transform.translation.y + y_odom

            self.start_pose = (x_map, y_map)

            if not self.start_received:
                self.start_received = True
                self.get_logger().info(
                    f"Transformed start pose from /odom to /map: ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f})"
                )

        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed in start_callback_odom: {e}")

    def goal_callback(self, msg: PoseStamped):
        new_goal_pos = (msg.pose.position.x, msg.pose.position.y)
        new_goal_quat = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([new_goal_quat.x, new_goal_quat.y, new_goal_quat.z, new_goal_quat.w])
        
        if self.goal_pose != new_goal_pos:
            self.goal_pose = new_goal_pos
            self.goal_pose_orientation = yaw
            self.goal_received = True
            self.get_logger().info(f"New navigation goal received: ({self.goal_pose[0]:.2f}, {self.goal_pose[1]:.2f}) with orientation {math.degrees(self.goal_pose_orientation):.2f} degrees.")
            
            if self.navigating:
                self.get_logger().info("Navigation interrupted. Re-planning for the new goal.")
                self.navigating = False

    def publish_path(self, path_world):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for x, y in path_world:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def publish_waypoints(self, waypoints):
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        self.waypoint_pub.publish(marker_array)

    def waypoint_reached(self):
        self.current_waypoint_index += 1
        
        if self.current_waypoint_index < len(self.current_waypoints):
            next_wp = self.current_waypoints[self.current_waypoint_index]
            self.get_logger().info(f"Waypoint {self.current_waypoint_index}/{len(self.current_waypoints)} reached. Moving to next waypoint: ({next_wp[0]:.2f}, {next_wp[1]:.2f})")
            
            is_final = self.current_waypoint_index == len(self.current_waypoints) - 1
            if is_final:
                next_wp = (next_wp[0], next_wp[1], self.goal_pose_orientation)
            
            self.local_planner.navigate_to(next_wp, is_final=is_final)
        else:
            self.get_logger().info("All waypoints reached! Navigation to goal complete.")
            self.navigating = False
            goal_reached_msg = Bool()
            goal_reached_msg.data = True
            self.goal_reached_pub.publish(goal_reached_msg)
            self.get_logger().info("Published goal completion signal to /goal_reached.")
            
            self.goal_pose = None
            self.get_logger().info("Ready for a new goal.")

    def try_plan(self):
        if self.map is None or not self.start_received or not self.goal_received:
            return

        if self.navigating: 
            return

        self.get_logger().info("Received map, initial pose, and goal. Initiating global path planning...")

        planner = AStarPlanner(self.inflated_map, self.resolution, self.origin)
        
        try:
            path = planner.plan(self.start_pose, self.goal_pose)
        except ValueError as e:
            self.get_logger().warn(f"[A* Planner] Planning failed: {e}")
            return
        
        if path is None:
            self.get_logger().error("No path found to the goal. Check if the goal is valid and not in an obstacle.")
            self.goal_received = False
            return

        waypoints = WaypointExtractor.extract_waypoints(path, method='combined', distance_threshold=0.65, angle_threshold=0.4)
        self.get_logger().info(f"Global path planned. Extracted {len(waypoints)} waypoints.")

        self.current_waypoints = waypoints
        self.current_waypoint_index = 0
        self.navigating = True

        self.publish_path(path)
        self.publish_waypoints(waypoints)

        if waypoints:
            self.get_logger().info(f"Starting navigation to first waypoint: ({waypoints[0][0]:.2f}, {waypoints[0][1]:.2f})")
            
            is_final = len(waypoints) == 1
            if is_final:
                self.local_planner.navigate_to((waypoints[0][0], waypoints[0][1], self.goal_pose_orientation), is_final=True)
            else:
                self.local_planner.navigate_to((waypoints[0][0], waypoints[0][1]), is_final=False)

        self.goal_received = False
        self.get_logger().info("Global planning cycle complete. Robot is now navigating.")

def main(args=None):
    rclpy.init(args=args)
    node = NavigationManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()