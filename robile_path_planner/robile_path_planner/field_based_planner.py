# field_based_planner.py

import math
import rclpy
import numpy as np
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult

class Planner:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        self.cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        node.declare_parameter("ka", 0.75)
        node.declare_parameter("kr", 3.0)
        node.declare_parameter("p_0", 0.3)
        node.declare_parameter("stop_threshold", 0.2)
        node.declare_parameter("slowdown_radius", 0.55)
        node.declare_parameter("max_linear_vel", 0.3)
        node.declare_parameter("angular_tolerance", 0.1) 

        self.ka = node.get_parameter("ka").get_parameter_value().double_value
        self.kr = node.get_parameter("kr").get_parameter_value().double_value
        self.p_0 = node.get_parameter("p_0").get_parameter_value().double_value
        self.stop_threshold = node.get_parameter("stop_threshold").get_parameter_value().double_value
        self.slowdown_radius = node.get_parameter("slowdown_radius").get_parameter_value().double_value
        self.max_linear_vel = node.get_parameter("max_linear_vel").get_parameter_value().double_value
        self.angular_tolerance = node.get_parameter("angular_tolerance").get_parameter_value().double_value
        
        self.logger.info(f"Loaded parameters: ka={self.ka}, kr={self.kr}, p_0={self.p_0}, stop_threshold={self.stop_threshold}, slowdown_radius={self.slowdown_radius}, max_linear_vel={self.max_linear_vel}, angular_tolerance={self.angular_tolerance}")

        node.add_on_set_parameters_callback(self.param_callback)

        self.latest_scan = None
        self.robot_pose = {"x": 0, "y": 0, "theta": 0}
        self.goal_pose = None
        self.goal_reached_callback = None
        self.is_final_waypoint = False # Flag to distinguish final goal from intermediate waypoints

    def set_goal_reached_callback(self, callback):
        self.goal_reached_callback = callback

    def update_scan(self, msg):
        self.latest_scan = msg

    def navigate_to(self, goal_pose, is_final=False):
        self.goal_pose = {
            "x": goal_pose[0],
            "y": goal_pose[1],
            "theta": goal_pose[2] if len(goal_pose) > 2 else None
        }
        self.is_final_waypoint = is_final
        self.logger.info(f"Navigating to: ({goal_pose[0]:.2f}, {goal_pose[1]:.2f})")
        if is_final and self.goal_pose["theta"] is not None:
            self.logger.info(f"Final goal orientation: {math.degrees(self.goal_pose['theta']):.2f} degrees")

    def param_callback(self, params):
        for param in params:
            if param.name == "ka":
                self.ka = param.value
            elif param.name == "kr":
                self.kr = param.value
            elif param.name == "p_0":
                self.p_0 = param.value
            elif param.name == "stop_threshold":
                self.stop_threshold = param.value
            elif param.name == "slowdown_radius":
                self.slowdown_radius = param.value
            elif param.name == "max_linear_vel":
                self.max_linear_vel = param.value
            elif param.name == "angular_tolerance":
                self.angular_tolerance = param.value
        self.logger.info(f"Updated parameters: ka={self.ka}, kr={self.kr}, p_0={self.p_0}, stop_threshold={self.stop_threshold}, slowdown_radius={self.slowdown_radius}, max_linear_vel={self.max_linear_vel}, angular_tolerance={self.angular_tolerance}")
        return SetParametersResult(successful=True)

    def get_robot_pose(self):
        timeout = Duration(seconds=0.5)
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(), timeout)
            t = transform.transform.translation
            r = transform.transform.rotation
            quat = [r.x, r.y, r.z, r.w]
            _, _, yaw = euler_from_quaternion(quat)
            self.robot_pose = {"x": t.x, "y": t.y, "theta": yaw}
            return True
        except Exception as e:
            return False

    def planner_loop(self):
        if self.goal_pose is None or self.latest_scan is None:
            return
        
        if not self.get_robot_pose():
            return

        x, y, theta = self.robot_pose.values()
        goal_x, goal_y, goal_theta = self.goal_pose.values()

        dx = goal_x - x
        dy = goal_y - y
        dist_to_goal = math.sqrt(dx ** 2 + dy ** 2)

        twist = Twist()

        # Phase 1: Go to goal position
        if dist_to_goal > self.stop_threshold:
            # Attractive force with slowdown logic
            attr_force_magnitude = self.ka
            if dist_to_goal < self.slowdown_radius:
                attr_force_magnitude = self.ka * (dist_to_goal / self.slowdown_radius)
            
            v_attr_x = attr_force_magnitude * (dx / dist_to_goal)
            v_attr_y = attr_force_magnitude * (dy / dist_to_goal)

            # Repulsive force
            v_rep_x = 0.0
            v_rep_y = 0.0
            for i, r in enumerate(self.latest_scan.ranges):
                if r == float('inf') or r == 0.0 or r > self.p_0:
                    continue
                
                if 0.01 < r <= self.p_0:
                    angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
                    force = self.kr * (1.0 / r - 1.0 / self.p_0) / (r ** 2)
                    force = min(force, 5.0)

                    world_angle = theta + angle
                    v_rep_x += -force * math.cos(world_angle)
                    v_rep_y += -force * math.sin(world_angle)

            # Total velocity in world frame
            vx_world = v_attr_x + v_rep_x
            vy_world = v_attr_y + v_rep_y

            # Transform to robot frame
            vx_robot = vx_world * math.cos(theta) + vy_world * math.sin(theta)
            vy_robot = -vx_world * math.sin(theta) + vy_world * math.cos(theta)
            
            twist.linear.x = np.clip(vx_robot, 0, self.max_linear_vel)

            desired_heading = math.atan2(vy_world, vx_world)
            angle_diff = (desired_heading - theta + math.pi) % (2 * math.pi) - math.pi
            twist.angular.z = np.clip(angle_diff * 1.5, -0.5, 0.5)
            
            if abs(twist.linear.x) < 0.1:
                twist.linear.x = 0.0

        # Phase 2: At goal position, but not final goal or need to align orientation
        elif self.is_final_waypoint and goal_theta is not None:
            # Robot is at the goal position, now orient to the goal heading
            angle_diff = (goal_theta - theta + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) > self.angular_tolerance:
                twist.linear.x = 0.0
                twist.angular.z = np.clip(angle_diff * 1.5, -0.5, 0.5)
            else:
                self.stop_robot()
                self.logger.info(f"Waypoint reached! Distance: {dist_to_goal:.3f}, Orientation aligned: {math.degrees(angle_diff):.2f} degrees")
                if self.goal_reached_callback:
                    self.goal_reached_callback()
                return
        
        # Phase 3: At goal position, no final orientation required
        else:
            self.stop_robot()
            self.logger.info(f"Waypoint reached! Distance: {dist_to_goal:.3f}")
            if self.goal_reached_callback:
                self.goal_reached_callback()
            return

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """Publishes a zero Twist message to stop the robot."""
        twist = Twist()
        self.cmd_pub.publish(twist)
        self.goal_pose = None
        self.is_final_waypoint = False
        self.logger.info("Stopping robot.")