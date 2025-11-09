# extract_waypoints.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion
from rclpy.duration import Duration
import numpy as np
import math
from typing import List, Tuple, Optional



"""
Utility class to extract reduced waypoint sequences from a full A* path
based on distance or angular deviation heuristics.
"""


class WaypointExtractor:
    """Extract waypoints from A* path"""
    
    @staticmethod
    def extract_waypoints(path: List[Tuple[float, float]], 
                         method='distance', 
                         distance_threshold=1.0,
                         angle_threshold=0.5) -> List[Tuple[float, float]]:
        """
        Extract waypoints from path using different methods:
        - 'distance': waypoint every distance_threshold meters
        - 'angle': waypoint at direction changes > angle_threshold radians
        - 'combined': combination of both methods
        """
        if len(path) <= 2:
            return path
            
        waypoints = [path[0]]  # Always include start
        
        if method == 'distance':
            current_waypoint = path[0]
            for point in path[1:]:
                dist = math.sqrt((point[0] - current_waypoint[0])**2 + 
                               (point[1] - current_waypoint[1])**2)
                if dist >= distance_threshold:
                    waypoints.append(point)
                    current_waypoint = point
        
        elif method == 'angle':
            for i in range(1, len(path) - 1):
                prev_point = path[i-1]
                curr_point = path[i]
                next_point = path[i+1]
                
                # Calculate angles
                angle1 = math.atan2(curr_point[1] - prev_point[1], 
                                  curr_point[0] - prev_point[0])
                angle2 = math.atan2(next_point[1] - curr_point[1], 
                                  next_point[0] - curr_point[0])
                
                # Calculate angle difference
                angle_diff = abs(angle2 - angle1)
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                
                if angle_diff > angle_threshold:
                    waypoints.append(curr_point)
        
        elif method == 'combined':
            current_waypoint = path[0]
            for i in range(1, len(path) - 1):
                point = path[i]
                next_point = path[i+1]
                
                # Distance check
                dist = math.sqrt((point[0] - current_waypoint[0])**2 + 
                               (point[1] - current_waypoint[1])**2)
                
                # Angle check
                if i > 0:
                    prev_point = path[i-1]
                    angle1 = math.atan2(point[1] - prev_point[1], 
                                      point[0] - prev_point[0])
                    angle2 = math.atan2(next_point[1] - point[1], 
                                      next_point[0] - point[0])
                    angle_diff = abs(angle2 - angle1)
                    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                else:
                    angle_diff = 0
                
                if (dist >= distance_threshold and angle_diff > angle_threshold):
                    waypoints.append(point)
                    current_waypoint = point
        
        waypoints.append(path[-1])  # Always include goal
        return waypoints