# particle_filter.py

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import math
from scipy.ndimage import distance_transform_edt

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, PoseWithCovarianceStamped, TransformStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter_node')

        # Number of particles
        self.num_particles = 100
        self.particles = None

        # Map and related data
        self.map = None
        self.map_resolution = None
        self.map_origin = None
        self.occupancy_data = None
        self.distance_field = None

        # Last odometry pose for motion delta
        self.last_odom_pose = None

        # Motion noise parameters
        # Random noise added to particles during prediction
        # represent uncertainty in robot's odometry
        self.alpha1 = 0.05      # From Rotation while turning
        self.alpha2 = 0.05      # From rotation while moving straight
        self.alpha3 = 0.05       # From moving straight
        self.alpha4 = 0.05      # From moving straight while turning

        # ROS topic subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 10)

        # ROS topic publications
        self.pose_pub = self.create_publisher(PoseStamped, '/mcl_pose', 10)
        self.particle_cloud_pub = self.create_publisher(PoseArray, '/particle_cloud', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Store latest odom and scan for use in timer
        self.last_odom_msg = None
        self.last_scan_msg = None

        # Timer to trigger particle filter update loop
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Particle Filter Node Initialized")

    def map_callback(self, msg):
        '''Processes incoming OccupancyGrid map data and computes a distance field'''
        self.map = msg  # Store the raw map message
        self.map_resolution = msg.info.resolution   # Size of each grid cell in meters
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.occupancy_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # Create a binary obstacle mask: 1 for occupied cells (>50), 0 for free/unknown cells
        obstacle_mask = (self.occupancy_data > 50).astype(np.uint8)

        # Compute the Euclidean Distance Transform (EDT)
        # EDT calculates for each cell its distance to the nearest obstacle
        # `1 - obstacle_mask` inverts the mask so that obstacles are 0 and free space is 1,
        # which is what distance_transform_edt expects to calculate distances FROM obstacles
        self.distance_field = distance_transform_edt(1 - obstacle_mask) * self.map_resolution

        self.get_logger().info(f"Map received: size {self.occupancy_data.shape}")

        # Initialize particles once map is loaded
        self.initialize_particles()

    def initialize_particles(self):
        '''Initializes particles randomly across the free space of the loaded map'''
        # Check if map is loaded 
        if self.occupancy_data is None:
            self.get_logger().warn("Cannot initialize particles: no occupancy data yet")
            return

        # Sample free cells from the occupancy grid to initialize particles
        free_cells = np.argwhere(self.occupancy_data == 0)
        num_to_sample = min(len(free_cells), self.num_particles)

        # Select random free cells for particle initialization
        chosen_indices = free_cells[np.random.choice(free_cells.shape[0], num_to_sample, replace=False)]
        self.particles = np.zeros((num_to_sample, 4))

        # Initialize particles at the chosen free cells with random orientations and equal weights
        # Each particle is represented as [x, y, theta, weight]
        for i, (my, mx) in enumerate(chosen_indices):
            x = self.map_origin[0] + mx * self.map_resolution
            y = self.map_origin[1] + my * self.map_resolution
            theta = np.random.uniform(-math.pi, math.pi)
            weight = 1.0 / num_to_sample
            self.particles[i] = [x, y, theta, weight]

        self.get_logger().info(f"Initialized {num_to_sample} particles")

    def initialpose_callback(self, msg):
        '''Re-initializes particles around a specific user-provided initial pose with Gaussian noise'''
        # Get coordinates and orientation from the initial pose message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.yaw_from_quaternion(msg.pose.pose.orientation)

        # Extract standard deviations from the covariance matrix for spreading particles
        # If covariance is missing or invalid, use default small values to ensure some spread
        std_x = math.sqrt(msg.pose.covariance[0]) if msg.pose.covariance[0] > 0 else 0.2
        std_y = math.sqrt(msg.pose.covariance[7]) if msg.pose.covariance[7] > 0 else 0.2
        std_theta = math.sqrt(msg.pose.covariance[35]) if msg.pose.covariance[35] > 0 else 0.1

        # Re-create the particles array
        self.particles = np.zeros((self.num_particles, 4))

        # Initialize each particle by drawing samples from a Gaussian (normal) distribution
        # centered at the provided initial pose, with standard deviations from the covariance
        for i in range(self.num_particles):
            self.particles[i, 0] = np.random.normal(x, std_x)
            self.particles[i, 1] = np.random.normal(y, std_y)
            self.particles[i, 2] = np.random.normal(theta, std_theta)
            self.particles[i, 3] = 1.0 / self.num_particles

        # Clear the last odometry pose to ensure the next odometry reading is treated as a new start
        self.last_odom_pose = None
        self.get_logger().info("Particles initialized from /initialpose")

    def odom_callback(self, msg):
        '''Stores the most recently received Odometry message'''
        self.last_odom_msg = msg

    def scan_callback(self, msg):
        '''Stores the most recently received LaserScan message'''
        self.last_scan_msg = msg

    def timer_callback(self):
        '''The main loop of the particle filter, triggered periodically by a timer'''
    
        # Check if we have the necessary data to perform an update
        if self.map is None or self.last_odom_msg is None or self.last_scan_msg is None:
            self.get_logger().info("Waiting for map, odom, and scan data to be ready")
            return

        # Initialize particles if they haven't been set up yet (e.g., no /initialpose or map received yet).
        if self.particles is None:
            self.initialize_particles()
            # If initialization fails, skip 
            if self.particles is None:
                self.get_logger().warn("Particle initialization failed, skipping update")
                return

        # Particle Filter Steps (Prediction, Update, Resampling)
        self.predict(self.last_odom_msg)                # Motion Model Step, moves particles based on odometry adding noise
        self.update_weights(self.last_scan_msg)         # Measurement Model, computes weights based on laser scan data
        self.resample_particles()                       # Resampling Step, resamples particles based on their weights
        
        # Post-processing and publishing
        mean_pose = self.estimate_pose()                # Estimate the best pose from weighted average of all particles
        self.publish_estimate(mean_pose)                # Publish the estimated pose and the transform from map to odom
        self.publish_particle_cloud()                   # Publish all the particles for visualization

    def predict(self, odom_msg):
        '''Predicts the new poses of all particles based on the robot's odometry movement and adds noise'''
        # Get the current odometry pose
        x_now = odom_msg.pose.pose.position.x
        y_now = odom_msg.pose.pose.position.y
        theta_now = self.yaw_from_quaternion(odom_msg.pose.pose.orientation)
        
        # Store the first odometry pose
        if self.last_odom_pose is None:
            self.last_odom_pose = (x_now, y_now, theta_now)
            return

        # Calculate the change in pose since the last odometry message
        dx = x_now - self.last_odom_pose[0]
        dy = y_now - self.last_odom_pose[1]
        dtheta = np.arctan2(np.sin(theta_now - self.last_odom_pose[2]), np.cos(theta_now - self.last_odom_pose[2]))

        # Compute the deltas for the motion model
        # accounts for how a robot turns and then moves or moves and then turns
        delta_rot1 = np.arctan2(dy, dx) - self.last_odom_pose[2]    # Rotation before moving straight
        delta_trans = np.hypot(dx, dy)                              # Translation
        delta_rot2 = dtheta - delta_rot1                            # Rotation after moving straight

        # Update each particle 
        for i in range(len(self.particles)):
            # Add noise to each motion component based on the alpha parameters
            # This simulates uncertainty and allows particles to spread
            rot1_hat = delta_rot1 + np.random.normal(0, self.alpha1 * abs(delta_rot1) + self.alpha2 * delta_trans)
            trans_hat = delta_trans + np.random.normal(0, self.alpha3 * delta_trans + self.alpha4 * (abs(delta_rot1) + abs(delta_rot2)))
            rot2_hat = delta_rot2 + np.random.normal(0, self.alpha1 * abs(delta_rot2) + self.alpha2 * delta_trans)

            # Apply the noisy motion to the particle's current pose
            self.particles[i, 0] += trans_hat * np.cos(self.particles[i, 2] + rot1_hat)                     # Update x position
            self.particles[i, 1] += trans_hat * np.sin(self.particles[i, 2] + rot1_hat)                     # Update y position
            self.particles[i, 2] += rot1_hat + rot2_hat                                                     # Update orientation
            self.particles[i, 2] = np.arctan2(np.sin(self.particles[i, 2]), np.cos(self.particles[i, 2]))   # Normalize angle 

        # Update the last odometry pose to the current one
        self.last_odom_pose = (x_now, y_now, theta_now)

    def update_weights(self, scan_msg):
        '''Updates the weights of each particle based on how well its expected laser scan matches the actual scan data'''
        # Likelihood field measurement model update
        if self.occupancy_data is None or self.distance_field is None:
            self.get_logger().warn("No map or distance field available")
            return

        # initialize array to store new weights for each particle
        weights = np.zeros(len(self.particles))
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        step = max(1, len(angles) // 20)    # Reduce number of beams to speed up computation
        sigma = 0.2                         # likelihood field standard deviation, laser sensor noise

        # Iterate over each particle and compute its weight based on the laser scan
        for i, p in enumerate(self.particles):
            # Particle pose: px, py, ptheta
            px, py, ptheta = p[:3]
            score = 1.0

            # Iterate through a subset of laser scan beams
            for idx in range(0, len(angles), step):
                # actual distance reading from the laser scan
                r = scan_msg.ranges[idx]
                if np.isinf(r) or np.isnan(r) or r < scan_msg.range_min or r > scan_msg.range_max:
                    continue
                
                # Calculate the global angle of the laser beam if originating from this particle's pose
                beam_angle = ptheta + angles[idx]

                # Calculate the expected endpoint of the laser beam in world coordinates
                x_end = px + r * np.cos(beam_angle)
                y_end = py + r * np.sin(beam_angle)

                # Convert the world coords of end point to map coordinates
                mx = int((x_end - self.map_origin[0]) / self.map_resolution)
                my = int((y_end - self.map_origin[1]) / self.map_resolution)

                # Check if the calculated endpoint is within the map boundaries
                if 0 <= mx < self.distance_field.shape[1] and 0 <= my < self.distance_field.shape[0]:
                    # Get the pre-calculated distance from this map cell to the nearest obstacle.
                    dist = self.distance_field[my, mx]

                    # Calculate a probability (likelihood) based on this distance using a Gaussian model.
                    # A smaller 'dist' (closer to an obstacle on map) results in a higher 'prob'
                    prob = np.exp(-0.5 * (dist ** 2) / (sigma ** 2))

                    # Multiply the particle's overall score by this probability
                    score *= prob
                else:
                    score *= 0.1 # Apply a small penalty if the beam goes off the map

            # Store the final score for this particle, ensuring it's not exactly zero to avoid division by zero later
            weights[i] = max(score, 1e-10)

        # Sum of all weights
        total_weight = np.sum(weights)

        # Re-initialize weights if total weight is zero, all guesses are bad
        if total_weight == 0:
            self.get_logger().warn("All particle weights are zero! Reinitializing.")
            self.initialize_particles()
            return

        # Normalize the weights
        weights /= total_weight

        # Update the particles with their new weights
        self.particles[:, 3] = weights

    def resample_particles(self):
        '''
        Resamples the particles based on their current weights, keeping high-weight particles and discarding low-weight ones
        based on Low-variance resampling algorithm which efficiently selects new particles based on their probabilities
        '''
           
        weights = self.particles[:, 3]              # get the current weights of the particles  
        N = len(self.particles)                     # number of particles
        cumulative_sum = np.cumsum(weights)         # cumulative sum of weights, sum upto each particle
        cumulative_sum[-1] = 1.0                    # Ensure last element is exactly 1.0 due to floating point inaccuracies

        indexes = np.zeros(N, dtype=int)            
        r = np.random.uniform(0, 1.0 / N)           # Start point for the "wheel"
        i = 0                                       # index for the cumulative sum
        
        # Spin the "wheel" N times
        for m in range(N):
            U = r + m / N                           # Position on the wheel
            while U > cumulative_sum[i]:            # Find which particle segment U falls into
                i += 1
            indexes[m] = i                          # Store the index of the selected particle  

        self.particles = self.particles[indexes]    # Select particles using the chosen indexes
        self.particles[:, 3] = 1.0 / N              # Reset weights to be equal for the new set of particles

    def estimate_pose(self):
        '''Estimates the robot's most likely pose by computing the weighted average of all particles'''

        # Extract the x, y, theta, and weights from the particles
        xs = self.particles[:, 0]
        ys = self.particles[:, 1]
        thetas = self.particles[:, 2]
        weights = self.particles[:, 3]

        # Calculate the weighted mean for x, y, and theta
        mean_x = np.sum(xs * weights)
        mean_y = np.sum(ys * weights)
        sin_sum = np.sum(np.sin(thetas) * weights)
        cos_sum = np.sum(np.cos(thetas) * weights)
        mean_theta = math.atan2(sin_sum, cos_sum)

        return (mean_x, mean_y, mean_theta)

    def publish_estimate(self, mean_pose):
        '''Publishes the estimated robot pose and the transform from the 'map' frame to the 'odom' frame'''

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map' # Frame where this pose is defined
        pose_msg.pose.position.x = mean_pose[0]
        pose_msg.pose.position.y = mean_pose[1]
        pose_msg.pose.position.z = 0.0
        
        quat = quaternion_from_euler(0.0, 0.0, mean_pose[2])
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)
        
        # Calculate and publish the transform from 'map' to 'odom'.
        # This transform aligns the odometry frame with the global map frame, correcting odometry drift.
        # It's derived from the estimated robot pose in map frame (T_map_base_link)
        # and the robot's current odometry pose (T_odom_base_link).
        # We need to find T_map_odom such that T_map_base_link = T_map_odom * T_odom_base_link.
        # This implies T_map_odom = T_map_base_link * (T_odom_base_link)^-1.

        if self.last_odom_msg is None:
            self.get_logger().warn("Cannot publish map->odom transform: last_odom_msg is None.")
            return
        
        # The current odom_pose (relative to the odom frame itself)
        odom_x = self.last_odom_msg.pose.pose.position.x
        odom_y = self.last_odom_msg.pose.pose.position.y
        odom_theta = self.yaw_from_quaternion(self.last_odom_msg.pose.pose.orientation)
        
        # Calculate the inverse of the odometry transform (T_base_link_odom).
        # This tells us how the 'odom' frame looks from the 'base_link' frame
        inv_odom_x = -(odom_x * np.cos(odom_theta) + odom_y * np.sin(odom_theta))
        inv_odom_y = -(-odom_x * np.sin(odom_theta) + odom_y * np.cos(odom_theta)) # Corrected y
        inv_odom_theta = -odom_theta
        
        # Combine the estimated pose (T_map_base_link) with the inverse odometry (T_base_link_odom)
        # to get the required map-to-odom transform (T_map_odom).
        map_to_odom_x = mean_pose[0] + (inv_odom_x * np.cos(mean_pose[2]) - inv_odom_y * np.sin(mean_pose[2]))
        map_to_odom_y = mean_pose[1] + (inv_odom_x * np.sin(mean_pose[2]) + inv_odom_y * np.cos(mean_pose[2]))
        map_to_odom_theta = mean_pose[2] + inv_odom_theta
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom' # <--- Changed this to 'odom'
        t.transform.translation.x = map_to_odom_x
        t.transform.translation.y = map_to_odom_y
        t.transform.translation.z = 0.0 # Assuming 2D
        quat_map_to_odom = quaternion_from_euler(0.0, 0.0, map_to_odom_theta)
        t.transform.rotation.x = quat_map_to_odom[0]
        t.transform.rotation.y = quat_map_to_odom[1]
        t.transform.rotation.z = quat_map_to_odom[2]
        t.transform.rotation.w = quat_map_to_odom[3]
        self.tf_broadcaster.sendTransform(t)

    def publish_particle_cloud(self):
        '''Publishes all individual particles as a PoseArray message for visualization'''
        cloud_msg = PoseArray()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = 'map'

        # Iterate through each particle and convert its pose to a Pose message, then add to the array
        for p in self.particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = 0.0
            quat = quaternion_from_euler(0.0, 0.0, p[2])
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            cloud_msg.poses.append(pose)

        self.particle_cloud_pub.publish(cloud_msg)

    def yaw_from_quaternion(self, q):
        '''Helper function: Converts a quaternion orientation message to a yaw angle in radians'''
        return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
