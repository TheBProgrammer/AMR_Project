#!/usr/bin/env python3

"""
mcl_node.py
Monte Carlo Localization (MCL) ROS2 Node

This node implements a particle filter that:
 - waits for a user-provided `/initialpose` before starting
 - Uses log-space likelihoods to avoid underflow
 - Uses ESS to trigger systematic (low-variance) resampling
 - Publishes `/mcl_pose` (PoseStamped) and broadcasts `map -> odom` transform
 - Publishes `/particle_cloud` (PoseArray) for RViz visualization
"""

import rclpy
from rclpy.node import Node

import numpy as np
import math
from scipy.ndimage import distance_transform_edt

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseArray,
    Pose,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster

from tf2_ros import Buffer, TransformListener, TransformException
import rclpy.duration

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

class ParticleFilter(Node):
    """ROS2 Node implementing a particle filter for 2D localization.

    Behaviour notes:
    - If `wait_for_initial_pose` is True (default), the node will NOT run the
      predict/update/resample loop until `/initialpose` is received from the user
      (RViz "2D Pose Estimate"). This prevents early particle collapse.
    - If `wait_for_initial_pose` is False, the node will do a global random
      initialization once the map is available.
    """

    def __init__(self):
        """Node initialization: parameters, topics, state."""
        super().__init__('particle_filter_node')

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter('num_particles', 100)
        self.declare_parameter('alpha1', 0.02)
        self.declare_parameter('alpha2', 0.01)
        self.declare_parameter('alpha3', 0.05)
        self.declare_parameter('alpha4', 0.01)
        self.declare_parameter('laser_beams', 20)
        self.declare_parameter('likelihood_sigma', 0.05)
        self.declare_parameter('resample_threshold', 0.7)
        self.declare_parameter("base_frame", "base_footprint")
        # If True, wait for /initialpose (RViz) before running filter cycles
        self.declare_parameter('wait_for_initial_pose', True)

        # load params
        self.num_particles = int(self.get_parameter('num_particles').value)
        self.alpha1 = float(self.get_parameter('alpha1').value)
        self.alpha2 = float(self.get_parameter('alpha2').value)
        self.alpha3 = float(self.get_parameter('alpha3').value)
        self.alpha4 = float(self.get_parameter('alpha4').value)
        self.laser_beams = int(self.get_parameter('laser_beams').value)
        self.likelihood_sigma = float(self.get_parameter('likelihood_sigma').value)
        self.resample_threshold = float(self.get_parameter('resample_threshold').value)
        self.base_frame = self.get_parameter("base_frame").value
        self.wait_for_initial_pose = bool(self.get_parameter('wait_for_initial_pose').value)

        # --------------------
        # Internal state
        # --------------------
        self.map = None
        self.map_resolution = None
        self.map_origin = None
        self.occupancy_data = None
        self.distance_field = None

        # particles array: Nx4 = [x, y, theta, weight]
        self.particles = None
        self.last_odom_pose = None

        # latest messages
        self.last_odom_msg = None
        self.last_scan_msg = None

        # whether we have received an /initialpose and are allowed to run
        self.initialized = False

        # TF broadcaster for map->odom
        self.tf_broadcaster = TransformBroadcaster(self)

        # Use transient local / reliable QoS for the map so late starters (RViz) still get it
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # sensor data profile for LaserScan (best-effort, high throughput)
        scan_qos = qos_profile_sensor_data

        # --------------------
        # Subscriptions
        # --------------------
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile=map_qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=qos_profile_system_default)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=scan_qos)
        self.create_subscription(PoseWithCovarianceStamped, 
                                 '/initialpose', 
                                 self.initialpose_callback, 
                                 qos_profile=qos_profile_system_default)

        # --------------------
        # Publishers
        # --------------------
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/mcl_pose', 10)
        self.particle_cloud_pub = self.create_publisher(PoseArray, '/particle_cloud', 10)

        # Setup TF2 buffer and listener (required for looking up odom → base_frame)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------
        # Timer
        # --------------------
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('Particle Filter Node initialized (waiting_for_initial_pose=%s)' % str(self.wait_for_initial_pose))

    # --------------------
    # Callbacks
    # --------------------
    def map_callback(self, msg: OccupancyGrid):
        """Process map and compute distance field. Does NOT automatically start filter
        when wait_for_initial_pose is True. If waiting is disabled, initializes particles
        globally here.
        """
        self.map = msg
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        # reshape occupancy data (height x width)
        self.occupancy_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        # treat unknown as obstacle for distance field safety
        obstacle_mask = (self.occupancy_data > 50) | (self.occupancy_data == -1)
        obstacle_mask = obstacle_mask.astype(np.uint8)
        self.distance_field = distance_transform_edt(1 - obstacle_mask) * self.map_resolution

        self.get_logger().info(f"Map received: size={self.occupancy_data.shape}, res={self.map_resolution}")

        # If user chose NOT to wait for initial pose, initialize globally now
        if not self.wait_for_initial_pose and (self.particles is None or len(self.particles) == 0):
            self.get_logger().info('Global initialization of particles (no /initialpose required)')
            self.initialize_particles(global_init=True)

    def initialize_particles(self, global_init=False, center_pose=None, cov=None):
        """Initialize particles in two modes:

        - global_init=True : randomly distribute particles across free map cells
        - center_pose provided (x,y,theta) : sample around that pose using cov (if given)
        """
        if self.occupancy_data is None:
            self.get_logger().warn('initialize_particles called but occupancy data is None')
            return

        # ensure there is at least one free cell
        free_cells = np.argwhere(self.occupancy_data == 0)
        if free_cells.size == 0:
            self.get_logger().error('No free cells in map to initialize particles')
            return

        # allocate particle array exactly num_particles
        self.particles = np.zeros((self.num_particles, 4))

        if center_pose is not None:
            # sample around provided center pose using covariance if available
            x_c, y_c, theta_c = center_pose
            # covariance diagonal fallback values
            std_x = math.sqrt(cov[0]) if cov is not None and cov[0] > 0 else 0.2
            std_y = math.sqrt(cov[7]) if cov is not None and cov[7] > 0 else 0.2
            std_theta = math.sqrt(cov[35]) if cov is not None and cov[35] > 0 else 0.1
            for i in range(self.num_particles):
                self.particles[i, 0] = np.random.normal(x_c, std_x)
                self.particles[i, 1] = np.random.normal(y_c, std_y)
                self.particles[i, 2] = np.random.normal(theta_c, std_theta)
                self.particles[i, 3] = 1.0 / self.num_particles
            self.get_logger().info(f'Initialized {self.num_particles} particles around initial pose')
            return

        # GLOBAL initialization: sample free cells (with replacement if fewer free cells than N)
        num_free = free_cells.shape[0]
        replace = num_free < self.num_particles
        chosen_idx = np.random.choice(num_free, self.num_particles, replace=replace)
        chosen = free_cells[chosen_idx]

        for i, (my, mx) in enumerate(chosen):
            # place particle at center of chosen cell for better accuracy
            x = self.map_origin[0] + (mx + 0.5) * self.map_resolution
            y = self.map_origin[1] + (my + 0.5) * self.map_resolution
            theta = np.random.uniform(-math.pi, math.pi)
            self.particles[i] = [x, y, theta, 1.0 / self.num_particles]

        self.get_logger().info(f'Globally initialized {self.num_particles} particles')

    def initialpose_callback(self, msg):
        '''Re-initializes particles around a specific user-provided initial pose with Gaussian noise'''
        # Get coordinates and orientation from the initial pose message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.yaw_from_quaternion(msg.pose.pose.orientation)

        # Extract standard deviations from the covariance matrix for spreading particles
        std_x = math.sqrt(msg.pose.covariance[0]) if msg.pose.covariance[0] > 0 else 0.2
        std_y = math.sqrt(msg.pose.covariance[7]) if msg.pose.covariance[7] > 0 else 0.2
        std_theta = math.sqrt(msg.pose.covariance[35]) if msg.pose.covariance[35] > 0 else 0.1

        # Re-create the particles array
        self.particles = np.zeros((self.num_particles, 4))

        # Initialize each particle by drawing samples from a Gaussian (normal) distribution
        for i in range(self.num_particles):
            self.particles[i, 0] = np.random.normal(x, std_x)
            self.particles[i, 1] = np.random.normal(y, std_y)
            self.particles[i, 2] = np.random.normal(theta, std_theta)
            self.particles[i, 3] = 1.0 / self.num_particles

        # Clear the last odometry pose to ensure the next odometry reading is treated as a new start
        self.last_odom_pose = None

        # Log and return as before
        self.get_logger().info("Particles initialized from /initialpose")
        self.initialized = True

    def odom_callback(self, msg: Odometry):
        """Store latest odometry message."""
        self.last_odom_msg = msg

    def scan_callback(self, msg: LaserScan):
        """Store latest laser scan message."""
        self.last_scan_msg = msg

    # --------------------
    # Main loop
    # --------------------
    def timer_callback(self):
        """Periodic filter loop. Only runs if initialized (or waiting disabled).

        Steps: predict -> update -> resample -> publish
        """

        # ensure map and messages available
        if self.map is None: 
            self.get_logger().warn('Waiting for map...')
            return

        if self.last_odom_msg is None:
            self.get_logger().warn('Waiting for odom...')
            return
        
        if self.last_scan_msg is None:
            self.get_logger().warn('Waiting for scan...')
            return

        # if configured to wait for initial pose and not yet initialized, skip
        if self.wait_for_initial_pose and not self.initialized:
            self.get_logger().warn('Waiting for user /initialpose — not running filter yet')
            return

        # ensure particles exist (in case map arrived after initialpose)
        if self.particles is None or len(self.particles) == 0:
            # if initial pose was provided, initialization already happened there; if not,
            # we can perform a global initialization here as a fallback
            self.get_logger().info('Particles missing — performing global initialization as fallback')
            self.initialize_particles(global_init=True)
            if self.particles is None:
                self.get_logger().warn('Particle initialization failed — skipping cycle')
                return

        # run the filter steps
        self.predict(self.last_odom_msg)
        self.update_weights(self.last_scan_msg)
        self.resample_particles()

        mean_pose = self.estimate_pose()
        self.get_logger().warn(f'Mean pose: {mean_pose}')
        if mean_pose is not None:
            self.publish_estimate(mean_pose)
            self.publish_particle_cloud()

    # --------------------
    # Particle filter internals
    # --------------------
    def predict(self, odom_msg: Odometry):
        """Motion update using odometry delta + probabilistic noise.

        Uses alphas to derive noise magnitudes per component as in Thrun et al.
        """
        x_now = odom_msg.pose.pose.position.x
        y_now = odom_msg.pose.pose.position.y
        theta_now = euler_from_quaternion([
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        ])[2]

        if self.last_odom_pose is None:
            self.last_odom_pose = (x_now, y_now, theta_now)
            return

        dx = x_now - self.last_odom_pose[0]
        dy = y_now - self.last_odom_pose[1]
        dtheta = math.atan2(math.sin(theta_now - self.last_odom_pose[2]), math.cos(theta_now - self.last_odom_pose[2]))

        delta_rot1 = math.atan2(dy, dx) - self.last_odom_pose[2]
        delta_trans = math.hypot(dx, dy)
        delta_rot2 = dtheta - delta_rot1

        for i in range(len(self.particles)):
            rot1_hat = delta_rot1 + np.random.normal(0, self.alpha1 * abs(delta_rot1) + self.alpha2 * delta_trans)
            trans_hat = delta_trans + np.random.normal(0, self.alpha3 * delta_trans + self.alpha4 * (abs(delta_rot1) + abs(delta_rot2)))
            rot2_hat = delta_rot2 + np.random.normal(0, self.alpha1 * abs(delta_rot2) + self.alpha2 * delta_trans)

            self.particles[i, 0] += trans_hat * math.cos(self.particles[i, 2] + rot1_hat)
            self.particles[i, 1] += trans_hat * math.sin(self.particles[i, 2] + rot1_hat)
            self.particles[i, 2] += (rot1_hat + rot2_hat)
            self.particles[i, 2] = math.atan2(math.sin(self.particles[i, 2]), math.cos(self.particles[i, 2]))

        self.last_odom_pose = (x_now, y_now, theta_now)

    def update_weights(self, scan_msg: LaserScan):
        """Measurement update using a likelihood field computed from endpoint distances.

        Computation is done in log-space and turned back into normalized weights.
        """
        if self.distance_field is None:
            self.get_logger().warn('No distance field available')
            return

        N = len(self.particles)
        log_weights = np.full(N, -np.inf)

        total_beams = len(scan_msg.ranges)
        beam_step = max(1, total_beams // self.laser_beams)
        indices = range(0, total_beams, beam_step)
        angle_increment = scan_msg.angle_increment

        origin_x, origin_y = self.map_origin
        res = self.map_resolution
        max_y = self.distance_field.shape[0]
        max_x = self.distance_field.shape[1]
        sigma = self.likelihood_sigma

        for i, p in enumerate(self.particles):
            px, py, ptheta = float(p[0]), float(p[1]), float(p[2])
            logp = 0.0
            for idx in indices:
                r = scan_msg.ranges[idx]
                if np.isinf(r) or np.isnan(r) or r < scan_msg.range_min or r > scan_msg.range_max:
                    continue
                beam_angle = ptheta + (scan_msg.angle_min + idx * angle_increment)
                x_end = px + r * math.cos(beam_angle)
                y_end = py + r * math.sin(beam_angle)
                mx = int(math.floor((x_end - origin_x) / res))
                my = int(math.floor((y_end - origin_y) / res))
                if 0 <= mx < max_x and 0 <= my < max_y:
                    dist = float(self.distance_field[my, mx])
                    logp += -0.5 * (dist ** 2) / (sigma ** 2)
                else:
                    logp += math.log(0.1)
            log_weights[i] = logp

        max_log = np.max(log_weights)
        if not np.isfinite(max_log):
            self.get_logger().warn('All log-weights -inf: skipping update')
            return

        weights = np.exp(log_weights - max_log)
        total = np.sum(weights)
        if total <= 0 or not np.isfinite(total):
            self.get_logger().warn('Invalid weights after exp — reinitializing particles')
            self.initialize_particles(global_init=True)
            return

        weights /= total
        self.particles[:, 3] = weights

    def resample_particles(self):
        """ESS-based systematic resampling with small jitter for diversity."""
        weights = self.particles[:, 3]
        N = len(weights)
        if N == 0:
            return

        ess = 1.0 / np.sum(weights ** 2)
        if ess >= self.resample_threshold * N:
            self.get_logger().warn(f'ESS {ess:.1f} >= threshold: skipping resample')
            return

        positions = (np.arange(N) + np.random.uniform(0.0, 1.0)) / N
        cumulative = np.cumsum(weights)
        indexes = np.searchsorted(cumulative, positions)
        indexes = np.clip(indexes, 0, N - 1)

        self.particles = self.particles[indexes].copy()
        jitter_xy = 0.01 * self.map_resolution if self.map_resolution is not None else 0.01
        self.particles[:, 0] += np.random.normal(0, jitter_xy, size=N)
        self.particles[:, 1] += np.random.normal(0, jitter_xy, size=N)
        self.particles[:, 3] = 1.0 / N
        self.get_logger().warn(f'Resampled particles; ESS was {ess:.2f}')

    def estimate_pose(self):
        """Return weighted mean pose from particle set."""
        if self.particles is None or len(self.particles) == 0:
            return None
        xs = self.particles[:, 0]
        ys = self.particles[:, 1]
        thetas = self.particles[:, 2]
        weights = self.particles[:, 3]
        mean_x = float(np.sum(xs * weights))
        mean_y = float(np.sum(ys * weights))
        sin_sum = float(np.sum(np.sin(thetas) * weights))
        cos_sum = float(np.sum(np.cos(thetas) * weights))
        mean_theta = math.atan2(sin_sum, cos_sum)
        return (mean_x, mean_y, mean_theta)

    def _transform_2d(self, x, y, theta):
        """Return 3x3 homogeneous transform for (x,y,theta)."""
        c = math.cos(theta); s = math.sin(theta)
        return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])
    
    def publish_estimate(self, mean_pose):
        """Publish PoseWithCovarianceStamped and broadcast map->odom transform from mean pose."""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose.position.x = mean_pose[0]
        pose_msg.pose.pose.position.y = mean_pose[1]
        pose_msg.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0.0, 0.0, mean_pose[2])
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        # Compute covariance
        xs = self.particles[:, 0]
        ys = self.particles[:, 1]
        thetas = self.particles[:, 2]
        weights = self.particles[:, 3]

        mean_x = np.sum(xs * weights)
        mean_y = np.sum(ys * weights)
        sin_mean = np.sum(np.sin(thetas) * weights)
        cos_mean = np.sum(np.cos(thetas) * weights)
        mean_theta = math.atan2(sin_mean, cos_mean)

        def angle_diff(a, b):
            return math.atan2(math.sin(a - b), math.cos(a - b))

        var_x = np.sum((xs - mean_x) ** 2 * weights)
        var_y = np.sum((ys - mean_y) ** 2 * weights)
        var_theta = np.sum([angle_diff(t, mean_theta) ** 2 * w for t, w in zip(thetas, weights)])

        cov = [0.0] * 36
        cov[0] = var_x
        cov[7] = var_y
        cov[35] = var_theta
        pose_msg.pose.covariance = cov

        self.pose_pub.publish(pose_msg)

        T_map_base = self._transform_2d(mean_pose[0], mean_pose[1], mean_pose[2])

        try:
            # Always use latest TF available (Time(0)) to avoid timestamp mismatch
            tf = self.tf_buffer.lookup_transform(
                "odom",
                self.base_frame,
                rclpy.time.Time(),  # latest
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            ox = tf.transform.translation.x
            oy = tf.transform.translation.y
            otheta = self.yaw_from_quaternion(tf.transform.rotation)

            T_odom_base = self._transform_2d(ox, oy, otheta)
            T_map_odom = T_map_base @ np.linalg.inv(T_odom_base)

            tx = T_map_odom[0, 2]
            ty = T_map_odom[1, 2]
            rot_theta = math.atan2(T_map_odom[1, 0], T_map_odom[0, 0])

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()  # current time
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'
            t.transform.translation.x = tx
            t.transform.translation.y = ty
            t.transform.translation.z = 0.0
            quat_map_to_odom = quaternion_from_euler(0.0, 0.0, rot_theta)
            t.transform.rotation.x = quat_map_to_odom[0]
            t.transform.rotation.y = quat_map_to_odom[1]
            t.transform.rotation.z = quat_map_to_odom[2]
            t.transform.rotation.w = quat_map_to_odom[3]
            self.tf_broadcaster.sendTransform(t)

        except TransformException as e:
            self.get_logger().warn(f"TF Lookup failed (odom -> {self.base_frame}): {e}")


    def publish_particle_cloud(self):
        """Publish particles as PoseArray for visualization."""
        if self.particles is None or len(self.particles) == 0:
            return
        cloud = PoseArray()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = self.map.header.frame_id if self.map is not None else 'map'
        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p[0]); pose.position.y = float(p[1]); pose.position.z = 0.0
            quat = quaternion_from_euler(0.0, 0.0, float(p[2]))
            pose.orientation.x = quat[0]; pose.orientation.y = quat[1]; pose.orientation.z = quat[2]; pose.orientation.w = quat[3]
            cloud.poses.append(pose)
        self.particle_cloud_pub.publish(cloud)

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
