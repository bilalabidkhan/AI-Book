---
sidebar_position: 4
title: "Nav2 for Humanoid Navigation: Path Planning and Motion Control"
---

# Nav2 for Humanoid Navigation: Path Planning and Motion Control

This chapter covers navigation with Nav2 for humanoid robots, focusing on path planning and motion control that accounts for the unique dynamics and constraints of humanoid locomotion. Nav2 provides the standard navigation framework for ROS 2 systems, with extensions for humanoid-specific requirements.

## Introduction to Nav2 for Humanoid Robots

Navigation2 (Nav2) is the standard navigation framework for ROS 2, providing a complete stack for path planning, motion control, and obstacle avoidance. For humanoid robots, Nav2 requires special considerations for:

- **Dynamic balance**: Maintaining stability during movement
- **Multi-contact locomotion**: Managing foot placement and contact points
- **Center of Mass (CoM) control**: Managing the robot's center of mass during navigation
- **Step planning**: Planning discrete footstep locations
- **Whole-body control**: Coordinating multiple degrees of freedom

### Nav2 Architecture for Humanoid Robots
The standard Nav2 architecture includes:

```yaml
# Nav2 architecture for humanoid robots
nav2:
  lifecycle_manager:
    ros__parameters:
      node_names: ["map_server", "amcl", "bt_navigator", "controller_server", "planner_server", "recoveries_server", "waypoint_follower"]
      autostart: true

  map_server:
    ros__parameters:
      yaml_filename: "turtlebot3_world.yaml"
      topic_name: "map"
      frame_id: "map"

  amcl:
    ros__parameters:
      use_sim_time: True
      alpha1: 0.2
      alpha2: 0.2
      alpha3: 0.2
      alpha4: 0.2
      alpha5: 0.2
      base_frame_id: "base_footprint"  # For humanoid robots
      beam_skip_distance: 0.5
      beam_skip_error_threshold: 0.9
      beam_skip_threshold: 0.3
      do_beamskip: false
      global_frame_id: "map"
      lambda_short: 0.1
      laser_likelihood_max_dist: 2.0
      laser_max_range: 100.0
      laser_min_range: -1.0
      laser_model_type: "likelihood_field"
      max_beams: 60
      max_particles: 2000
      min_particles: 500
      odom_frame_id: "odom"
      pf_err: 0.05
      pf_z: 0.5
      recovery_alpha_fast: 0.0
      recovery_alpha_slow: 0.0
      resample_interval: 1
      robot_model_type: "nav2_amcl::LikelihoodFieldModelHeuristic"
      save_pose_rate: 0.5
      set_initial_pose: true
      sigma_hit: 0.2
      tf_broadcast: true
      transform_tolerance: 1.0
      update_min_a: 0.2
      update_min_d: 0.2
      z_hit: 0.5
      z_max: 0.5
      z_rand: 0.5
      z_short: 0.05

  bt_navigator:
    ros__parameters:
      use_sim_time: True
      global_frame: "map"
      robot_base_frame: "base_footprint"  # For humanoid robots
      odom_topic: "odom"
      bt_loop_duration: 10
      default_server_timeout: 20
      # Note: Modify the below for your own BT
      plugin_lib_names:
        - nav2_compute_path_to_pose_action_bt_node
        - nav2_compute_path_through_poses_action_bt_node
        - nav2_follow_path_action_bt_node
        - nav2_spin_action_bt_node
        - nav2_wait_action_bt_node
        - nav2_clear_costmap_service_bt_node
        - nav2_is_stuck_condition_bt_node
        - nav2_have_feedback_condition_bt_node
        - nav2_have_odom_condition_bt_node
        - nav2_have_costmap_condition_bt_node
        - nav2_initial_pose_received_condition_bt_node
        - nav2_reinitialize_global_localization_service_bt_node
        - nav2_rate_controller_bt_node
        - nav2_distance_controller_bt_node
        - nav2_speed_controller_bt_node
        - nav2_truncate_path_action_bt_node
        - nav2_goal_updater_node_bt_node
        - nav2_recovery_node_bt_node
        - nav2_pipeline_sequence_bt_node
        - nav2_round_robin_node_bt_node
        - nav2_transform_available_condition_bt_node
        - nav2_time_expired_condition_bt_node
        - nav2_path_expiring_timer_condition
        - nav2_distance_traveled_condition_bt_node
        - nav2_single_trigger_bt_node
        - nav2_is_battery_low_condition_bt_node
        - nav2_navigate_through_poses_action_bt_node
        - nav2_navigate_to_pose_action_bt_node
        - nav2_remove_passed_goals_action_bt_node
        - nav2_planner_selector_bt_node
        - nav2_controller_selector_bt_node
        - nav2_goal_checker_selector_bt_node
        - nav2_controller_cancel_bt_node
        - nav2_path_longer_on_approach_bt_node
        - nav2_wait_cancel_bt_node
        - nav2_spin_cancel_bt_node
        - nav2_is_battery_charging_condition_bt_node
```

## Path Planning with Dynamic Constraints

### Humanoid-Specific Path Planning
Humanoid robots require specialized path planning that accounts for their unique kinematic and dynamic constraints:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidPathPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Publishers
        self.path_pub = self.create_publisher(Path, '/humanoid/path', 10)
        self.footstep_pub = self.create_publisher(Marker, '/humanoid/footsteps', 10)

        # Subscriptions
        self.goal_sub = self.create_subscription(
            PoseStamped, '/humanoid/goal', self.goal_callback, 10
        )

        # Humanoid-specific parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters (for stepping over obstacles)
        self.max_step_up = 0.1  # maximum height to step up
        self.max_step_down = 0.15  # maximum height to step down
        self.com_height = 0.8   # Center of mass height

        # Balance constraints
        self.support_polygon_margin = 0.05  # Safety margin for CoM support
        self.max_lean_angle = 0.2  # Maximum lean angle in radians

    def goal_callback(self, msg):
        # Plan path for humanoid robot considering its constraints
        planned_path = self.plan_humanoid_path(msg.pose)

        # Publish the path
        self.publish_path(planned_path)

    def plan_humanoid_path(self, goal_pose):
        # This is a simplified example
        # Real implementation would use a specialized planner
        # that considers humanoid kinematics and balance

        # Get current robot position
        current_pose = self.get_current_pose()

        # Calculate straight-line path
        path = self.calculate_straight_path(current_pose, goal_pose)

        # Adapt path for humanoid locomotion
        adapted_path = self.adapt_path_for_humanoid(path)

        # Plan footsteps
        footsteps = self.plan_footsteps(adapted_path)

        # Visualize footsteps
        self.publish_footsteps(footsteps)

        return adapted_path

    def calculate_straight_path(self, start_pose, goal_pose):
        # Calculate straight-line path
        path = Path()
        path.header.frame_id = 'map'

        # Calculate distance
        dx = goal_pose.position.x - start_pose.position.x
        dy = goal_pose.position.y - start_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)

        # Create path points
        num_points = max(10, int(distance / 0.1))  # 10cm spacing
        for i in range(num_points + 1):
            t = i / num_points
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = start_pose.position.x + t * dx
            pose.pose.position.y = start_pose.position.y + t * dy
            pose.pose.position.z = start_pose.position.z  # Maintain height

            # Interpolate orientation
            start_quat = np.array([start_pose.orientation.x, start_pose.orientation.y,
                                  start_pose.orientation.z, start_pose.orientation.w])
            goal_quat = np.array([goal_pose.orientation.x, goal_pose.orientation.y,
                                 goal_pose.orientation.z, goal_pose.orientation.w])

            # Spherical linear interpolation (slerp) would be better here
            quat = self.interpolate_quaternion(start_quat, goal_quat, t)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            path.poses.append(pose)

        return path

    def adapt_path_for_humanoid(self, path):
        # Adapt path considering humanoid constraints
        adapted_path = Path()
        adapted_path.header = path.header

        # Consider step constraints
        for i, pose in enumerate(path.poses):
            # Ensure path is at appropriate height for humanoid
            if abs(pose.pose.position.z - self.com_height) > 0.1:
                # Adjust height based on terrain
                pose.pose.position.z = self.com_height

            adapted_path.poses.append(pose)

        return adapted_path

    def plan_footsteps(self, path):
        # Plan discrete footsteps for the humanoid
        footsteps = []

        # Simplified footstep planning
        # Real implementation would consider balance, terrain, etc.
        for i in range(0, len(path.poses), 3):  # Every 3rd pose
            pose = path.poses[i]

            # Plan left and right foot positions
            left_foot = self.calculate_foot_position(pose, 'left', len(footsteps))
            right_foot = self.calculate_foot_position(pose, 'right', len(footsteps))

            footsteps.extend([left_foot, right_foot])

        return footsteps

    def calculate_foot_position(self, body_pose, foot_side, step_index):
        # Calculate foot position relative to body
        foot_pose = PoseStamped()
        foot_pose.header = body_pose.header

        # Offset based on foot side
        if foot_side == 'left':
            lateral_offset = self.step_width / 2.0
        else:  # right
            lateral_offset = -self.step_width / 2.0

        # Calculate position with slight forward offset
        forward_offset = (step_index % 2) * self.step_length / 2.0  # Alternate steps

        # Apply transformations
        foot_pose.pose.position.x = body_pose.pose.position.x + forward_offset
        foot_pose.pose.position.y = body_pose.pose.position.y + lateral_offset
        foot_pose.pose.position.z = 0.0  # On ground

        # Copy orientation
        foot_pose.pose.orientation = body_pose.pose.orientation

        return foot_pose

    def interpolate_quaternion(self, q1, q2, t):
        # Linear interpolation of quaternions (simplified)
        # For real applications, use spherical linear interpolation
        q = q1 * (1 - t) + q2 * t
        # Normalize
        q = q / np.linalg.norm(q)
        return q

    def publish_path(self, path):
        self.path_pub.publish(path)

    def publish_footsteps(self, footsteps):
        # Publish footsteps as markers for visualization
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.a = 1.0

        for i, foot_pose in enumerate(footsteps):
            point = Point()
            point.x = foot_pose.pose.position.x
            point.y = foot_pose.pose.position.y
            point.z = foot_pose.pose.position.z
            marker.points.append(point)

            # Color based on left/right foot
            if i % 2 == 0:  # Left foot
                marker.colors.append(self.create_color(1.0, 0.0, 0.0, 1.0))  # Red
            else:  # Right foot
                marker.colors.append(self.create_color(0.0, 0.0, 1.0, 1.0))  # Blue

        self.footstep_pub.publish(marker)

    def create_color(self, r, g, b, a):
        from std_msgs.msg import ColorRGBA
        color = ColorRGBA()
        color.r = r
        color.g = g
        color.b = b
        color.a = a
        return color

    def get_current_pose(self):
        # This would typically get the current robot pose from TF or localization
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        return pose
```

### Balance-Aware Path Planning
Consider balance constraints in path planning:

```python
class BalanceAwarePathPlanner(HumanoidPathPlanner):
    def __init__(self):
        super().__init__()

        # Balance-specific parameters
        self.support_polygon = self.calculate_support_polygon()
        self.com_margin = 0.05  # Safety margin around CoM

    def calculate_support_polygon(self):
        # Calculate support polygon based on foot positions
        # This is a simplified 2D polygon
        # Real implementation would consider 3D support volume

        # For a biped, support polygon is between feet
        # This would be updated as feet move
        return [
            (-0.1, -self.step_width/2),   # Back-left
            (-0.1, self.step_width/2),    # Back-right
            (self.step_length, self.step_width/2),  # Front-right
            (self.step_length, -self.step_width/2)  # Front-left
        ]

    def is_path_balanced(self, path):
        # Check if path maintains balance
        for pose in path.poses:
            com_pos = self.calculate_com_position(pose)
            if not self.is_com_in_support_polygon(com_pos):
                return False
        return True

    def calculate_com_position(self, pose):
        # Calculate CoM position relative to feet
        # Simplified: CoM is at fixed height and relative to feet
        return (pose.pose.position.x, pose.pose.position.y)

    def is_com_in_support_polygon(self, com_pos):
        # Check if CoM is within support polygon
        # Using point-in-polygon algorithm
        x, y = com_pos
        poly = self.support_polygon

        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
```

## Motion Control for Humanoid Locomotion

### Whole-Body Motion Control
Humanoid motion control requires coordination of multiple degrees of freedom:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidMotionController:
    def __init__(self):
        # Robot-specific parameters
        self.robot_mass = 30.0  # kg
        self.com_height = 0.8   # meters
        self.foot_size = (0.2, 0.1)  # width, length

        # Control parameters
        self.com_gain = np.diag([5.0, 5.0, 5.0])
        self.foot_gain = np.diag([10.0, 10.0, 10.0])
        self.orientation_gain = 10.0

    def compute_control(self, desired_state, current_state):
        # Compute whole-body control commands
        # This is a simplified example
        control_commands = {}

        # Center of Mass control
        com_control = self.compute_com_control(
            desired_state['com'], current_state['com'],
            desired_state['com_vel'], current_state['com_vel']
        )

        # Foot placement control
        left_foot_control = self.compute_foot_control(
            desired_state['left_foot'], current_state['left_foot'],
            desired_state['left_foot_vel'], current_state['left_foot_vel']
        )

        right_foot_control = self.compute_foot_control(
            desired_state['right_foot'], current_state['right_foot'],
            desired_state['right_foot_vel'], current_state['right_foot_vel']
        )

        # Torso orientation control
        torso_control = self.compute_orientation_control(
            desired_state['torso_orientation'], current_state['torso_orientation'],
            desired_state['torso_angular_vel'], current_state['torso_angular_vel']
        )

        control_commands['com'] = com_control
        control_commands['left_foot'] = left_foot_control
        control_commands['right_foot'] = right_foot_control
        control_commands['torso'] = torso_control

        return control_commands

    def compute_com_control(self, desired_com, current_com, desired_vel, current_vel):
        # PD control for CoM
        pos_error = desired_com - current_com
        vel_error = desired_vel - current_vel

        control = self.com_gain @ pos_error + 2 * np.sqrt(self.com_gain) @ vel_error

        return control

    def compute_foot_control(self, desired_pos, current_pos, desired_vel, current_vel):
        # PD control for foot position
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel

        control = self.foot_gain @ pos_error + 2 * np.sqrt(self.foot_gain) @ vel_error

        return control

    def compute_orientation_control(self, desired_quat, current_quat, desired_omega, current_omega):
        # Control for torso orientation
        # Convert quaternions to rotation vectors for error calculation
        desired_rot = R.from_quat(desired_quat)
        current_rot = R.from_quat(current_quat)

        # Calculate orientation error
        error_rot = desired_rot * current_rot.inv()
        error_rotvec = error_rot.as_rotvec()

        # PD control for orientation
        omega_error = desired_omega - current_omega
        control = self.orientation_gain * error_rotvec + 2 * np.sqrt(self.orientation_gain) * omega_error

        return control

class HumanoidStepController:
    def __init__(self):
        self.step_phase = 'double_support'  # double_support, left_swing, right_swing
        self.step_duration = 1.0  # seconds
        self.current_time = 0.0

    def update_step_phase(self, dt):
        self.current_time += dt

        # Simple FSM for step phasing
        if self.step_phase == 'double_support':
            if self.current_time > self.step_duration * 0.1:  # 10% of step time
                self.step_phase = 'left_swing'  # Assuming left foot swings next
                self.current_time = 0.0
        elif self.step_phase == 'left_swing':
            if self.current_time > self.step_duration * 0.8:  # 80% of step time
                self.step_phase = 'double_support'
                self.current_time = 0.0
        elif self.step_phase == 'right_swing':
            if self.current_time > self.step_duration * 0.8:  # 80% of step time
                self.step_phase = 'double_support'
                self.current_time = 0.0

    def get_support_foot(self):
        if self.step_phase == 'left_swing':
            return 'right'  # Right foot is support
        elif self.step_phase == 'right_swing':
            return 'left'   # Left foot is support
        else:  # double_support
            return 'both'
```

### Walking Pattern Generation
Generate walking patterns for humanoid locomotion:

```python
class WalkingPatternGenerator:
    def __init__(self):
        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters (clearance)
        self.walking_speed = 0.5  # m/s
        self.step_period = 1.0  # seconds

        # ZMP (Zero Moment Point) parameters
        self.zmp_margin = 0.05  # Safety margin

    def generate_walk_pattern(self, distance, direction='forward'):
        # Generate walking pattern for a given distance
        steps_needed = int(distance / self.step_length)

        walk_pattern = []

        for i in range(steps_needed):
            # Calculate step position based on gait cycle
            step_info = self.calculate_step_position(i, direction)
            walk_pattern.append(step_info)

        return walk_pattern

    def calculate_step_position(self, step_index, direction):
        # Calculate position for a specific step
        step_info = {}

        # Forward progression
        forward_offset = step_index * self.step_length

        # Lateral alternation (left-right)
        if step_index % 2 == 0:  # Left foot
            lateral_offset = self.step_width / 2
            foot_side = 'left'
        else:  # Right foot
            lateral_offset = -self.step_width / 2
            foot_side = 'right'

        step_info['position'] = np.array([forward_offset, lateral_offset, 0.0])
        step_info['foot_side'] = foot_side
        step_info['timing'] = step_index * self.step_period
        step_info['height'] = self.step_height  # Step clearance

        return step_info

    def generate_com_trajectory(self, walk_pattern):
        # Generate CoM trajectory that maintains balance during walking
        com_trajectory = []

        for i, step in enumerate(walk_pattern):
            # Calculate CoM position based on step location
            # This is a simplified inverted pendulum model

            # CoM should be positioned over the support foot
            support_foot_pos = self.get_support_foot_position(walk_pattern, i)

            # Calculate CoM position with safety margin
            com_x = support_foot_pos[0]  # Follow support foot in x
            com_y = support_foot_pos[1]  # Follow support foot in y with margin
            com_z = self.com_height  # Maintain constant height

            com_pos = np.array([com_x, com_y, com_z])
            com_trajectory.append({
                'time': step['timing'],
                'position': com_pos,
                'velocity': self.calculate_com_velocity(com_pos, i, com_trajectory)
            })

        return com_trajectory

    def get_support_foot_position(self, walk_pattern, current_step_idx):
        # Determine which foot is in support at this step
        if current_step_idx == 0:
            # First step - assume starting stance
            return np.array([0.0, self.step_width/2, 0.0])  # Left foot position
        else:
            # Return the position of the stance foot
            # For simplicity, assume alternating stance
            if current_step_idx % 2 == 0:
                # Even steps: right foot is stance (for a left step pattern)
                prev_step = walk_pattern[current_step_idx - 1]
                return prev_step['position']
            else:
                # Odd steps: left foot is stance
                prev_step = walk_pattern[current_step_idx - 1]
                return prev_step['position']
```

## Integration with Perception Systems

### Perception-Driven Navigation
Integrate navigation with perception data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import numpy as np
import cv2
from scipy.ndimage import binary_dilation

class PerceptionIntegratedNavigation(Node):
    def __init__(self):
        super().__init__('perception_integrated_navigation')

        # Subscriptions
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/points', self.pointcloud_callback, 10
        )

        # Publishers
        self.local_map_pub = self.create_publisher(OccupancyGrid, '/local_map', 10)
        self.obstacle_markers_pub = self.create_publisher(MarkerArray, '/obstacles', 10)

        # Navigation components
        self.local_map_resolution = 0.05  # 5cm resolution
        self.local_map_size = 10.0  # 10m x 10m
        self.local_map = None

        # Obstacle detection
        self.obstacle_threshold = 0.3  # meters
        self.obstacle_buffer = 0.5  # meters

    def lidar_callback(self, msg):
        # Process LiDAR data to detect obstacles
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Filter out invalid ranges
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # Convert to Cartesian coordinates
        x_points = valid_ranges * np.cos(valid_angles)
        y_points = valid_ranges * np.sin(valid_angles)

        # Update local map with obstacle information
        self.update_local_map_with_lidar(x_points, y_points)

    def pointcloud_callback(self, msg):
        # Process point cloud data for more detailed obstacle detection
        # This would typically use libraries like PCL
        # For simplicity, we'll use a basic conversion
        pass

    def update_local_map_with_lidar(self, x_points, y_points):
        # Update local costmap with LiDAR data
        if self.local_map is None:
            self.initialize_local_map()

        # Convert world coordinates to map coordinates
        map_x = ((x_points - self.local_map.info.origin.position.x) /
                 self.local_map_resolution).astype(int)
        map_y = ((y_points - self.local_map.info.origin.position.y) /
                 self.local_map_resolution).astype(int)

        # Filter points within map bounds
        valid_mask = (
            (map_x >= 0) & (map_x < self.local_map.info.width) &
            (map_y >= 0) & (map_y < self.local_map.info.height)
        )

        # Update costmap
        for x, y in zip(map_x[valid_mask], map_y[valid_mask]):
            idx = y * self.local_map.info.width + x
            if 0 <= idx < len(self.local_map.data):
                # Mark as occupied (100 = definitely occupied)
                self.local_map.data[idx] = 100

        # Apply obstacle inflation
        self.inflate_obstacles()

        # Publish updated map
        self.local_map_pub.publish(self.local_map)

    def initialize_local_map(self):
        # Initialize local costmap
        from nav_msgs.msg import OccupancyGrid
        from geometry_msgs.msg import Point

        self.local_map = OccupancyGrid()
        self.local_map.header.frame_id = 'map'
        self.local_map.info.resolution = self.local_map_resolution
        self.local_map.info.width = int(self.local_map_size / self.local_map_resolution)
        self.local_map.info.height = int(self.local_map_size / self.local_map_resolution)

        # Set origin to robot's current position (simplified)
        self.local_map.info.origin.position.x = -self.local_map_size / 2
        self.local_map.info.origin.position.y = -self.local_map_size / 2

        # Initialize with unknown (value -1)
        self.local_map.data = [-1] * (self.local_map.info.width * self.local_map.info.height)

    def inflate_obstacles(self):
        # Inflate obstacles to account for robot size and safety margin
        if self.local_map is None:
            return

        # Convert to numpy array for processing
        grid = np.array(self.local_map.data).reshape(
            self.local_map.info.height, self.local_map.info.width
        )

        # Create binary mask of occupied cells
        occupied = (grid > 50)  # Threshold for "occupied"

        # Calculate inflation radius in pixels
        inflation_radius = int(self.obstacle_buffer / self.local_map_resolution)

        # Dilate occupied areas
        if inflation_radius > 0:
            from scipy.ndimage import binary_dilation
            structure = np.ones((inflation_radius*2+1, inflation_radius*2+1))
            inflated = binary_dilation(occupied, structure=structure)
        else:
            inflated = occupied

        # Update grid with inflated obstacles
        grid[inflated] = 100  # Mark as definitely occupied

        # Convert back to flat list
        self.local_map.data = grid.flatten().astype(int).tolist()

    def get_path_with_perception(self, start, goal):
        # Plan path considering perception data
        # This would typically call a path planner that uses the local map
        pass
```

## Navigation in Complex Environments

### Multi-Layer Navigation
Handle navigation in complex environments with multiple levels:

```python
class MultiLevelNavigation:
    def __init__(self):
        self.navigation_layers = {
            'ground': self.create_ground_navigation(),
            'stairs': self.create_stair_navigation(),
            'ramps': self.create_ramp_navigation(),
            'obstacles': self.create_obstacle_navigation()
        }

    def create_ground_navigation(self):
        # Standard ground navigation with Nav2
        config = {
            'planner': 'NavfnROS',
            'controller': 'DWBLocalPlanner',
            'recovery_behaviors': ['spin', 'backup']
        }
        return config

    def create_stair_navigation(self):
        # Specialized navigation for stairs
        config = {
            'planner': 'StepPlanner',  # Custom planner for stairs
            'controller': 'StairController',  # Custom controller
            'step_height_threshold': 0.15,  # Max step height
            'approach_distance': 0.5  # Distance to approach stairs
        }
        return config

    def create_ramp_navigation(self):
        # Navigation for ramps and inclines
        config = {
            'planner': 'RampPlanner',
            'controller': 'RampController',
            'max_incline': 0.3,  # Maximum incline (30%)
            'traction_control': True
        }
        return config

    def create_obstacle_navigation(self):
        # Navigation around dynamic obstacles
        config = {
            'planner': 'TEBPlanner',  # Timed Elastic Band for dynamic obstacles
            'controller': 'MPCController',  # Model Predictive Control
            'prediction_horizon': 3.0,
            'obstacle_buffer': 0.8
        }
        return config

    def select_navigation_mode(self, environment_data):
        # Determine appropriate navigation mode based on environment
        if self.is_stairs_present(environment_data):
            return 'stairs'
        elif self.is_ramp_present(environment_data):
            return 'ramps'
        elif self.is_dynamic_obstacles_present(environment_data):
            return 'obstacles'
        else:
            return 'ground'

    def is_stairs_present(self, env_data):
        # Check if stairs are detected in environment
        # This would analyze elevation data, point clouds, etc.
        return False  # Simplified

    def is_ramp_present(self, env_data):
        # Check if ramps are detected
        return False  # Simplified

    def is_dynamic_obstacles_present(self, env_data):
        # Check for moving obstacles
        return False  # Simplified
```

### Adaptive Navigation
Adapt navigation behavior based on environmental conditions:

```python
class AdaptiveNavigation:
    def __init__(self):
        self.current_mode = 'normal'
        self.adaptation_thresholds = {
            'crowded': 0.7,  # Crowd density threshold
            'narrow': 0.8,   # Narrow passage threshold
            'dynamic': 0.5   # Dynamic obstacle threshold
        }

    def adapt_navigation_parameters(self, environment_state):
        # Adapt navigation parameters based on environment
        new_mode = self.determine_navigation_mode(environment_state)

        if new_mode != self.current_mode:
            self.update_navigation_configuration(new_mode)
            self.current_mode = new_mode

    def determine_navigation_mode(self, env_state):
        # Determine appropriate navigation mode
        if env_state.get('crowd_density', 0) > self.adaptation_thresholds['crowded']:
            return 'social_navigation'
        elif env_state.get('narrow_passage', False):
            return 'careful_navigation'
        elif env_state.get('dynamic_obstacles', 0) > self.adaptation_thresholds['dynamic']:
            return 'reactive_navigation'
        else:
            return 'normal'

    def update_navigation_configuration(self, mode):
        # Update Nav2 configuration based on mode
        if mode == 'social_navigation':
            # Increase personal space buffer
            # Adjust speed for social compliance
            pass
        elif mode == 'careful_navigation':
            # Reduce speed
            # Increase obstacle clearance
            pass
        elif mode == 'reactive_navigation':
            # Increase sensor update frequency
            # Reduce prediction horizon
            pass
```

## Performance Considerations

### Computational Optimization
Optimize navigation for real-time performance:

```python
import threading
import time
from collections import deque

class OptimizedNavigation:
    def __init__(self):
        # Threading for parallel processing
        self.path_planning_thread = threading.Thread(target=self.path_planning_loop)
        self.path_planning_thread.daemon = True
        self.path_planning_thread.start()

        # Caching for repeated calculations
        self.path_cache = {}
        self.cache_size_limit = 100

        # Performance monitoring
        self.planning_times = deque(maxlen=50)
        self.execution_times = deque(maxlen=50)

    def path_planning_loop(self):
        # Separate thread for path planning
        while True:
            # Check for new planning requests
            if self.has_planning_request():
                start_time = time.time()
                self.process_planning_request()
                planning_time = time.time() - start_time
                self.planning_times.append(planning_time)

    def has_planning_request(self):
        # Check if there's a planning request
        return False  # Simplified

    def process_planning_request(self):
        # Process the planning request
        pass

    def optimize_map_representation(self, map_resolution):
        # Use hierarchical maps for different planning needs
        # Global: low resolution for long-term planning
        # Local: high resolution for obstacle avoidance
        pass

    def get_performance_metrics(self):
        # Get performance metrics
        avg_planning_time = sum(self.planning_times) / len(self.planning_times) if self.planning_times else 0
        return {
            'avg_planning_time': avg_planning_time,
            'planning_frequency': len(self.planning_times) / 50 if len(self.planning_times) == 50 else 0
        }
```

## Nav2 Configuration Files and Motion Control Parameters Examples

### Complete Nav2 Configuration for Humanoid Robot
Full example of Nav2 configuration for humanoid navigation:

```yaml
# Complete Nav2 configuration for humanoid robot
bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Humanoid-specific BT
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_have_feedback_condition_bt_node
      - nav2_have_odom_condition_bt_node
      - nav2_have_costmap_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_is_battery_charging_condition_bt_node

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIC"
      debug_trajectory_details: False
      control_frequency: 20.0
      velocity_scaling_tolerance: 0.1
      velocity_scaling_min: 0.05
      scaling_mechanism: "none"
      max_scaling_factor: 1.0
      # Humanoid-specific parameters
      max_linear_speed: 0.3  # Slower for balance
      max_angular_speed: 0.5
      linear_proportional_gain: 2.0
      angular_proportional_gain: 2.0
      transform_tolerance: 0.3
      use_cost_regulated_linear_velocity_scaling: True
      cost_scaling_dist: 0.6
      cost_scaling_gain: 1.0
      inflation_cost_scaling_factor: 3.0
      global_path_resolution: 0.1
      goal_dist_tolerance: 0.25  # Larger for humanoid
      goal_yaw_tolerance: 0.25
      simulate_to_goal: False

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: False
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05  # Higher resolution for humanoid
      origin_x: 0.0
      origin_y: 0.0
      # Humanoid-specific inflation
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0  # Higher for humanoid safety
        inflation_radius: 0.8     # Larger for humanoid footprint
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: False
      robot_radius: 0.3  # Larger for humanoid
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 2.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5  # Larger tolerance for humanoid
      use_astar: false
      allow_unknown: true

smoother_server:
  ros__parameters:
    use_sim_time: False
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_behaviors/Spin"
      spin_dist: 1.57  # 90 degrees for humanoid
    backup:
      plugin: "nav2_behaviors/BackUp"
      backup_dist: 0.15  # Shorter backup for humanoid
      backup_speed: 0.025
    wait:
      plugin: "nav2_behaviors/Wait"
      wait_duration: 1s
```

### Motion Control Parameters
Example motion control configuration for humanoid robots:

```yaml
# Humanoid motion control parameters
humanoid_motion_control:
  ros__parameters:
    # Balance control parameters
    balance_control:
      com_height: 0.8
      com_gain_p: [5.0, 5.0, 0.0]  # x, y, z
      com_gain_d: [2.2, 2.2, 0.0]  # x, y, z
      support_polygon_margin: 0.05
      max_lean_angle: 0.2

    # Walking gait parameters
    walking_gait:
      step_length: 0.3
      step_width: 0.2
      step_height: 0.05
      step_period: 1.0
      walking_speed: 0.3
      step_timing:
        double_support_ratio: 0.1
        swing_phase_ratio: 0.8

    # Footstep planning
    footstep_planning:
      max_step_length: 0.4
      max_step_width: 0.3
      min_step_length: 0.1
      step_clearance: 0.05
      terrain_adaptation: true
      step_cost_weights:
        distance: 1.0
        rotation: 0.5
        terrain: 2.0

    # Joint control
    joint_control:
      stiffness:
        hip: 500.0
        knee: 400.0
        ankle: 300.0
        arm: 200.0
      damping_ratio: 0.7
      position_tolerance: 0.01
      velocity_tolerance: 0.1

    # Safety limits
    safety_limits:
      max_linear_velocity: 0.5
      max_angular_velocity: 0.5
      max_acceleration: 0.5
      fall_threshold_angle: 0.5
      emergency_stop_timeout: 2.0
```

## Prerequisites

To effectively work with Nav2 for humanoid navigation, you should have:
- Understanding of ROS/ROS2 navigation concepts
- Basic knowledge of mobile robotics and path planning
- Familiarity with humanoid robot kinematics and dynamics
- Experience with motion control and balance concepts
- Understanding of sensor integration for navigation

This chapter provides the foundation for implementing navigation systems with Nav2 for humanoid robots, including path planning that accounts for dynamic constraints and motion control for stable locomotion.