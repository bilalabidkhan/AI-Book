---
title: Python Agents with ROS 2
description: Practical Python examples using rclpy for connecting AI agents to robot controllers
tags: [python, rclpy, ai-agents, robotics, ros2]
---

# Python Agents with ROS 2

[ROS 2 Fundamentals](./ros2-fundamentals.md) | [Humanoid Modeling with URDF](./humanoid-modeling-urdf.md)

## Introduction

This chapter focuses on connecting AI agents to robot controllers using Python and the rclpy client library. Python's simplicity and rich ecosystem make it an excellent choice for developing AI agents that interface with ROS 2-based robotic systems.

The rclpy library provides Python bindings for ROS 2, allowing Python programs to participate in ROS 2 communication patterns as nodes. This enables seamless integration of AI algorithms with robotic hardware and simulation environments.

## rclpy Basics

rclpy is the Python client library for ROS 2. It provides the necessary interfaces to create ROS 2 nodes, publish and subscribe to topics, make service calls, and provide services.

### Core Components

- **Node**: The basic execution unit that can communicate with other nodes
- **Publisher**: Sends messages to topics
- **Subscriber**: Receives messages from topics
- **Client**: Makes service requests
- **Service**: Provides service responses
- **Timer**: Executes callbacks at specified intervals

### Initialization Pattern

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2 communications
    node = MyNode()        # Create node instance
    rclpy.spin(node)       # Keep node alive
    node.destroy_node()    # Clean up
    rclpy.shutdown()       # Shutdown ROS 2 communications
```

## Connecting AI Logic to Robot Controllers

### Architecture Overview

The connection between AI agents and robot controllers involves several key components:

1. **AI Agent**: Contains the decision-making logic
2. **ROS 2 Interface**: Handles communication with the robot
3. **Robot Controller**: Executes commands on the physical or simulated robot
4. **Sensors**: Provide feedback to the AI agent

### Communication Patterns

AI agents typically use multiple communication patterns:

- **Topics** for streaming sensor data and continuous commands
- **Services** for discrete actions and configuration
- **Actions** for goal-oriented tasks with feedback

## Practical Python Examples

### Simple AI Agent Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class SimpleAIAgent(Node):
    def __init__(self):
        super().__init__('simple_ai_agent')

        # Create subscriber for laser scan data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10)

        # Create publisher for velocity commands
        self.publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        # Timer for AI decision making
        self.timer = self.create_timer(0.1, self.ai_decision_loop)

        self.laser_data = None
        self.get_logger().info('Simple AI Agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg.ranges

    def ai_decision_loop(self):
        """Main AI decision making logic"""
        if self.laser_data is None:
            return

        # Simple obstacle avoidance logic
        cmd = Twist()

        # Check for obstacles in front
        front_ranges = self.laser_data[330:30] + self.laser_data[330:360]  # Wrap around
        min_distance = min(front_ranges)

        if min_distance < 1.0:  # Obstacle within 1 meter
            cmd.angular.z = 0.5  # Turn right
        else:
            cmd.linear.x = 0.5   # Move forward

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    agent = SimpleAIAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced AI Integration with External Libraries

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class ComputerVisionAgent(Node):
    def __init__(self):
        super().__init__('computer_vision_agent')

        # Create subscriber for camera images
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)

        # Create publisher for velocity commands
        self.publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        # Timer for processing
        self.timer = self.create_timer(0.05, self.process_frame)

        self.bridge = CvBridge()
        self.current_image = None
        self.target_found = False

        self.get_logger().info('Computer Vision Agent initialized')

    def image_callback(self, msg):
        """Convert ROS image message to OpenCV format"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_frame(self):
        """Process current frame and make decisions"""
        if self.current_image is None:
            return

        # Simple color-based object detection
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)

        # Define range for red color (example)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cmd = Twist()

        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
                # Calculate center of contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Calculate error from center of image
                    img_center_x = self.current_image.shape[1] // 2
                    error_x = cx - img_center_x

                    # Simple proportional control
                    cmd.angular.z = -error_x * 0.005  # Turn toward object
                    cmd.linear.x = 0.3  # Move forward
                    self.target_found = True
                else:
                    self.target_found = False
            else:
                self.target_found = False
        else:
            self.target_found = False
            # Search for target if not found
            cmd.angular.z = 0.5  # Turn slowly to search

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    agent = ComputerVisionAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sending Commands to Robot Controllers

### Velocity Commands

```python
from geometry_msgs.msg import Twist

def send_velocity_command(self, linear_x=0.0, angular_z=0.0):
    """Send velocity command to robot"""
    cmd = Twist()
    cmd.linear.x = linear_x
    cmd.angular.z = angular_z
    self.cmd_vel_publisher.publish(cmd)
```

### Joint Position Commands

```python
from std_msgs.msg import Float64MultiArray

def send_joint_positions(self, positions):
    """Send joint position commands"""
    msg = Float64MultiArray()
    msg.data = positions
    self.joint_cmd_publisher.publish(msg)
```

### Custom Action Commands

```python
from rclpy.action import ActionClient
from example_interfaces.action import FollowJointTrajectory

class ActionCommander(Node):
    def __init__(self):
        super().__init__('action_commander')
        self._action_client = ActionClient(self, FollowJointTrajectory, 'joint_trajectory_controller/follow_joint_trajectory')

    def send_trajectory_goal(self, trajectory_points):
        """Send trajectory goal to controller"""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory_points

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        """Handle action feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Current trajectory progress: {feedback}')
```

## Receiving Sensor Feedback from Robots

### Handling Sensor Messages

```python
def sensor_callback(self, msg):
    """Generic sensor callback"""
    self.last_sensor_data = msg
    self.process_sensor_data()

def process_sensor_data(self):
    """Process sensor data for AI agent"""
    if self.last_sensor_data:
        # Apply AI logic to sensor data
        ai_output = self.ai_model.process(self.last_sensor_data)
        self.send_commands(ai_output)
```

### Multiple Sensor Fusion

```python
class SensorFusionAgent(Node):
    def __init__(self):
        super().__init__('sensor_fusion_agent')

        # Multiple sensor subscribers
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        self.imu_data = None
        self.odom_data = None
        self.scan_data = None

    def imu_callback(self, msg):
        self.imu_data = msg
        self.update_sensor_fusion()

    def odom_callback(self, msg):
        self.odom_data = msg
        self.update_sensor_fusion()

    def scan_callback(self, msg):
        self.scan_data = msg
        self.update_sensor_fusion()

    def update_sensor_fusion(self):
        """Combine sensor data for AI decision making"""
        if all([self.imu_data, self.odom_data, self.scan_data]):
            # Fuse sensor data
            fused_state = self.fuse_sensors(self.imu_data, self.odom_data, self.scan_data)
            # Make AI decisions based on fused state
            ai_decision = self.ai_model.decide(fused_state)
            # Send commands based on decision
            self.execute_decision(ai_decision)
```

## Step-by-Step Instructions for Creating a Basic Python Agent

### 1. Project Setup

Create a new ROS 2 Python package for your AI agent:

```bash
# Create package
ros2 pkg create --build-type ament_python my_ai_agent
cd my_ai_agent
```

### 2. Package Dependencies

Update `setup.py` with required dependencies:

```python
entry_points={
    'console_scripts': [
        'my_agent = my_ai_agent.my_agent:main',
    ],
},
```

### 3. Basic Agent Structure

```python
# my_ai_agent/my_agent.py
import rclpy
from rclpy.node import Node

class MyAIAgent(Node):
    def __init__(self):
        super().__init__('my_ai_agent')
        # Initialize your agent components here
        self.get_logger().info('AI Agent initialized')

def main(args=None):
    rclpy.init(args=args)
    agent = MyAIAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4. Running the Agent

```bash
# Build and source the workspace
colcon build
source install/setup.bash

# Run the agent
ros2 run my_ai_agent my_agent
```

## Conclusion

Python agents with ROS 2 provide a powerful platform for connecting AI algorithms to robotic systems. The rclpy library enables seamless integration of Python-based AI with ROS 2's distributed architecture, allowing for complex robot behaviors that can adapt and learn from sensor feedback.

By following the patterns and examples in this chapter, you can create AI agents that effectively interface with robot controllers, process sensor data, and execute complex robotic tasks.