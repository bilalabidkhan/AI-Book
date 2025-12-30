---
title: ROS 2 Fundamentals
description: Comprehensive guide to ROS 2 core concepts including nodes, topics, services, and actions
tags: [ros2, fundamentals, robotics, nodes, topics, services, actions]
---

# ROS 2 Fundamentals

[Python Agents with ROS 2](./python-agents-ros2.md) | [Humanoid Modeling with URDF](./humanoid-modeling-urdf.md)

## Introduction

Robot Operating System 2 (ROS 2) serves as the "Robotic Nervous System" for connecting AI agents to humanoid robot bodies. This chapter introduces the core concepts that form the foundation of ROS 2 architecture.

ROS 2 is not an actual operating system, but rather a flexible framework for writing robot software. It provides services designed specifically for a heterogeneous computer cluster such as hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.

## Nodes

In ROS 2, a node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system. Each node is designed to perform a specific task and can communicate with other nodes through topics, services, or actions.

### Purpose of Nodes

- Encapsulate robot functionality in modular units
- Enable distributed computation across multiple machines
- Allow for independent development and testing of robot capabilities
- Facilitate code reuse and sharing

### Node Implementation

Nodes can be written in various programming languages, with C++ and Python being the most commonly used. Each node initializes a ROS 2 client library, creates publishers/subscribers, and runs a processing loop.

## Topics

Topics enable asynchronous communication between nodes using a publish-subscribe pattern. This communication method is ideal for streaming data like sensor readings or motor commands.

### Publish-Subscribe Pattern

- Publishers send messages to a topic without knowing which nodes will receive them
- Subscribers receive messages from a topic without knowing which nodes are publishing
- Multiple publishers and subscribers can exist for the same topic
- Communication is decoupled in time and space

### Use Cases for Topics

- Sensor data distribution (camera images, LIDAR scans, IMU data)
- Motor command streaming
- Robot state broadcasting
- Diagnostic information sharing

## Services

Services provide synchronous request-response communication between nodes. This pattern is suitable for operations that have a clear input and output, and typically require a response.

### Request-Response Pattern

- A client sends a request and waits for a response
- The service processes the request and sends back a response
- Communication is synchronous and blocking
- Ideal for operations that must complete before continuing

### Use Cases for Services

- Parameter configuration
- Map loading/unloading
- Path planning requests
- Database queries
- Task execution confirmation

## Actions

Actions extend services with the ability to provide feedback during long-running operations and support for canceling operations. They are ideal for goal-oriented tasks.

### Action Features

- Goal: The desired outcome of the action
- Feedback: Periodic updates on the progress toward the goal
- Result: The final outcome of the action
- Cancel: Ability to interrupt a running action

### Use Cases for Actions

- Navigation to a specific location
- Object manipulation tasks
- Complex multi-step robot behaviors
- Long-duration operations requiring monitoring

## DDS-Based Communication

ROS 2 uses Data Distribution Service (DDS) as its communication middleware. DDS provides:

- Quality of Service (QoS) settings for different communication needs
- Discovery mechanisms for automatic node detection
- Reliable and best-effort delivery options
- Data persistence and lifecycle management
- Security features for protected communication

### QoS Settings

Quality of Service settings allow fine-tuning of communication behavior:

- Reliability: Reliable vs best-effort delivery
- Durability: Volatile vs transient-local data persistence
- History: Keep-all vs keep-last policies
- Deadline: Maximum time between consecutive messages
- Lifespan: Maximum time data remains valid

## Practical Examples

### Simple Node Communication via Topics

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    client = ServiceClient()
    response = client.send_request(1, 2)
    print(f'Result: {response.sum}')
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Conclusion

ROS 2 fundamentals form the foundation for building complex robotic systems. Understanding nodes, topics, services, and actions is essential for creating distributed robotic applications that can connect AI agents to humanoid robot bodies effectively.

These concepts enable the creation of modular, scalable, and maintainable robotic systems that can adapt to various hardware configurations and application requirements.