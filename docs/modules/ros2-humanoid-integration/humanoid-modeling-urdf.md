---
title: Humanoid Modeling with URDF
description: URDF modeling examples with links, joints, kinematics, and physical properties
tags: [urdf, humanoid, modeling, robotics, ros2, kinematics]
---

# Humanoid Modeling with URDF

[ROS 2 Fundamentals](./ros2-fundamentals.md) | [Python Agents with ROS 2](./python-agents-ros2.md)

## Introduction

Unified Robot Description Format (URDF) is the standard XML format used in ROS for representing robot models. This chapter covers the fundamentals of creating humanoid robot models with URDF, including defining links, joints, kinematics, and physical properties essential for humanoid robot simulation and control.

URDF enables the description of a robot's physical structure, including its kinematic chains, visual representation, collision properties, and dynamics. For humanoid robots, URDF is crucial for creating accurate models that can be used in simulation environments and for planning robot movements.

## URDF Basics

URDF (Unified Robot Description Format) is an XML-based format that describes a robot's structure and properties. A URDF file contains:

- **Links**: Rigid parts of the robot body
- **Joints**: Connections between links
- **Visual**: How the robot looks in visualization tools
- **Collision**: How the robot interacts with the environment in physics simulation
- **Inertial**: Mass properties for physics simulation

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
  </joint>

  <link name="wheel_link">
    <!-- ... -->
  </link>
</robot>
```

## Links in URDF

Links represent rigid bodies in the robot. Each link can have multiple elements that define its properties.

### Link Components

1. **Visual**: Defines how the link appears in visualization
2. **Collision**: Defines how the link interacts with the environment in simulation
3. **Inertial**: Defines the physical properties for dynamics simulation

### Visual Properties

The visual element describes the appearance of a link:

```xml
<link name="head_link">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
</link>
```

### Collision Properties

The collision element defines how the link interacts with the environment:

```xml
<link name="arm_link">
  <collision>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <geometry>
      <box size="0.05 0.05 0.2"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

The inertial element defines the mass properties for physics simulation:

```xml
<link name="torso_link">
  <inertial>
    <mass value="2.0"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

## Joints in URDF

Joints define the connection between links and specify how they can move relative to each other.

### Joint Types

1. **revolute**: Rotational joint with limits
2. **continuous**: Rotational joint without limits
3. **prismatic**: Linear sliding joint with limits
4. **fixed**: No movement between links
5. **floating**: 6DOF movement (for base of floating robots)
6. **planar**: Movement on a plane

### Joint Definition

```xml
<joint name="shoulder_joint" type="revolute">
  <parent link="torso_link"/>
  <child link="upper_arm_link"/>
  <origin xyz="0.1 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

### Joint Limits and Properties

- **limit**: Defines position, velocity, and effort limits
- **safety_controller**: Defines safety limits for real robot operation
- **calibration**: Defines joint position when the calibration sensor is triggered
- **dynamics**: Defines joint friction and damping

## Kinematics in URDF

Kinematics refers to the geometric relationships between links and joints, describing how the robot moves without considering forces.

### Forward Kinematics

Forward kinematics calculates the position and orientation of the end effector based on joint angles. URDF defines the kinematic structure that enables forward kinematics calculations.

### Inverse Kinematics

Inverse kinematics calculates the required joint angles to achieve a desired end-effector position. URDF provides the kinematic chain structure needed for IK solvers.

### Kinematic Chains

```xml
<!-- Example of a simple arm kinematic chain -->
<link name="base_link"/>
<joint name="joint1" type="revolute">
  <parent link="base_link"/>
  <child link="link1"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
<link name="link1"/>
<joint name="joint2" type="revolute">
  <parent link="link1"/>
  <child link="link2"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
<link name="link2"/>
```

## Visual Properties in URDF

Visual properties define how a robot appears in simulation and visualization tools.

### Geometry Types

1. **Box**: Rectangular prism
2. **Cylinder**: Cylindrical shape
3. **Sphere**: Spherical shape
4. **Mesh**: Complex 3D model from external file

### Visual with Mesh Example

```xml
<link name="head_link">
  <visual>
    <geometry>
      <mesh filename="package://my_robot_description/meshes/head.stl" scale="1 1 1"/>
    </geometry>
    <material name="white">
      <color rgba="1 1 1 1"/>
    </material>
  </visual>
</link>
```

### Materials

Materials define the visual appearance of links:

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>
<material name="blue">
  <color rgba="0 0 1 1"/>
</material>
<material name="black">
  <color rgba="0 0 0 1"/>
</material>
```

## Collision Properties in URDF

Collision properties define how a robot interacts with the environment in physics simulation.

### Collision vs Visual

- **Visual** elements are for appearance and rendering
- **Collision** elements are for physics simulation
- They can use the same geometry or different simplified geometries

### Collision Example

```xml
<link name="torso_link">
  <collision>
    <geometry>
      <box size="0.3 0.2 0.5"/>
    </geometry>
  </collision>
  <visual>
    <geometry>
      <mesh filename="package://my_robot_description/meshes/torso.dae"/>
    </geometry>
  </visual>
</link>
```

## Inertial Properties in URDF

Inertial properties define the mass distribution of each link, essential for accurate physics simulation.

### Mass and Center of Mass

- **mass**: Total mass of the link
- **origin**: Location of the center of mass relative to the link frame

### Inertia Tensor

The inertia tensor describes how mass is distributed in the link. For a solid box:

```
ixx = (m * (h² + d²)) / 12
iyy = (m * (w² + d²)) / 12
izz = (m * (w² + h²)) / 12
```

Where m = mass, w = width, h = height, d = depth.

### Inertial Example

```xml
<link name="link1">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
  </inertial>
</link>
```

## Practical URDF Examples

### Simple Humanoid Torso

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
    </inertial>
  </link>
</robot>
```

### Humanoid Arm with Multiple Joints

```xml
<!-- Upper Arm -->
<link name="upper_arm">
  <visual>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
    <origin xyz="0 0 0.15" rpy="1.57079632679 0 0"/>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
    <origin xyz="0 0 0.15" rpy="1.57079632679 0 0"/>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<!-- Elbow Joint -->
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="2.0" effort="20" velocity="1"/>
</joint>

<!-- Lower Arm -->
<link name="lower_arm">
  <visual>
    <geometry>
      <cylinder length="0.25" radius="0.04"/>
    </geometry>
    <origin xyz="0 0 0.125" rpy="1.57079632679 0 0"/>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.25" radius="0.04"/>
    </geometry>
    <origin xyz="0 0 0.125" rpy="1.57079632679 0 0"/>
  </collision>
  <inertial>
    <mass value="0.8"/>
    <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.007"/>
  </inertial>
</link>

<!-- Wrist Joint -->
<joint name="wrist_joint" type="revolute">
  <parent link="lower_arm"/>
  <child link="hand"/>
  <origin xyz="0 0 0.25" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
</joint>

<!-- Hand -->
<link name="hand">
  <visual>
    <geometry>
      <box size="0.1 0.08 0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.1 0.08 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.3"/>
    <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0004" iyz="0.0" izz="0.0003"/>
  </inertial>
</link>
```

## Conclusion

URDF modeling is fundamental to humanoid robot development in ROS. By properly defining links, joints, kinematics, and physical properties, you create accurate robot models essential for simulation, visualization, and control.

Understanding URDF enables you to create complex humanoid robots with realistic kinematic chains and physical properties that can be used in simulation environments, for motion planning, and for connecting AI agents to robot controllers effectively.