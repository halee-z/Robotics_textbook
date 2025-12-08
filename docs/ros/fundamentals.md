---
sidebar_position: 2
---

# ROS 2 Fundamentals for Humanoid Robotics

## Overview

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. In the context of humanoid robotics, ROS 2 provides the communication infrastructure needed for the complex sensorimotor loops and distributed processing requirements.

## ROS 2 vs ROS 1: Key Differences

ROS 2 addresses several limitations of the original ROS, making it more suitable for humanoid robotics applications:

- **Real-Time Support**: Critical for humanoid robot control systems that require deterministic timing
- **Improved Security**: Essential when humanoid robots interact with humans in educational settings
- **Better Multi-Robot Support**: Necessary for scenarios involving multiple humanoid robots
- **Quality of Service (QoS) Settings**: Allows specification of delivery guarantees for different types of data
- **DDS-Based Communication**: Provides more robust and configurable communication patterns

## Core Concepts

### Nodes
In ROS 2, a node is a process that performs computation. Each node in a ROS graph can be written in different programming languages (C++, Python, etc.) and can run on different machines. For humanoid robots, common nodes include:

- Sensor processing nodes (IMU, cameras, LIDAR)
- Control nodes (walking, manipulation, balance)
- Perception nodes (object recognition, localization)
- Planning nodes (motion planning, path planning)

### Topics and Messages
Topics are named buses over which nodes exchange messages. In humanoid robotics, common topics include:

- `/joint_states` - Current joint positions, velocities, and efforts
- `/cmd_vel` - Velocity commands for base movement
- `/sensor_msgs/Image` - Camera image data
- `/tf` - Transform data for coordinate frames

### Services
Services provide a request/reply communication pattern. Common services in humanoid robotics include:

- `/set_parameters` - Dynamically configure robot parameters
- `/get_plans` - Request motion plans from planning services
- `/calibrate` - Service for sensor calibration

### Actions
Actions are a goal-based communication pattern suitable for long-running tasks. In humanoid robotics:

- `/move_base` - Send navigation goals
- `/joint_trajectory` - Execute complex joint movement sequences
- `/pick_place` - Perform manipulation tasks

## Setting Up a Humanoid Robot in ROS 2

### URDF Models
Unified Robot Description Format (URDF) is an XML format for representing robot models. A humanoid robot URDF typically includes:

- Kinematic tree describing joint connections
- Physical properties (mass, inertia)
- Visual and collision properties
- Sensor mountings

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  
  <!-- Add additional links for legs, arms, head, etc. -->
  <link name="left_leg">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.05"/>
      </geometry>
    </visual>
  </link>
  
  <joint name="base_to_left_leg" type="fixed">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <origin xyz="0 -0.2 -0.3" rpy="0 0 0"/>
  </joint>
</robot>
```

### Launch Files
Launch files allow you to start multiple nodes with a single command. For a humanoid robot:

```xml
<launch>
  <!-- Start robot state publisher -->
  <node pkg="robot_state_publisher" 
        exec="robot_state_publisher" 
        name="robot_state_publisher">
    <param name="robot_description" 
           value="$(var robot_description_file)"/>
  </node>

  <!-- Start joint state publisher -->
  <node pkg="joint_state_publisher_gui" 
        exec="joint_state_publisher_gui" 
        name="joint_state_publisher_gui"/>

  <!-- Start simulation or hardware interface -->
  <include file="$(find-pkg-share my_humanoid_description)/launch/gazebo.launch.py"/>
</launch>
```

## Best Practices for Humanoid Robotics

### Modular Design
Break down complex humanoid behaviors into smaller, reusable nodes. For example:

- Separate perception from planning from control
- Create specialized nodes for different locomotion patterns
- Use service nodes for calibration and configuration

### Error Handling
Implement robust error handling in all nodes:

- Establish timeouts for communication
- Handle sensor failures gracefully
- Implement safe states for robot control

### Performance Considerations
- Minimize message passing between nodes running on the same machine
- Use appropriate QoS settings for real-time requirements
- Profile nodes to identify bottlenecks

## Hands-on Exercise

Create a simple ROS 2 package that implements a basic walking pattern for a simulated humanoid robot:

1. Use the educational AI to generate ROS 2 code for a walking controller
2. Simulate the walking pattern in the simulation environment
3. Observe the robot's behavior and joint trajectories

The next section will cover ROS 2 packages and workspaces in greater detail, including how to structure code for humanoid robot applications.