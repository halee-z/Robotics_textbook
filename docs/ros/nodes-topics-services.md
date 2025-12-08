---
sidebar_position: 3
---

# Nodes, Topics, and Services in ROS 2

## Understanding ROS 2 Communication

ROS 2 uses a distributed computing architecture where different processes (nodes) communicate with each other through a publish-subscribe pattern (topics), request-response pattern (services), and goal-oriented pattern (actions).

For humanoid robots, this communication architecture is crucial for coordinating:
- Sensor data processing
- Control command distribution
- Perception systems
- Planning and execution modules

## Nodes in Humanoid Robotics

In ROS 2, a node is an executable that uses ROS 2 client libraries to communicate with other nodes. In humanoid robotics systems, common nodes include:

### Sensor Processing Nodes
- Camera image processing
- IMU data filtering
- LIDAR point cloud processing
- Joint position/velocity/effort monitoring

### Control Nodes
- Walking pattern generation
- Whole-body controllers
- Trajectory execution
- Balance maintenance

### Perception Nodes
- Object detection
- Human detection
- Environment mapping
- Semantic segmentation

### Planning Nodes
- Path planning
- Motion planning
- Task planning
- Grasp planning

## Topics for Humanoid Robots

Topics are named buses over which nodes exchange messages. In humanoid robots, the following topics are commonly used:

### Joint State Information
```bash
/joint_states
```
Publishes joint positions, velocities, and efforts for all joints. Used by the robot state publisher to generate transforms.

### Transform Information
```bash
/tf
/tf_static
```
Publishes coordinate frame transforms. Essential for spatial relationships between different parts of the robot and between robot and environment.

### Actuator Commands
```bash
/joint_group_position_controller/commands
/effort_controller/commands
```
Sends commands to robot actuators to achieve desired positions, velocities, or efforts.

### Sensor Data
```bash
/camera/image_raw
/imu/data
/scan
```
Provides raw sensor data that other nodes process for perception and control.

### Robot Commands
```bash
/cmd_vel
/joint_trajectory
```
Accepts navigation or manipulation commands from higher-level controllers or user interfaces.

## Services for Humanoid Robots

Services provide a request-reply communication pattern. In humanoid robots, common services include:

### System Configuration
- `set_parameters`: Dynamically configure node parameters
- `get_parameter_types`: Query parameter types
- `describe_parameters`: Get parameter descriptions

### Robot Calibration
- `calibrate_sensors`: Calibrate IMU, cameras, etc.
- `zero_force_torque`: Zero force/torque sensors
- `find_joint_zero`: Find mechanical zero positions

### Control Services
- `start_trajectory`: Begin executing a trajectory
- `stop_trajectory`: Stop current trajectory
- `set_control_mode`: Change controller operation mode

## Actions for Humanoid Robots

Actions are a goal-based communication pattern suitable for long-running tasks:

### Navigation
- `move_base`: Navigate to a goal position
- `follow_waypoints`: Follow a sequence of waypoints

### Manipulation
- `pick`: Execute a pick action
- `place`: Execute a place action
- `move_group`: Plan and execute arm movements

### Locomotion
- `walk_to`: Navigate with walking gait
- `step`: Execute a single stepping motion

## Quality of Service (QoS) Considerations

For humanoid robots, proper QoS configuration is crucial for reliable operation:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# For critical control data (must have delivery, low latency)
control_qos = QoSProfile(
    history=QoSHistoryPolicy.RMW_QOS_HISTORY_KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE
)

# For sensor data (can lose packets, need latest values)
sensor_qos = QoSProfile(
    history=QoSHistoryPolicy.RMW_QOS_HISTORY_KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE
)
```

## Best Practices for Humanoid Robots

### Topic Design
- Use standard message types when possible
- Consider message rate for real-time performance
- Implement proper frame IDs for transforms
- Use compressed topics for high-bandwidth data

### Service Design
- Use services for state-changing operations
- Implement proper error handling
- Consider service timeouts
- Document request and response formats

### Node Organization
- Follow the principle of single responsibility
- Organize nodes by function (perception, control, etc.)
- Use launch files to coordinate multiple nodes
- Implement proper shutdown procedures

## Practical Exercise

Use the ROS 2 Specialist AI agent to explore:

1. How would you design a node structure for a humanoid robot that needs to walk, avoid obstacles, and respond to voice commands?
2. What topics would you need for a humanoid robot performing a simple pick-and-place task?
3. How would you use ROS 2 tools to monitor the communication between nodes during robot operation?

The next section will explore ROS 2 launch files and how to coordinate complex humanoid robot systems.