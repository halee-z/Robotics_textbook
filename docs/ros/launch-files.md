---
sidebar_position: 4
---

# ROS 2 Launch Files and Workspaces for Humanoid Robotics

## Overview

Launch files and workspaces are essential organizational tools in ROS 2 that enable the coordination of multiple nodes for complex humanoid robot systems. This section covers how to create and structure launch files and workspaces specifically for humanoid robotics applications.

## ROS 2 Workspaces

A ROS 2 workspace is a directory containing packages that are built and used together. For humanoid robotics, we typically organize workspaces to manage:

- Robot-specific packages
- Simulation packages
- Perception packages
- Control packages
- Hardware interface packages

### Workspace Structure

```
humanoid_ws/                 # Root workspace directory
├── src/                     # Source code directory
│   ├── humanoid_bringup/    # Launch and configuration files
│   ├── humanoid_description # Robot URDF and mesh files
│   ├── humanoid_control/    # Controllers and control algorithms
│   ├── humanoid_perception/ # Perception algorithms
│   ├── humanoid_navigation/ # Navigation algorithms
│   └── humanoid_sim/        # Simulation-specific packages
```

### Creating a Workspace

```bash
# Create the workspace directory
mkdir -p ~/humanoid_ws/src

# Navigate to the workspace
cd ~/humanoid_ws

# Build all packages in the workspace
colcon build --packages-select humanoid_description humanoid_control

# Source the workspace
source install/setup.bash
```

## Launch Files for Humanoid Robots

Launch files use Python or XML to specify which nodes should be started together with what parameters. For humanoid robots, we often need to launch:

- Robot state publisher
- Joint state publisher
- Control managers
- Perception nodes
- Simulation interfaces

### Python Launch Files

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_description_path = LaunchConfiguration(
        'robot_description_path',
        default=os.path.join(
            get_package_share_directory('humanoid_description'),
            'urdf',
            'humanoid.urdf'
        )
    )

    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': ParameterValue(robot_description_path, value_type=str)}
        ]
    )

    # Launch joint state publisher (GUI for testing)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    # Launch controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description_path,
            os.path.join(
                get_package_share_directory('humanoid_control'),
                'config',
                'controllers.yaml'
            )
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Use GUI for joint state publisher'
        ),
        robot_state_publisher,
        joint_state_publisher_gui,
        controller_manager
    ])
```

### Launch Configuration Files

Launch files often use YAML configuration files for robot-specific parameters:

```yaml
# controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    humanoid_joint_publisher:
      type: joint_state_broadcaster/JointStateBroadcaster

    humanoid_controller:
      type: humanoid_controller/HumanoidController

humanoid_controller:
  ros__parameters:
    # Balance control parameters
    kp_balance: [100.0, 100.0, 100.0]  # x, y, z position gains
    kd_balance: [10.0, 10.0, 10.0]     # x, y, z velocity gains
    
    # Walking gait parameters
    step_height: 0.1                    # meters
    step_length: 0.3                    # meters
    step_duration: 1.0                  # seconds
```

## Complex Launch Scenarios for Humanoid Robots

### Simulation Launch

```python
# launch/simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    # Launch Gazebo simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )
    
    # Spawn robot in simulation
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot'
        ],
        output='screen'
    )
    
    # Launch robot control nodes
    robot_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_control'),
                'launch',
                'control.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        gazebo,
        spawn_entity,
        robot_control_launch
    ])
```

### Real Robot Launch

```python
# launch/real_robot.launch.py
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Set environment variables for real robot
    set_robot_ip = SetEnvironmentVariable(
        name='ROBOT_IP',
        value=os.environ.get('ROBOT_IP', '192.168.1.10')
    )
    
    # Launch hardware interface
    hardware_interface = Node(
        package='humanoid_hardware_interface',
        executable='hardware_interface_node',
        name='hardware_interface',
        parameters=[
            os.path.join(
                get_package_share_directory('humanoid_hardware_interface'),
                'config',
                'hardware.yaml'
            )
        ],
        remappings=[
            ('/joint_commands', '/position_controller/commands'),
            ('/joint_states', '/joint_states')
        ]
    )
    
    # Launch robot monitoring
    robot_monitor = Node(
        package='humanoid_monitoring',
        executable='robot_monitor',
        name='robot_monitor',
        parameters=[
            {'robot_name': 'humanoid_robot'},
            {'check_battery': True},
            {'check_temperature': True}
        ]
    )
    
    return LaunchDescription([
        set_robot_ip,
        hardware_interface,
        robot_monitor
    ])
```

## Parameter Management

For humanoid robots with many configurable parameters, proper parameter management is crucial:

```yaml
# config/robot_parameters.yaml
humanoid_robot:
  ros__parameters:
    # Physical properties
    mass: 50.0  # kg
    height: 1.5  # meters
    com_height: 0.8  # meters from ground
    
    # Control parameters
    control_loop_rate: 100  # Hz
    max_joint_velocity: 2.0  # rad/s
    max_joint_torque: 100.0  # Nm
    
    # Safety parameters
    max_tilt_angle: 0.5  # rad, beyond which robot stops
    emergency_stop_timeout: 1.0  # seconds without control before emergency stop
```

## Launch File Best Practices

### Organization
- Separate launch files for different scenarios (simulation vs. real robot)
- Use includes to reuse common launch components
- Group related nodes in logical sets

### Parameter Handling
- Use launch arguments for configurable options
- Load robot-specific parameters from YAML files
- Validate parameters at startup

### Error Handling
- Implement proper shutdown procedures
- Use event handlers for managing node dependencies
- Include monitoring and recovery mechanisms

## Practical Exercise

1. Create a launch file for a humanoid robot that includes:
   - Robot state publisher
   - Joint state broadcaster
   - A simple walking controller
   - A basic perception node

2. Use the educational AI to help design launch arguments that allow switching between simulation and real robot modes

3. Create a parameter file with realistic values for a humanoid robot's physical and control properties

The next section will cover workspace management tools and best practices for maintaining complex humanoid robot software systems.