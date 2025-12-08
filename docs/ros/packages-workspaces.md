---
sidebar_position: 5
---

# ROS 2 Packages and Workspaces for Humanoid Robotics

## Package Organization for Humanoid Robots

ROS 2 packages are the fundamental units of software organization in ROS. For humanoid robotics, effective package organization is crucial due to the complexity and interdependence of different subsystems.

### Core Package Categories

A humanoid robot system typically includes packages in these categories:

#### Robot Description
- **Purpose**: Contains URDF files, mesh files, and materials
- **Package naming**: `humanoid_description`, `robot_name_description`
- **Contents**:
  - URDF files defining robot kinematics
  - Mesh files for visual and collision geometry
  - Material definitions
  - Inertial parameters

#### Control Systems
- **Purpose**: Implement controllers for robot motion
- **Package naming**: `humanoid_control`, `robot_name_control`
- **Contents**:
  - Controller configurations
  - Custom controller implementations
  - Balance control algorithms
  - Walking pattern generators

#### Hardware Interface
- **Purpose**: Interface with physical robot hardware
- **Package naming**: `humanoid_hardware_interface`, `robot_name_hardware`
- **Contents**:
  - ros2_control hardware interface implementations
  - Communication protocols with actuators
  - Sensor interface implementations

#### Perception Systems
- **Purpose**: Process sensor data for environment understanding
- **Package naming**: `humanoid_perception`, `robot_name_perception`
- **Contents**:
  - Camera processing nodes
  - Object detection algorithms
  - Mapping algorithms
  - SLAM implementations

#### Navigation and Planning
- **Purpose**: Plan and execute robot motion
- **Package naming**: `humanoid_navigation`, `robot_name_navigation`
- **Contents**:
  - Path planning algorithms
  - Motion planning implementations
  - Footstep planning for walking robots
  - Trajectory execution

#### Simulation
- **Purpose**: Support robot simulation in various environments
- **Package naming**: `humanoid_sim`, `robot_name_gazebo`
- **Contents**:
  - Gazebo plugins
  - Simulation configurations
  - Sensor simulation implementations
  - World files

## Package Structure

A well-structured ROS 2 package for humanoid robotics follows this layout:

```
humanoid_package/
├── CMakeLists.txt          # Build configuration
├── package.xml            # Package metadata
├── config/                # Configuration files
│   ├── controllers.yaml
│   ├── parameters.yaml
│   └── hardware.yaml
├── launch/                # Launch files
│   ├── control.launch.py
│   └── bringup.launch.py
├── src/                   # Source code (for C++ packages)
├── scripts/               # Python scripts
├── include/               # Header files (for C++ packages)
├── meshes/                # 3D mesh files
├── urdf/                  # URDF description files
└── test/                  # Test files
```

### Package.xml for Humanoid Robotics

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_control</name>
  <version>1.0.0</version>
  <description>Control package for humanoid robot with walking and balance control</description>
  <maintainer email="maintainer@robotics.edu">Robotics Lab</maintainer>
  <license>MIT</license>

  <url type="website">http://robotics.ros.org/humanoid_control</url>
  <author email="developer@robotics.edu">Robotics Developer</author>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>control_msgs</depend>
  <depend>controller_interface</depend>
  <depend>hardware_interface</depend>
  <depend>pluginlib</depend>
  <depend>realtime_tools</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Workspace Management

### Creating a Humanoid Robot Workspace

Creating a workspace for humanoid robotics involves several steps:

1. **Initialize the workspace**:
   ```bash
   mkdir -p ~/humanoid_ws/src
   cd ~/humanoid_ws
   ```

2. **Add packages**:
   ```bash
   # Clone or copy packages to src/
   cd src
   git clone https://github.com/organization/humanoid_description.git
   git clone https://github.com/organization/humanoid_control.git
   # ... additional packages
   ```

3. **Build the workspace**:
   ```bash
   cd ~/humanoid_ws
   colcon build --packages-select humanoid_description humanoid_control
   # Or build all packages:
   colcon build
   ```

4. **Source the workspace**:
   ```bash
   source install/setup.bash
   # Or add to ~/.bashrc for permanent sourcing:
   echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc
   ```

### Build Optimization for Complex Systems

For complex humanoid robot systems with many packages, consider these optimization strategies:

```bash
# Build with parallel jobs (using all CPU cores)
colcon build --parallel-workers $(nproc)

# Build specific packages to speed up iteration
colcon build --packages-select humanoid_control humanoid_perception

# Use isolated builds to isolate package dependencies
colcon build --packages-select humanoid_description --isolated

# Build with verbose output to debug issues
colcon build --event-handlers console_direct+
```

## Dependency Management

Humanoid robot systems often have complex dependencies between packages:

### C++ Package Dependencies (CMakeLists.txt)
```cmake
cmake_minimum_required(VERSION 3.8)
project(humanoid_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)

# Declare library
add_library(humanoid_controller SHARED
  src/humanoid_controller.cpp
)

# Link dependencies
target_link_libraries(humanoid_controller)
ament_target_dependencies(humanoid_controller
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  control_msgs
  controller_interface
  hardware_interface
)

# Install targets
install(TARGETS humanoid_controller
  LIBRARY DESTINATION lib
)

ament_export_libraries(humanoid_controller)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  control_msgs
  controller_interface
  hardware_interface
)

ament_package()
```

### Python Package Dependencies (setup.py)
```python
from setuptools import setup
import os
from glob import glob

package_name = 'humanoid_control_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.py'))),
        # Include config files
        (os.path.join('share', package_name, 'config'), 
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robotics Lab',
    maintainer_email='maintainer@robotics.edu',
    description='Python control package for humanoid robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'humanoid_controller = humanoid_control_py.controller_node:main',
        ],
    },
)
```

## Package Development Workflow

### Version Control for Robot Packages

For humanoid robot packages, consider this Git workflow:

```bash
# Create a feature branch for new functionality
git checkout -b feature/walking-controller

# Make changes to implement walking controller
# ...

# Commit changes with descriptive messages
git add .
git commit -m "Implement basic walking controller with ZMP-based gait"

# Run tests before pushing
colcon test --packages-select humanoid_control
colcon test-result --verbose

# Push and create pull request
git push origin feature/walking-controller
```

### Testing Strategies

Test packages thoroughly, especially for humanoid robots where failure can cause physical damage:

```cpp
// Example C++ test for a humanoid controller
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "humanoid_control/humanoid_controller.hpp"

class TestHumanoidController : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rclcpp::init(0, nullptr);
        controller_ = std::make_shared<HumanoidController>();
    }

    void TearDown() override
    {
        rclcpp::shutdown();
    }

    std::shared_ptr<HumanoidController> controller_;
};

TEST_F(TestHumanoidController, TestBalanceControl)
{
    // Test that the controller can maintain balance
    EXPECT_TRUE(controller_->is_stable());
    
    // Apply perturbation
    controller_->apply_perturbation(10.0);  // 10N force
    
    // Check that controller recovers balance
    EXPECT_TRUE(controller_->is_stable_after_control());
}
```

## Best Practices for Humanoid Robotics Packages

### Naming Conventions
- Use descriptive, consistent names across packages
- Prefix robot-specific packages: `robot_name_description`, `robot_name_control`
- Use underscores for multi-word package names

### Documentation
- Document all public APIs and interfaces
- Include usage examples in package README files
- Document configuration parameters and their effects

### Code Quality
- Follow ROS 2 coding standards (Cpp and Python)
- Use static analysis tools (cppcheck, cpplint, flake8, etc.)
- Implement comprehensive unit tests

### Safety
- Implement safety checks in all control packages
- Include emergency stop functionality
- Validate inputs to prevent dangerous commands

## Practical Exercise

1. Using the educational AI agents, design a package structure for a new humanoid robot project that includes:
   - Robot description package
   - Control package
   - Perception package
   - Simulation package

2. Create a sample `package.xml` file for one of these packages with appropriate dependencies and metadata.

3. Design a launch file that would bring up all necessary nodes for a basic walking demonstration.

The next section will explore how to integrate these packages with the broader educational platform.