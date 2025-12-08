---
sidebar_position: 2
---

# Chapter 2 Exercises: ROS 2 for Humanoid Robotics

## Exercise 1: ROS 2 Node Development for Robot Control

### Objective
Create a ROS 2 node that controls a humanoid robot's walking gait by publishing joint trajectories to the appropriate topics.

### Instructions
1. Create a new ROS 2 package called `humanoid_walker`
2. Implement a publisher node that publishes joint trajectory messages
3. Use `trajectory_msgs/JointTrajectory` to send coordinated commands to both legs
4. Implement a simple walking gait pattern with alternating leg movements
5. Test your node in simulation

### Tasks
1. Set up the package structure with proper dependencies
2. Create the publisher node with appropriate ROS parameters
3. Implement the trajectory generation algorithm
4. Test with a humanoid robot simulation
5. Document your implementation

### Requirements
- Node must publish to `/joint_trajectory` controller
- Should handle parameter changes (walking speed, step size)
- Must include safety checks to prevent dangerous joint positions
- Should be able to start/stop walking on command

### Questions
1. What are the key joints involved in humanoid walking?
2. How do you ensure the robot maintains balance during walking?
3. What parameters would you expose for customizing the gait?
4. How would you implement obstacle avoidance in your walking controller?

---

## Exercise 2: Sensor Integration and State Estimation

### Objective
Develop a ROS 2 node that integrates IMU and joint state sensors to estimate the robot's center of mass and zero moment point (ZMP) for balance control.

### Instructions
1. Create a subscriber node that listens to `/imu/data` and `/joint_states` topics
2. Implement algorithms to calculate center of mass position
3. Compute the zero moment point based on CoM and gravity
4. Publish the results to a new topic `/balance_state`
5. Visualize the results in RViz

### Tasks
1. Subscribe to IMU and joint state topics
2. Implement center of mass calculation using robot URDF
3. Compute ZMP from CoM and acceleration data
4. Add filtering to smooth the estimates
5. Publish balance state messages

### Requirements
- Subscribe to `sensor_msgs/Imu` and `sensor_msgs/JointState` topics
- Use robot_description for kinematic calculations
- Implement appropriate filtering (e.g., Kalman filter or moving average)
- Publish `geometry_msgs/Point` for CoM and ZMP positions
- Include timestamp synchronization between sensor sources

### Questions
1. Why is sensor fusion important for humanoid balance?
2. How does the center of mass calculation depend on joint positions?
3. What are the limitations of ZMP control for dynamic movements?
4. How could you incorporate force/torque sensor data if available?

---

## Exercise 3: Vision-Based Object Detection and Grasping

### Objective
Create a ROS 2 node that uses camera input to detect objects and coordinates with the arm controller for grasping.

### Instructions
1. Develop a node that subscribes to camera image topics
2. Implement object detection using OpenCV or a DNN
3. Calculate the 3D position of detected objects using depth information
4. Interface with a motion planner to generate grasp trajectories
5. Test the complete pipeline in simulation

### Tasks
1. Set up image and depth image subscribers
2. Implement object detection algorithm
3. Convert 2D image coordinates to 3D world coordinates
4. Plan and execute arm motion to grasp object
5. Integrate with existing grasp planning frameworks

### Requirements
- Subscribe to `sensor_msgs/Image` and optionally depth topics
- Use `cv_bridge` for image processing
- Implement coordinate transformation from camera to robot base frame
- Interface with moveit_commander or similar planning framework
- Include safety checks during grasping

### Questions
1. How do you handle coordinate transformations in ROS?
2. What are the challenges of 3D position estimation from 2D images?
3. How would you verify that a grasp was successful?
4. What are the trade-offs between different object detection approaches?

---

## Exercise 4: Multi-Node Coordination for Complex Tasks

### Objective
Design a system of coordinated ROS 2 nodes that work together to complete a multi-step humanoid robot task (e.g., navigate to a location, identify an object, and pick it up).

### Instructions
1. Create multiple specialized nodes:
   - Navigation node
   - Perception node  
   - Manipulation node
   - Task coordinator node
2. Design message interfaces between nodes
3. Implement a state machine for task coordination
4. Handle failure cases and recovery
5. Test the complete system

### Tasks
1. Design the communication architecture between nodes
2. Implement individual specialized nodes
3. Create the task coordinator logic
4. Add error handling and recovery mechanisms
5. Test the complete task pipeline

### Requirements
- Each node should have a single, well-defined responsibility
- Use ROS services for synchronous communication when needed
- Use topics for continuous data streams
- Implement proper error handling and state management
- Include logging and monitoring capabilities

### Questions
1. What are the advantages of decomposing complex tasks into multiple nodes?
2. How do you handle timing constraints between nodes?
3. What challenges arise when coordinating multiple control systems?
4. How would you extend this system to handle multiple robots?

---

## Exercise 5: Advanced Control with Real-time Performance

### Objective
Implement a ROS 2 node that provides real-time control for humanoid robot balance, meeting strict timing requirements.

### Instructions
1. Create a high-frequency control loop (200Hz+) for balance control
2. Implement a controller that maintains zero moment point within support polygon
3. Use real-time scheduling if possible
4. Profile and optimize the performance
5. Add safety mechanisms to handle control failures

### Tasks
1. Implement the high-frequency control loop
2. Develop the balance control algorithm
3. Optimize code for real-time performance
4. Add monitoring and safety features
5. Test response times and stability

### Requirements
- Control loop must run at specified frequency (e.g., 200Hz)
- Use real-time ROS features if available
- Implement PID or other appropriate control algorithm
- Include safety checks to prevent dangerous outputs
- Monitor and log control performance metrics

### Questions
1. What are the challenges of achieving real-time performance in ROS?
2. How do you ensure control loop timing in a multitasking system?
3. What are the key considerations for safety in real-time robot control?
4. How would you handle communication delays in the control loop?

---

## Challenge Exercise: Humanoid Robot Teleoperation Interface

### Objective
Design and implement a complete ROS 2-based system for teleoperating a humanoid robot, including haptic feedback and multiple sensory modalities.

### Instructions
1. Develop a user input interface (joystick, keyboard, or VR)
2. Create robot-side control nodes that process teleoperation commands
3. Implement sensory feedback systems (visual, audio, haptic)
4. Add safety features to protect the robot and environment
5. Demonstrate the complete system in simulation

### Tasks
1. Design the user interaction interface
2. Implement the robot-side command processing
3. Create sensory feedback mechanisms
4. Add collision avoidance and safety systems
5. Test and refine the complete teleoperation experience

### Advanced Requirements
- Implement force feedback to the operator
- Include multiple camera views for enhanced situational awareness
- Add speech interface for commanding the robot
- Implement shared autonomy features
- Create a failsafe system for emergency stops

### Questions
1. How do you manage the complexity of multi-modal teleoperation interfaces?
2. What are the communication and latency challenges in teleoperation?
3. How do you maintain operator situational awareness in remote operation?
4. What are the safety considerations for teleoperated humanoid robots?

## Solutions

Solutions for these exercises will be available after attempting the problems. Consider using the educational AI agents to help clarify concepts and verify your understanding. Each exercise builds on the fundamental concepts of ROS 2 while introducing humanoid-specific challenges and requirements.