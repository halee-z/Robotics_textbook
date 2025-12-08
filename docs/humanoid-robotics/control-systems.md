---
sidebar_position: 2
---

# Control Systems for Humanoid Robots

## Overview

Control systems for humanoid robots are fundamentally more complex than traditional robotics due to the need for dynamic balance, multi-degree-of-freedom coordination, and real-time stability. Unlike wheeled or stationary robots, humanoid robots must manage their center of mass while performing tasks, making control design one of the most challenging aspects of humanoid robotics.

## Control Architecture Hierarchy

Humanoid control systems typically employ a hierarchical architecture with multiple layers:

```
High-Level Planning (1-10 Hz)
├── Walking pattern generation
├── Task planning and sequencing  
├── Footstep planning
└── Trajectory optimization

Mid-Level Control (50-200 Hz)
├── Balance control and recovery
├── Trajectory generation
├── Contact planning
└── Whole-body control

Low-Level Control (100-1000 Hz)
├── Joint position/velocity control
├── Torque control
├── Sensor feedback processing
└── Safety monitoring
```

## Zero Moment Point (ZMP) Theory

The Zero Moment Point (ZMP) is a fundamental concept in humanoid robotics that describes the point on the ground where the moment of the ground reaction force is zero. For a humanoid robot to maintain dynamic equilibrium, the ZMP must remain within the support polygon (usually the area defined by the feet).

### Mathematical Foundation

For a robot with center of mass (CoM) at position (x_com, y_com, z_com):

ZMP_x = x_com - (z_com/g) * x_com_dd
ZMP_y = y_com - (z_com/g) * y_com_dd

Where:
- g is gravitational acceleration
- x_com_dd and y_com_dd are the second derivatives (acceleration) of the CoM position

```python
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class ZMPController:
    def __init__(self, com_height=0.8, gravity=9.81):
        """
        Initialize ZMP controller with robot parameters
        
        Args:
            com_height: Initial estimate of center of mass height (meters)
            gravity: Gravitational acceleration (m/s^2)
        """
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)  # Natural frequency parameter
        
        # ZMP tracking error integral for PID control
        self.zmp_error_integral = np.array([0.0, 0.0])
        self.prev_zmp_error = np.array([0.0, 0.0])
        
        # PID controller parameters
        self.kp = 10.0  # Proportional gain
        self.ki = 1.0   # Integral gain
        self.kd = 2.0   # Derivative gain
        
        # Support polygon (simplified as rectangle for dual support)
        self.support_polygon = {
            "x_range": [-0.1, 0.1],   # 20cm in x direction
            "y_range": [-0.15, 0.15]  # 30cm in y direction
        }
        
    def calculate_zmp(self, com_pos, com_acc):
        """
        Calculate ZMP from current CoM position and acceleration
        
        Args:
            com_pos: [x, y, z] position of center of mass
            com_acc: [x, y, z] acceleration of center of mass
            
        Returns:
            [x, y] ZMP coordinates
        """
        x_com, y_com, z_com = com_pos
        x_acc, y_acc, z_acc = com_acc
        
        zmp_x = x_com - (z_com / self.gravity) * x_acc
        zmp_y = y_com - (z_com / self.gravity) * y_acc
        
        return np.array([zmp_x, zmp_y])
    
    def is_stable(self, zmp):
        """
        Check if ZMP is within support polygon
        
        Args:
            zmp: [x, y] ZMP coordinates
            
        Returns:
            Boolean indicating stability
        """
        zmp_x, zmp_y = zmp
        x_min, x_max = self.support_polygon["x_range"]
        y_min, y_max = self.support_polygon["y_range"]
        
        return x_min <= zmp_x <= x_max and y_min <= zmp_y <= y_max
    
    def balance_control(self, current_zmp, desired_zmp, dt=0.01):
        """
        Generate balance control commands using ZMP feedback
        
        Args:
            current_zmp: [x, y] current ZMP position
            desired_zmp: [x, y] desired ZMP position (usually center of support polygon)
            dt: Time step for integration
            
        Returns:
            Control output for CoM adjustment
        """
        # Calculate ZMP error
        zmp_error = desired_zmp - current_zmp
        
        # Update integral term with anti-windup
        self.zmp_error_integral += zmp_error * dt
        # Limit integral windup
        self.zmp_error_integral = np.clip(self.zmp_error_integral, -1.0, 1.0)
        
        # Calculate derivative term
        zmp_error_derivative = (zmp_error - self.prev_zmp_error) / dt if dt > 0 else np.array([0.0, 0.0])
        
        # PID control law
        control_output = (self.kp * zmp_error + 
                         self.ki * self.zmp_error_integral + 
                         self.kd * zmp_error_derivative)
        
        # Store previous error for next derivative calculation
        self.prev_zmp_error = zmp_error
        
        # Limit control output to reasonable values
        control_output = np.clip(control_output, -0.5, 0.5)  # Limit to 50cm adjustment
        
        return control_output
    
    def update_com_height(self, new_height):
        """
        Update CoM height when it changes (e.g., during walking)
        
        Args:
            new_height: New CoM height estimate
        """
        self.com_height = new_height
        self.omega = sqrt(self.gravity / self.com_height)

# Example usage
zmp_ctrl = ZMPController(com_height=0.85)

# Simulate balance control
current_zmp = np.array([0.05, 0.02])  # Slightly off-center
desired_zmp = np.array([0.0, 0.0])    # Center of support polygon
dt = 0.01  # 100Hz control

control_output = zmp_ctrl.balance_control(current_zmp, desired_zmp, dt)
print(f"Current ZMP: {current_zmp}")
print(f"Desired ZMP: {desired_zmp}")
print(f"Control output: {control_output}")
```

## Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model is a simplified representation of bipedal walking that assumes the robot's center of mass moves at a constant height:

```
ẍ = ω²(x - x_zmp)
```

Where ω = √(g/h) and h is the CoM height.

```python
import numpy as np
from scipy.integrate import odeint

class LinearInvertedPendulum:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)
        
    def dynamics(self, state, t, zmp_position):
        """
        Define the dynamics of the inverted pendulum
        
        Args:
            state: [x_position, x_velocity, y_position, y_velocity]
            t: Time
            zmp_position: [zmp_x, zmp_y] current ZMP position
        """
        x, dx, y, dy = state
        zmp_x, zmp_y = zmp_position
        
        # LIPM dynamics
        ddx = self.omega**2 * (x - zmp_x)
        ddy = self.omega**2 * (y - zmp_y)
        
        return [dx, ddx, dy, ddy]
    
    def simulate_trajectory(self, initial_state, zmp_trajectory, time_points):
        """
        Simulate CoM trajectory given ZMP reference
        
        Args:
            initial_state: [x0, vx0, y0, vy0] initial CoM state
            zmp_trajectory: Function that returns [zmp_x, zmp_y] for time t
            time_points: Array of time points to simulate
        
        Returns:
            Solution array with CoM trajectory
        """
        def wrapped_dynamics(state, t):
            zmp_pos = zmp_trajectory(t)
            return self.dynamics(state, t, zmp_pos)
        
        solution = odeint(wrapped_dynamics, initial_state, time_points)
        return solution
    
    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point - where to step to stop the robot
        
        Args:
            com_pos: [x, y] current CoM position
            com_vel: [vx, vy] current CoM velocity
            
        Returns:
            [x_cp, y_cp] capture point coordinates
        """
        x_com, y_com = com_pos
        vx_com, vy_com = com_vel
        
        capture_point_x = x_com + vx_com / self.omega
        capture_point_y = y_com + vy_com / self.omega
        
        return np.array([capture_point_x, capture_point_y])

# Example usage
lipm = LinearInvertedPendulum(com_height=0.85)

# Define a simple ZMP trajectory (constant ZMP for 1 second)
def zmp_ref(t):
    return np.array([0.0, 0.0])  # Constant ZMP at origin

# Initial conditions: CoM at (0.1, 0.05) with some velocity
initial_state = [0.1, 0.2, 0.05, 0.1]  # [x_pos, x_vel, y_pos, y_vel]

# Time vector (10 seconds at 100Hz)
time_points = np.linspace(0, 3, 300)

# Simulate
solution = lipm.simulate_trajectory(initial_state, zmp_ref, time_points)

# Calculate capture point at the beginning
capture_point = lipm.calculate_capture_point(
    [initial_state[0], initial_state[2]],  # x, y position
    [initial_state[1], initial_state[3]]   # vx, vy velocity
)

print(f"Initial capture point: ({capture_point[0]:.3f}, {capture_point[1]:.3f})")
print(f"Final CoM position: ({solution[-1, 0]:.3f}, {solution[-1, 2]:.3f})")
```

## Walking Pattern Generation

### Preview Control Approach

Preview control uses future ZMP references to generate stable CoM trajectories:

```python
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

class PreviewController:
    def __init__(self, com_height=0.8, dt=0.01, preview_window=2.0):
        """
        Initialize preview controller
        
        Args:
            com_height: Estimated CoM height
            dt: Control time step
            preview_window: How far ahead to look (in seconds)
        """
        self.com_height = com_height
        self.gravity = 9.81
        self.dt = dt
        self.omega = np.sqrt(self.gravity / com_height)
        
        # Number of steps to look ahead
        self.preview_steps = int(preview_window / dt)
        
        # Discrete state-space matrices for x-axis (same for y-axis)
        # State: x = [com_pos, com_vel]
        self.A = np.array([
            [1, dt],
            [self.omega**2 * dt, 1]
        ])
        
        self.B = np.array([0, -self.omega**2 * dt])
        
        # Output matrix: measure ZMP position
        self.C = np.array([1, -1/(self.omega**2)])  # ZMP = x - (1/ω²)*ẍ
        
        # Design LQR controller
        Q = np.array([[100, 0], [0, 1]])  # State cost
        R = 1  # Control cost
        
        # Solve discrete algebraic Riccati equation
        P = solve_discrete_are(self.A.T, self.C.T, Q, R)
        
        # Calculate LQR gain
        self.K = np.array([1]) / (self.B.T @ P @ self.B + R) @ self.B.T @ P @ self.A
    
    def compute_control(self, current_state, zmp_reference_sequence):
        """
        Compute control using preview control law
        
        Args:
            current_state: Current state [x, vx]
            zmp_reference_sequence: Sequence of future ZMP references
            
        Returns:
            Control input
        """
        # Feedback term
        feedback_control = -self.K @ current_state
        
        # Preview term (feedforward)
        preview_control = 0.0
        
        for k in range(min(len(zmp_reference_sequence), self.preview_steps)):
            # Calculate preview gain for step k
            A_pow = np.linalg.matrix_power(self.A, k)
            C_A_pow_B = self.C @ A_pow @ self.B
            
            # Weight decreases with preview horizon
            weight = np.exp(-0.1 * k)  # Exponential discounting
            
            # Calculate preview contribution
            if k < len(zmp_reference_sequence):
                preview_control += weight * (zmp_reference_sequence[k] - self.C @ A_pow @ current_state)
        
        return feedback_control + preview_control
    
    def generate_walking_trajectory(self, step_length=0.3, step_height=0.15, step_duration=1.0, steps=4):
        """
        Generate complete walking trajectory using preview control
        
        Args:
            step_length: Forward distance per step
            step_height: Maximum foot lift height
            step_duration: Time per step
            steps: Number of steps to generate
            
        Returns:
            Dictionary with CoM and foot trajectories
        """
        total_time = steps * step_duration
        time_points = np.arange(0, total_time, self.dt)
        
        # Initialize state
        state = np.array([0.0, 0.0])  # [com_pos, com_vel]
        
        # Trajectory arrays
        com_positions = []
        com_velocities = []
        zmp_positions = []
        
        # Generate ZMP reference for walking (alternating support)
        zmp_ref = self.generate_zmp_reference(step_length, step_duration, total_time)
        
        # Simulate walking
        for t in time_points:
            idx = int(t / self.dt)
            if idx >= len(zmp_ref):
                break
                
            # Get preview window of ZMP references
            end_preview_idx = min(idx + self.preview_steps, len(zmp_ref))
            zmp_preview = zmp_ref[idx:end_preview_idx]
            
            # Compute control
            u = self.compute_control(state, zmp_preview)
            
            # Update state using discrete dynamics
            state = self.A @ state + self.B * u
            
            # Calculate resulting ZMP
            current_zmp = self.C @ state
            
            # Store values
            com_positions.append(state[0])
            com_velocities.append(state[1])
            zmp_positions.append(current_zmp)
        
        # Generate foot trajectories synchronized with CoM
        left_foot_x, right_foot_x = self.generate_foot_trajectories(
            step_length, step_duration, steps, time_points
        )
        
        return {
            "time": time_points,
            "com_position": np.array(com_positions),
            "com_velocity": np.array(com_velocities),
            "zmp_position": np.array(zmp_positions),
            "left_foot_x": left_foot_x,
            "right_foot_x": right_foot_x,
            "step_times": [i * step_duration for i in range(steps)]
        }
    
    def generate_zmp_reference(self, step_length, step_duration, total_time):
        """
        Generate ZMP reference trajectory for walking
        """
        num_points = int(total_time / self.dt)
        zmp_ref = np.zeros(num_points)
        
        for i in range(num_points):
            t = i * self.dt
            step_num = int(t / step_duration)
            
            # Alternate ZMP between feet support
            if step_num % 2 == 0:  # Left foot support
                zmp_ref[i] = -0.05  # Slightly left of center
            else:  # Right foot support
                zmp_ref[i] = 0.05   # Slightly right of center
            
            # Add forward progression
            zmp_ref[i] += (step_num * step_length * 0.8)  # Scaled forward
        
        return zmp_ref
    
    def generate_foot_trajectories(self, step_length, step_duration, steps, time_array):
        """
        Generate foot trajectories synchronized with CoM
        """
        left_foot_x = np.zeros_like(time_array)
        right_foot_x = np.zeros_like(time_array)
        
        for i, t in enumerate(time_array):
            step_num = int(t / step_duration)
            step_progress = (t % step_duration) / step_duration
            
            # Calculate foot positions based on current step and phase
            if step_num % 2 == 0:  # Left foot swing
                # Left foot trajectory
                left_foot_x[i] = self.swing_trajectory(step_progress, step_num * step_length, (step_num + 1) * step_length)
                
                # Right foot stays in place during left foot swing
                right_foot_x[i] = step_num * step_length
            else:  # Right foot swing
                # Right foot trajectory
                right_foot_x[i] = self.swing_trajectory(step_progress, step_num * step_length, (step_num + 1) * step_length)
                
                # Left foot stays in place during right foot swing
                left_foot_x[i] = step_num * step_length
        
        return left_foot_x, right_foot_x
    
    def swing_trajectory(self, progress, start_pos, end_pos):
        """
        Generate foot swing trajectory using smooth interpolation
        """
        # Use 5th-order polynomial for smooth acceleration/deceleration
        # and zero velocity/acceleration at start/end
        if progress < 0:
            return start_pos
        if progress > 1:
            return end_pos
            
        # 5th order polynomial coefficients (ensures smooth start/end)
        # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        s = (-20*(progress**3) + 30*(progress**4) - 10*(progress**5))
        
        return start_pos + s * (end_pos - start_pos)

# Example usage
preview_ctrl = PreviewController(com_height=0.85, dt=0.01)

# Generate walking trajectory
walking_traj = preview_ctrl.generate_walking_trajectory(
    step_length=0.3,
    step_height=0.15,
    step_duration=1.0,
    steps=4
)

print(f"Generated trajectory for {len(walking_traj['step_times'])} steps")
print(f"Final CoM position: {walking_traj['com_position'][-1]:.3f} m")
```

## Whole-Body Control Framework

For complex humanoid robots, whole-body controllers coordinate all joints to achieve multiple simultaneous objectives:

```python
import numpy as np
from scipy.optimize import minimize

class WholeBodyController:
    def __init__(self, robot_model):
        self.robot_model = robot_model  # Kinematic/dynamic model
        self.tasks = []  # Priority-ordered tasks
        self.weights = []  # Task weights
        
    def add_task(self, task_function, priority, weight=1.0):
        """
        Add a control task with priority and weight
        
        Args:
            task_function: Function that returns task error
            priority: Integer priority (lower number = higher priority)
            weight: Weight for this task in optimization
        """
        self.tasks.append((priority, task_function, weight))
        # Sort by priority (ascending - higher priority first)
        self.tasks.sort(key=lambda x: x[0])
    
    def compute_control(self, current_state, reference_state):
        """
        Compute whole-body control using hierarchical optimization
        
        Args:
            current_state: Current robot state (joint positions, velocities)
            reference_state: Desired state for all joints
            
        Returns:
            Joint commands to achieve all tasks
        """
        # Organize tasks by priority
        high_priority_tasks = [(f, w) for p, f, w in self.tasks if p == 1]
        mid_priority_tasks = [(f, w) for p, f, w in self.tasks if p == 2]
        low_priority_tasks = [(f, w) for p, f, w in self.tasks if p == 3]
        
        # Higher priority tasks are solved first
        # For this example, we'll use a simplified sequential approach
        
        # Start with base posture (lowest priority)
        base_command = self.generate_posture_command(current_state, reference_state)
        
        # Add balance command (higher priority)
        balance_command = self.compute_balance_adjustment(current_state)
        
        # Add task-specific commands
        task_commands = []
        for task_func, weight in high_priority_tasks + mid_priority_tasks:
            task_cmd = task_func(current_state)
            task_commands.append(weight * task_cmd)
        
        # Combine all commands (with priority-based weighting)
        final_command = base_command + balance_command
        for cmd in task_commands:
            final_command += cmd
        
        return final_command
    
    def generate_posture_command(self, current_state, reference_state):
        """
        Generate command to maintain desired posture
        """
        # Simple PD control for posture maintenance
        posture_error = reference_state["joint_positions"] - current_state["joint_positions"]
        posture_command = 5.0 * posture_error + 1.0 * reference_state["joint_velocities"]  # PD controller
        return posture_command
    
    def compute_balance_adjustment(self, current_state):
        """
        Compute adjustments needed for balance maintenance
        """
        # Use ZMP controller to compute balance adjustments
        zmp_ctrl = ZMPController(com_height=0.85)
        
        # Get CoM state
        com_pos = current_state["com_position"]
        com_acc = current_state["com_acceleration"]
        
        # Calculate current ZMP
        current_zmp = zmp_ctrl.calculate_zmp(com_pos, com_acc)
        
        # Desired ZMP (usually center of support polygon)
        desired_zmp = np.array([0.0, 0.0])
        
        # Compute balance correction
        dt = 0.01  # Control time step
        balance_correction = zmp_ctrl.balance_control(current_zmp, desired_zmp, dt)
        
        # Convert balance correction to joint commands
        # This would involve inverse kinematics/dynamics
        # For now, return a simple representation
        joint_correction = np.zeros(len(current_state["joint_positions"]))
        
        # Apply correction to ankle, hip, and trunk joints for balance
        if len(joint_correction) >= 6:
            # Ankle joints (if available)
            joint_correction[-2] = balance_correction[0] * 0.1  # x direction
            joint_correction[-1] = balance_correction[1] * 0.1  # y direction
        
        return joint_correction
    
    def compute_task_jacobian(self, task_type, configuration):
        """
        Compute Jacobian for a specific task (e.g., end-effector position)
        
        Args:
            task_type: Type of task ('position', 'orientation', etc.)
            configuration: Joint configuration
            
        Returns:
            Jacobian matrix
        """
        # This would implement the actual kinematic Jacobian calculation
        # For this example, return a placeholder
        n_joints = len(configuration)
        if task_type == 'position':
            # 3D position task -> 3 x n_joints Jacobian
            return np.random.rand(3, n_joints)  # Placeholder
        elif task_type == 'orientation':
            # 3D orientation task -> 3 x n_joints Jacobian
            return np.random.rand(3, n_joints)  # Placeholder
        else:
            return np.zeros((1, n_joints))

# Example usage with a simple robot model
class SimpleRobotModel:
    def __init__(self, num_joints=12):
        self.num_joints = num_joints
        self.joint_limits = np.tile([-2.5, 2.5], (num_joints, 1))  # Joint limits
        self.mass_matrix = np.eye(num_joints) * 1.0  # Simplified mass matrix

simple_robot = SimpleRobotModel(num_joints=12)
wb_controller = WholeBodyController(simple_robot)

# Add tasks in order of priority
wb_controller.add_task(lambda state: np.zeros(12), priority=3, weight=0.1)  # Posture maintenance (low priority)
wb_controller.add_task(lambda state: np.zeros(12), priority=2, weight=1.0)  # Manipulation task (mid priority)
wb_controller.add_task(lambda state: np.zeros(12), priority=1, weight=10.0)  # Balance maintenance (high priority)

# Current and reference state (simplified)
current_state = {
    "joint_positions": np.random.rand(12),
    "joint_velocities": np.random.rand(12),
    "com_position": np.array([0.0, 0.0, 0.85]),
    "com_acceleration": np.array([0.1, 0.05, 0.0])
}

reference_state = {
    "joint_positions": np.zeros(12),
    "joint_velocities": np.zeros(12),
}

# Compute control
control_command = wb_controller.compute_control(current_state, reference_state)
print(f"Generated control command with {len(control_command)} joints")
print(f"Sample command values: {control_command[:5]}")

# Visualization of walking trajectory
plt.figure(figsize=(12, 8))

# Plot CoM trajectory
plt.subplot(2, 2, 1)
plt.plot(walking_traj["time"], walking_traj["com_position"], label="CoM X", linewidth=2)
plt.plot(walking_traj["time"], walking_traj["zmp_position"], label="ZMP X", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Center of Mass and ZMP Trajectory")
plt.legend()
plt.grid(True)

# Plot foot trajectories
plt.subplot(2, 2, 2)
plt.plot(walking_traj["time"], walking_traj["left_foot_x"], label="Left Foot", linewidth=2)
plt.plot(walking_traj["time"], walking_traj["right_foot_x"], label="Right Foot", linewidth=2)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label="Start position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Foot Trajectories")
plt.legend()
plt.grid(True)

# Plot CoM velocity
plt.subplot(2, 2, 3)
plt.plot(walking_traj["time"], walking_traj["com_velocity"], linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Center of Mass Velocity")
plt.grid(True)

# Show step timing
plt.subplot(2, 2, 4)
for step_time in walking_traj["step_times"]:
    plt.axvline(x=step_time, color='red', linestyle=':', alpha=0.7)
plt.plot(walking_traj["time"], [0]*len(walking_traj["time"]), alpha=0)  # Invisible baseline
plt.xlabel("Time (s)")
plt.title("Step Timing")
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Stability Analysis and Control

### Lyapunov Stability for Walking

Lyapunov functions can be used to analyze and ensure stability of walking controllers:

```python
class LyapunovWalkingStability:
    def __init__(self, com_height=0.85, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)
        
    def lyapunov_function(self, state):
        """
        Define Lyapunov function for walking stability
        
        State: [x_com, y_com, vx_com, vy_com, zmp_x, zmp_y]
        """
        x_com, y_com, vx_com, vy_com = state[:4]
        zmp_x, zmp_y = state[4:6]
        
        # Lyapunov function for LIPM: V = ½(ċom² + ω²(com - zmp)²)
        energy_term = 0.5 * (vx_com**2 + vy_com**2)  # Kinetic energy part
        potential_term = 0.5 * self.omega**2 * ((x_com - zmp_x)**2 + (y_com - zmp_y)**2)
        
        return energy_term + potential_term
    
    def lyapunov_derivative(self, state, state_derivative):
        """
        Calculate Lyapunov function derivative
        
        Args:
            state: Current system state
            state_derivative: Time derivative of state
        """
        x_com, y_com, vx_com, vy_com = state[:4]
        zmp_x, zmp_y = state[4:6]
        
        xdot, ydot, vxdot, vydot = state_derivative[:4]
        zmp_xdot, zmp_ydot = state_derivative[4:6]
        
        # Derivative of kinetic energy part
        kinetic_dv = vx_com * vxdot + vy_com * vydot
        
        # Derivative of potential energy part
        potential_dv = (self.omega**2 * 
                       ((x_com - zmp_x)*(vx_com - zmp_xdot) + 
                        (y_com - zmp_y)*(vy_com - zmp_ydot)))
        
        return kinetic_dv + potential_dv
    
    def is_asymptotically_stable(self, state, state_derivative, tolerance=0.01):
        """
        Check if system is asymptotically stable based on Lyapunov analysis
        """
        V = self.lyapunov_function(state)
        Vdot = self.lyapunov_derivative(state, state_derivative)
        
        # For asymptotic stability: V > 0 and Vdot < 0
        return V > tolerance and Vdot < -tolerance

# Example usage
stability_analyzer = LyapunovWalkingStability(com_height=0.85)

# Current state (simplified)
current_state = np.array([0.05, 0.02, 0.1, 0.05, 0.01, 0.0])  # [x_com, y_com, vx_com, vy_com, zmp_x, zmp_y]
state_deriv = np.array([0.1, 0.05, -0.5, -0.2, 0.0, 0.0])   # [ẋ, ẏ, v̇x, v̇y, żmp_x, żmp_y]

lyapunov_value = stability_analyzer.lyapunov_function(current_state)
lyapunov_derivative = stability_analyzer.lyapunov_derivative(current_state, state_deriv)
is_stable = stability_analyzer.is_asymptotically_stable(current_state, state_deriv)

print(f"Lyapunov function value: {lyapunov_value:.4f}")
print(f"Lyapunov derivative: {lyapunov_derivative:.4f}")
print(f"System asymptotically stable: {is_stable}")
```

## Adaptive Control for Humanoid Robots

### Model Reference Adaptive Control (MRAC)

```python
class ModelReferenceAdaptiveController:
    def __init__(self, reference_model_params, initial_controller_params):
        """
        Initialize MRAC system
        
        Args:
            reference_model_params: Desired closed-loop dynamics
            initial_controller_params: Initial estimates for controller params
        """
        self.reference_model = reference_model_params
        self.controller_params = initial_controller_params
        self.adaptation_rate = 0.01  # Learning rate
        
        # Storage for tracking errors and parameters
        self.state_error = 0
        self.param_error = 0
        self.integration_term = 0
    
    def update_model_reference(self, desired_response, actual_response):
        """
        Update reference model based on desired and actual responses
        """
        # In practice, this would involve more complex reference model adjustment
        pass
    
    def adapt_parameters(self, tracking_error, regressor_vector):
        """
        Adapt controller parameters based on tracking error
        
        Args:
            tracking_error: Difference between desired and actual output
            regressor_vector: Input vector that determines parameter adaptation
        """
        # Gradient descent adaptation law: θ̇ = -γ × φ × e
        param_adjustment = -self.adaptation_rate * regressor_vector * tracking_error
        self.controller_params += param_adjustment
        
        return self.controller_params
    
    def compute_control(self, state_error, input_signal):
        """
        Compute control using current parameters
        """
        # Simple parameterized controller: u = θ^T × φ(x,u)
        return self.controller_params.T @ input_signal

# Example implementation for joint control
class AdaptiveJointController:
    def __init__(self, joint_name, initial_params=np.array([1.0, 0.5])):
        self.joint_name = joint_name
        self.mrac = ModelReferenceAdaptiveController(
            reference_model_params={'natural_frequency': 10, 'damping_ratio': 0.7},
            initial_controller_params=initial_params
        )
        
    def update(self, desired_position, actual_position, dt=0.01):
        """
        Update controller with new measurements
        """
        # Calculate tracking error
        tracking_error = desired_position - actual_position
        
        # Define regressor vector (function of state that determines adaptation)
        regressor = np.array([actual_position, tracking_error])
        
        # Adapt parameters based on error
        updated_params = self.mrac.adapt_parameters(tracking_error, regressor)
        
        # Compute new control command
        control_signal = np.array([desired_position, tracking_error])
        control_output = self.mrac.compute_control(tracking_error, control_signal)
        
        return control_output, tracking_error

# Example usage
adaptive_ctrl = AdaptiveJointController("left_knee", initial_params=np.array([1.5, 0.8]))

# Simulate control loop
desired_pos = 0.5  # Desired joint position
actual_pos = 0.2   # Current joint position

control_out, error = adaptive_ctrl.update(desired_pos, actual_pos)
print(f"Adaptive control output: {control_out:.4f}")
print(f"Tracking error: {error:.4f}")
```

## Safety and Fault Tolerance

### Balance Recovery Control

```python
class BalanceRecoveryController:
    def __init__(self, zmp_controller, com_height=0.85):
        self.zmp_controller = zmp_controller
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)
        
        # Thresholds for emergency actions
        self.angle_threshold = 0.5  # 28.6 degrees
        self.velocity_threshold = 1.0
        self.zmp_threshold = 0.3  # 30cm outside support
        
        # Recovery strategies
        self.recovery_modes = {
            'ankle_strategy': {'priority': 1, 'max_torque': 50},
            'hip_strategy': {'priority': 2, 'max_torque': 100},
            'step_strategy': {'priority': 3, 'enabled': True}
        }
    
    def assess_imminent_fall(self, robot_state):
        """
        Assess if robot is at risk of falling
        
        Args:
            robot_state: Dictionary with robot status information
            
        Returns:
            Dictionary with fall risk assessment
        """
        risk_assessment = {
            'fall_risk_level': 'low',  # low, medium, high, imminent
            'risk_factors': [],
            'recovery_needed': False
        }
        
        # Check IMU angles (tilt)
        imu_angles = robot_state.get('imu_angles', [0, 0, 0])
        roll, pitch = imu_angles[0], imu_angles[1]
        
        if abs(roll) > self.angle_threshold or abs(pitch) > self.angle_threshold:
            risk_assessment['risk_factors'].append('excessive_tilt')
            risk_assessment['fall_risk_level'] = 'high'
            risk_assessment['recovery_needed'] = True
        
        # Check angular velocity
        angular_vel = robot_state.get('angular_velocity', [0, 0, 0])
        ang_roll_vel, ang_pitch_vel = angular_vel[0], angular_vel[1]
        
        if abs(ang_roll_vel) > self.velocity_threshold or abs(ang_pitch_vel) > self.velocity_threshold:
            risk_assessment['risk_factors'].append('high_angular_velocity')
            if risk_assessment['fall_risk_level'] == 'low':
                risk_assessment['fall_risk_level'] = 'medium'
            risk_assessment['recovery_needed'] = True
        
        # Check ZMP position relative to support polygon
        current_zmp = robot_state.get('zmp', [0, 0])
        support_polygon = {'x_range': [-0.1, 0.1], 'y_range': [-0.15, 0.15]}
        
        zmp_x, zmp_y = current_zmp
        x_min, x_max = support_polygon['x_range']
        y_min, y_max = support_polygon['y_range']
        
        zmp_margin_x = min(abs(zmp_x - x_min), abs(zmp_x - x_max))
        zmp_margin_y = min(abs(zmp_y - y_min), abs(zmp_y - y_max))
        
        if zmp_margin_x < 0.05 or zmp_margin_y < 0.05:  # Less than 5cm margin
            risk_assessment['risk_factors'].append('zmp_near_boundary')
            risk_assessment['fall_risk_level'] = 'high'
            risk_assessment['recovery_needed'] = True
        elif zmp_margin_x < 0.1 or zmp_margin_y < 0.1:  # Less than 10cm margin
            risk_assessment['risk_factors'].append('zmp_approaching_boundary')
            if risk_assessment['fall_risk_level'] == 'low':
                risk_assessment['fall_risk_level'] = 'medium'
        
        return risk_assessment
    
    def execute_recovery_strategy(self, risk_assessment, robot_state):
        """
        Execute appropriate recovery strategy based on risk level
        """
        if not risk_assessment['recovery_needed']:
            return {'action': 'none', 'commands': {}}
        
        # Determine appropriate recovery strategy
        highest_priority = max([self.recovery_modes[mode]['priority'] 
                               for mode in self.recovery_modes.keys()])
        
        recovery_commands = {}
        
        if risk_assessment['fall_risk_level'] in ['high', 'imminent']:
            # Immediate action needed
            if self.recovery_modes['step_strategy']['enabled']:
                # Step to new support location
                step_target = self.calculate_safe_step_location(robot_state)
                recovery_commands['step'] = step_target
                return {'action': 'step_recovery', 'commands': recovery_commands}
            
            # If step not available, try hip strategy
            elif self.recovery_modes['hip_strategy']['priority'] <= highest_priority:
                hip_command = self.calculate_hip_correction(robot_state)
                recovery_commands['hip_torques'] = hip_command
                return {'action': 'hip_recovery', 'commands': recovery_commands}
        
        # For medium risk, use ankle strategy
        else:
            ankle_command = self.calculate_ankle_correction(robot_state)
            recovery_commands['ankle_torques'] = ankle_command
            return {'action': 'ankle_recovery', 'commands': recovery_commands}
    
    def calculate_ankle_correction(self, robot_state):
        """
        Calculate ankle torques for small balance corrections
        """
        # Simple PD controller for ankle balance
        current_zmp = robot_state.get('zmp', [0, 0])
        desired_zmp = robot_state.get('desired_zmp', [0, 0])
        
        zmp_error = np.array(desired_zmp) - np.array(current_zmp)
        
        # Convert ZMP error to ankle torques (simplified model)
        ankle_torques = 200 * zmp_error  # Proportional gain
        ankle_torques = np.clip(ankle_torques, 
                               -self.recovery_modes['ankle_strategy']['max_torque'],
                                self.recovery_modes['ankle_strategy']['max_torque'])
        
        return ankle_torques
    
    def calculate_hip_correction(self, robot_state):
        """
        Calculate hip torques for larger balance corrections
        """
        # Similar approach but with higher torques and different kinematics
        current_com = robot_state.get('com_position', [0, 0, 0.85])
        desired_com = robot_state.get('desired_com', [0, 0, 0.85])
        
        com_error = np.array(desired_com)[:2] - np.array(current_com)[:2]  # x,y only
        
        hip_torques = 500 * com_error  # Higher gain for hip strategy
        hip_torques = np.clip(hip_torques,
                             -self.recovery_modes['hip_strategy']['max_torque'],
                              self.recovery_modes['hip_strategy']['max_torque'])
        
        return hip_torques
    
    def calculate_safe_step_location(self, robot_state):
        """
        Calculate safe step location to expand support polygon
        """
        current_zmp = robot_state.get('zmp', [0, 0])
        current_com = robot_state.get('com_position', [0, 0, 0.85])
        current_com_vel = robot_state.get('com_velocity', [0, 0, 0])
        
        # Use capture point concept to determine where to step
        capture_point_x = current_com[0] + current_com_vel[0] / self.omega
        capture_point_y = current_com[1] + current_com_vel[1] / self.omega
        
        # Choose step location near capture point but within reasonable range
        step_x = np.clip(capture_point_x, current_zmp[0] - 0.4, current_zmp[0] + 0.4)
        step_y = np.clip(capture_point_y, current_zmp[1] - 0.3, current_zmp[1] + 0.3)
        
        return [step_x, step_y, 0.0]  # [x, y, z]

# Example usage
zmp_ctrl = ZMPController(com_height=0.85)
recovery_ctrl = BalanceRecoveryController(zmp_ctrl)

# Simulate a potentially unstable state
robot_state = {
    'imu_angles': [0.6, 0.4, 0],  # Excessive roll and pitch (34.4° and 22.9°)
    'angular_velocity': [1.2, 0.8, 0.1],  # High angular velocities
    'zmp': [0.2, 0.1],  # ZMP near boundary
    'com_position': [0.05, 0.02, 0.85],
    'com_velocity': [0.3, 0.1, 0],
    'desired_zmp': [0.0, 0.0],
    'desired_com': [0.0, 0.0, 0.85]
}

risk_assessment = recovery_ctrl.assess_imminent_fall(robot_state)
recovery_action = recovery_ctrl.execute_recovery_strategy(risk_assessment, robot_state)

print(f"Risk assessment: {risk_assessment['fall_risk_level']}")
print(f"Recovery action: {recovery_action['action']}")
if 'step' in recovery_action['commands']:
    print(f"Recommended step location: {recovery_action['commands']['step']}")
```

## Hands-on Exercise

1. Implement a balance controller that can maintain stability when external forces are applied to the robot.

2. Design a walking trajectory generator that can adapt to uneven terrain.

3. Using the educational AI agents, explore how different control strategies (ZMP-based vs. whole-body control vs. learning-based) affect the stability and performance of the humanoid robot.

The next section will explore how these control systems are implemented in real hardware and the challenges of the sim-to-real transfer.