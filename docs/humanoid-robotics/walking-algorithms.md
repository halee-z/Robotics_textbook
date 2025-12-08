---
sidebar_position: 4
---

# Humanoid Robot Walking Algorithms and Gait Generation

## Overview

Walking is one of the most fundamental and challenging capabilities for humanoid robots. The complexity arises from the need to maintain dynamic balance while transferring weight from one foot to another, all while navigating potentially complex environments. This section explores the mathematical foundations, implementation approaches, and practical considerations for humanoid walking algorithms.

## Fundamentals of Bipedal Locomotion

### Key Concepts

Humanoid walking involves several fundamental concepts that distinguish it from other forms of locomotion:

1. **Dynamic Balance**: Unlike wheeled or static robots, humanoid robots must actively maintain balance during walking.
2. **Zero Moment Point (ZMP)**: The point where the moment of the ground reaction force is zero, crucial for stability.
3. **Capture Point**: The location where a robot should step to stop its current CoM motion.
4. **Support Polygon**: The area defined by the robot's points of contact with the ground.

### Phases of Walking

Humanoid walking typically consists of several distinct phases:
- **Double Support Phase**: Both feet are on the ground
- **Single Support Phase**: Only one foot is on the ground
- **Swing Phase**: The free foot moves forward
- **Preparation Phase**: Adjusting stance before stepping

```python
import numpy as np
from math import sin, cos, pi, sqrt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class WalkingPhase:
    DOUBLE_SUPPORT = 0
    LEFT_STANCE = 1
    RIGHT_STANCE = 2
    LEFT_SWING = 3
    RIGHT_SWING = 4

class WalkingPatternGenerator:
    def __init__(self, com_height=0.85, step_length=0.3, step_duration=1.0):
        self.com_height = com_height  # Center of mass height (m)
        self.step_length = step_length  # Forward step length (m)
        self.step_duration = step_duration  # Total time per step (s)
        self.gravity = 9.81  # Gravity constant
        self.omega = sqrt(self.gravity / self.com_height)  # Natural frequency parameter
        
        # Gait timing parameters
        self.ds_duration = 0.1 * self.step_duration  # Double support duration (10% of step time)
        self.ss_duration = (self.step_duration - self.ds_duration) / 2  # Single support duration per foot
        
        # Foot placement parameters
        self.step_width = 0.2  # Lateral step width (m)
        self.step_height = 0.1  # Maximum swing foot height (m)
        
    def generate_walking_pattern(self, n_steps=10, walking_direction="forward"):
        """
        Generate complete walking pattern for n steps
        
        Args:
            n_steps: Number of walking steps to generate
            walking_direction: "forward", "backward", "sideways", "turn"
            
        Returns:
            Dictionary containing CoM and foot trajectories
        """
        total_time = n_steps * self.step_duration
        dt = 0.01  # 100 Hz control rate
        time_points = np.arange(0, total_time, dt)
        
        # Initialize trajectory arrays
        com_x = np.zeros_like(time_points)
        com_y = np.zeros_like(time_points)
        com_z = np.full_like(time_points, self.com_height)  # Approximately constant CoM height
        
        # Foot trajectories
        left_foot_x = np.zeros_like(time_points)
        left_foot_y = np.zeros_like(time_points)
        left_foot_z = np.zeros_like(time_points)
        
        right_foot_x = np.zeros_like(time_points)
        right_foot_y = np.zeros_like(time_points)
        right_foot_z = np.zeros_like(time_points)
        
        # ZMP trajectory (for balance monitoring)
        zmp_x = np.zeros_like(time_points)
        zmp_y = np.zeros_like(time_points)
        
        # Generate pattern for each step
        for step_idx in range(n_steps):
            step_start_time = step_idx * self.step_duration
            step_end_time = (step_idx + 1) * self.step_duration
            
            # Determine step parameters based on walking direction
            if walking_direction == "forward":
                target_x = step_idx * self.step_length
                target_y = 0.0
            elif walking_direction == "backward":
                target_x = -step_idx * self.step_length
                target_y = 0.0
            elif walking_direction == "sideways":
                target_x = 0.0
                target_y = step_idx * self.step_width
            else:  # forward by default
                target_x = step_idx * self.step_length
                target_y = 0.0
            
            # Generate CoM trajectory for this step
            step_times_mask = (time_points >= step_start_time) & (time_points < step_end_time)
            step_relative_time = time_points[step_times_mask] - step_start_time
            
            # Generate CoM trajectory using 5th order polynomial (smooth transitions)
            com_x[step_times_mask] = self.generate_com_trajectory(
                step_relative_time, 
                step_phase=step_idx % 2,  # Alternate between left and right support
                step_length=self.step_length
            )
            
            # Generate foot trajectories for this step
            left_foot_x[step_times_mask], left_foot_y[step_times_mask], left_foot_z[step_times_mask] = \
                self.generate_foot_trajectory(
                    time=step_relative_time,
                    foot="left",
                    current_step=step_idx,
                    target_position=[target_x, target_y]
                )
                
            right_foot_x[step_times_mask], right_foot_y[step_times_mask], right_foot_z[step_times_mask] = \
                self.generate_foot_trajectory(
                    time=step_relative_time,
                    foot="right",
                    current_step=step_idx,
                    target_position=[target_x, target_y]
                )
        
        # Calculate ZMP from CoM trajectory
        com_vel_x = np.gradient(com_x, dt)
        com_vel_y = np.gradient(com_y, dt)
        com_acc_x = np.gradient(com_vel_x, dt)
        com_acc_y = np.gradient(com_vel_y, dt)
        
        zmp_x = com_x - (self.com_height / self.gravity) * com_acc_x
        zmp_y = com_y - (self.com_height / self.gravity) * com_acc_y
        
        return {
            'time': time_points,
            'com_trajectory': {
                'x': com_x,
                'y': com_y,
                'z': com_z
            },
            'left_foot_trajectory': {
                'x': left_foot_x,
                'y': left_foot_y,
                'z': left_foot_z
            },
            'right_foot_trajectory': {
                'x': right_foot_x,
                'y': right_foot_y,
                'z': right_foot_z
            },
            'zmp_trajectory': {
                'x': zmp_x,
                'y': zmp_y
            },
            'step_times': [i * self.step_duration for i in range(n_steps)],
            'support_polygon': self.calculate_support_polygon(n_steps)
        }
    
    def generate_com_trajectory(self, time, step_phase, step_length):
        """
        Generate CoM trajectory for a single step
        
        Args:
            time: Time array for the current step phase
            step_phase: 0 for left support, 1 for right support
            step_length: Forward distance of this step
            
        Returns:
            Array of CoM x-positions during this step
        """
        # Use a smooth function for CoM trajectory
        # The CoM typically moves forward smoothly with slight lateral movement
        # to maintain balance over the supporting foot
        
        # Normalize time within the step (0 to 1)
        normalized_time = time / self.step_duration
        
        # Generate CoM trajectory using 5th order polynomial for smooth acceleration/deceleration
        # This creates a smooth forward movement from one support to the next
        
        # Forward progression
        forward_progress = self.fifth_order_polynomial(normalized_time, 0, step_length * 0.8)
        
        # Lateral movement to maintain balance over support foot
        if step_phase == 0:  # Left foot support
            lateral_shift = self.fifth_order_polynomial(normalized_time, -self.step_width, 0.0)
        else:  # Right foot support
            lateral_shift = self.fifth_order_polynomial(normalized_time, self.step_width, 0.0)
        
        return forward_progress + lateral_shift
    
    def fifth_order_polynomial(self, t, start_val, end_val):
        """
        Generate a 5th order polynomial trajectory
        Ensures zero velocity and acceleration at start and end
        """
        # 5th order polynomial: s(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
        # Conditions: s(0)=start_val, s(1)=end_val, s'(0)=0, s'(1)=0, s''(0)=0, s''(1)=0
        progress = 6 * t**5 - 15 * t**4 + 10 * t**3
        return start_val + progress * (end_val - start_val)
    
    def generate_foot_trajectory(self, time, foot, current_step, target_position):
        """
        Generate trajectory for a foot during a step
        
        Args:
            time: Time array within step
            foot: "left" or "right"
            current_step: Current step number
            target_position: [x, y] where foot should end up
            
        Returns:
            Arrays of x, y, z positions
        """
        # Determine if this foot is in swing phase or stance phase
        step_number = current_step
        
        # For alternating steps, left foot is in swing during even-numbered steps for left support
        # This implementation models alternating walk
        if foot == "left":
            # Left foot: support during step 0, 2, 4, ... and swing during 1, 3, 5, ...
            is_swing = step_number % 2 == 1  # Left foot swings during odd-numbered steps
        else:  # right foot
            # Right foot: support during step 1, 3, 5, ... and swing during 0, 2, 4, ...
            is_swing = step_number % 2 == 0  # Right foot swings during even-numbered steps
        
        if not is_swing:
            # Foot is in stance - minimal movement
            x_pos = np.full_like(time, target_position[0])
            y_pos = np.full_like(time, target_position[1])
            z_pos = np.zeros_like(time)  # Foot on ground
            
            # Add slight movement to simulate stance adjustments
            # Small oscillation to model natural weight shifting
            if foot == "left":
                y_offset = -self.step_width / 2
            else:
                y_offset = self.step_width / 2
                
            # Add small periodic adjustment
            adjustment = 0.01 * np.sin(4 * pi * time / self.step_duration)
            y_pos += y_offset + adjustment
        
        else:
            # Foot is in swing phase - follow swinging trajectory
            x_pos, y_pos, z_pos = self.generate_swing_trajectory(time, foot, target_position)
        
        return x_pos, y_pos, z_pos
    
    def generate_swing_trajectory(self, time, foot, target_position):
        """
        Generate swing phase trajectory for one foot
        """
        # Calculate swing phase time normalization
        t_normalized = np.clip(time / self.step_duration, 0, 1)
        
        # Define start and end positions
        # Foot starts and ends at ground level (z=0), reaches max height in middle
        if foot == "left":
            start_x = target_position[0] - self.step_length  # Previous step position
            start_y = -self.step_width / 2
        else:  # right foot
            start_x = target_position[0] - self.step_length
            start_y = self.step_width / 2
        
        end_x = target_position[0] + self.step_length  # Next step position
        end_y = -start_y if foot == "left" else self.step_width / 2  # Opposite foot position
        
        # Horizontal trajectory (smooth transition)
        x_trajectory = self.fifth_order_polynomial(t_normalized, start_x, end_x)
        
        # Lateral trajectory 
        y_trajectory = self.fifth_order_polynomial(t_normalized, start_y, end_y)
        
        # Vertical trajectory (parabolic/elliptical lift)
        # Reach max height in middle of swing
        z_trajectory = np.zeros_like(time)
        for i, t in enumerate(t_normalized):
            # Parabolic curve: goes up then down
            if t <= 0.5:
                # Rising phase
                z_trajectory[i] = self.step_height * (4 * t**2)  # Parabolic rise
            else:
                # Falling phase
                z_trajectory[i] = self.step_height * (4 * (1-t)**2)  # Parabolic fall
        
        return x_trajectory, y_trajectory, z_trajectory
    
    def calculate_support_polygon(self, n_steps):
        """
        Calculate the support polygon for the walking pattern
        """
        polygons = []
        for step in range(n_steps):
            # Each support polygon is defined by the stance foot(s)
            # This is a simplified representation
            if step % 2 == 0:
                # Left foot is stance foot
                polygon = {
                    'step': step,
                    'stance_foot': 'left',
                    'vertices': [
                        [-0.1, -0.07],  # Approximate foot boundary
                        [0.1, -0.07],
                        [0.1, 0.07],
                        [-0.1, 0.07]
                    ]
                }
            else:
                # Right foot is stance foot
                polygon = {
                    'step': step,
                    'stance_foot': 'right', 
                    'vertices': [
                        [-0.1, -0.07 + self.step_width],  # Offset for right foot
                        [0.1, -0.07 + self.step_width],
                        [0.1, 0.07 + self.step_width],
                        [-0.1, 0.07 + self.step_width]
                    ]
                }
            polygons.append(polygon)
        
        return polygons

# Example usage
walker = WalkingPatternGenerator(com_height=0.85, step_length=0.3, step_duration=1.0)
walking_pattern = walker.generate_walking_pattern(n_steps=6, walking_direction="forward")

print(f"Generated walking pattern for {len(walking_pattern['step_times'])} steps")
print(f"Trajectory duration: {walking_pattern['time'][-1]:.2f} seconds")
print(f"Final CoM position: ({walking_pattern['com_trajectory']['x'][-1]:.2f}, {walking_pattern['com_trajectory']['y'][-1]:.2f})")
```

## Linear Inverted Pendulum Model (LIPM) for Walking

The Linear Inverted Pendulum Model is a foundational approach to understanding and generating stable walking patterns:

```python
class LinearInvertedPendulumWalking:
    def __init__(self, com_height=0.85, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)
    
    def com_dynamics(self, t, state):
        """
        Define the dynamics of the inverted pendulum
        State = [x_com, x_vel, y_com, y_vel]
        """
        x, x_dot, y, y_dot = state
        x_ddot = self.omega**2 * x
        y_ddot = self.omega**2 * y
        
        return [x_dot, x_ddot, y_dot, y_ddot]
    
    def solve_for_zmp(self, t, x_com, y_com):
        """
        Calculate ZMP position for given CoM position
        For LIPM: ZMP_x = CoM_x - (h/g) * CoM_x_dd
        """
        # Since we're solving the differential equation forward,
        # we can derive acceleration from position
        # In a real system, we would solve this differently
        pass
    
    def generate_lipm_trajectory(self, initial_state, final_state, duration):
        """
        Generate CoM trajectory using LIPM with ZMP planning
        """
        # For a simple example, use the LIPM solution
        # x(t) = A * e^(ωt) + B * e^(-ωt)
        x0, vx0, y0, vy0 = initial_state
        
        # Solve for A and B constants using initial conditions
        # x(0) = A + B = x0
        # x'(0) = ω(A - B) = vx0 => A - B = vx0/ω
        A_x = (x0 + vx0/self.omega) / 2
        B_x = (x0 - vx0/self.omega) / 2
        
        A_y = (y0 + vy0/self.omega) / 2
        B_y = (y0 - vy0/self.omega) / 2
        
        # Generate time array
        t_points = np.linspace(0, duration, int(duration * 100))  # 100Hz sampling
        
        # Calculate trajectories
        com_x = A_x * np.exp(self.omega * t_points) + B_x * np.exp(-self.omega * t_points)
        com_x_dot = self.omega * (A_x * np.exp(self.omega * t_points) - B_x * np.exp(-self.omega * t_points))
        com_x_ddot = self.omega**2 * (A_x * np.exp(self.omega * t_points) + B_x * np.exp(-self.omega * t_points))
        
        com_y = A_y * np.exp(self.omega * t_points) + B_y * np.exp(-self.omega * t_points)
        com_y_dot = self.omega * (A_y * np.exp(self.omega * t_points) - B_y * np.exp(-self.omega * t_points))
        com_y_ddot = self.omega**2 * (A_y * np.exp(self.omega * t_points) + B_y * np.exp(-self.omega * t_points))
        
        # Calculate ZMP trajectory
        zmp_x = com_x - (self.com_height / self.gravity) * com_x_ddot
        zmp_y = com_y - (self.com_height / self.gravity) * com_y_ddot
        
        return {
            'time': t_points,
            'com_trajectory': {
                'x': com_x,
                'y': com_y,
                'x_velocity': com_x_dot,
                'y_velocity': com_y_dot,
                'x_acceleration': com_x_ddot,
                'y_acceleration': com_y_ddot
            },
            'zmp_trajectory': {
                'x': zmp_x,
                'y': zmp_y
            }
        }

# Example LIPM usage
lipm_walker = LinearInvertedPendulumWalking(com_height=0.85)

# Start with CoM slightly off-center to see how it evolves
initial_state = [0.05, 0.1, 0.02, 0.05]  # [x, vx, y, vy]
final_state = [0.3, 0.05, 0.0, 0.0]  # Target state after movement

lipm_trajectory = lipm_walker.generate_lipm_trajectory(initial_state, final_state, 2.0)

print(f"LIPM trajectory calculated for 2 seconds")
print(f"Initial CoM: ({initial_state[0]:.3f}, {initial_state[2]:.3f})")
print(f"Final CoM: ({lipm_trajectory['com_trajectory']['x'][-1]:.3f}, {lipm_trajectory['com_trajectory']['y'][-1]:.3f})")
```

## Preview Control for Walking

Preview control uses planned future ZMP references to generate stable CoM trajectories:

```python
from scipy.linalg import solve_discrete_are, inv
from scipy import signal

class PreviewControllerWalking:
    def __init__(self, com_height=0.85, dt=0.01, preview_window=2.0):
        """
        Initialize preview controller for walking
        
        Args:
            com_height: Height of the center of mass
            dt: Control time step
            preview_window: How far ahead to look (seconds)
        """
        self.com_height = com_height
        self.gravity = 9.81
        self.dt = dt
        self.omega = np.sqrt(self.gravity / com_height)
        
        # Number of steps to look ahead
        self.preview_steps = int(preview_window / dt)
        
        # Discrete state-space system for LIPM
        # State: x = [CoM_x, CoM_x_dot]
        # Dynamics: x_{k+1} = A*x_k + B*zmp_k
        self.A = np.array([
            [1 + self.omega**2 * dt**2 / 2, dt],
            [self.omega**2 * dt, 1.0]
        ])
        
        self.B = np.array([self.omega**2 * dt**2 / 2, self.omega**2 * dt])
        
        # Output matrix to get ZMP from state
        # For our system: zmp = x - (h/g)*acc = x - (h/g)*(ω²(x - zmp)) 
        # Which gives: zmp = x - (h/g)*(ω²(x - zmp)), solving: zmp = x - (hω²/g)/(1+hω²/g)*x
        # Actually, let's use the LIPM relation directly
        # zmp = x - (h/g)*acc where acc = omega^2 * (x - zmp)
        # So zmp = x - (h/g)*omega^2*(x - zmp)
        # zmp + (h/g)*omega^2*zmp = x - (h/g)*omega^2*x
        # zmp = (x - (h/g)*omega^2*x) / (1 + (h/g)*omega^2)
        # zmp = x * (1 - (h/g)*omega^2) / (1 + (h/g)*omega^2)
        # Since h*omega^2/g = h*(g/h)/g = 1, this simplifies incorrectly
        # Actually zmp = x - (h/g)*acc, and acc = d^2x/dt^2
        # For numerical stability, we'll approximate output as x - offset
        
        # For simplicity, let's say output is just zmp (the control input)
        # Since the dynamics are defined by ZMP, ZMP is the control input
        
        # Design LQR controller
        Q = np.array([[100, 0], [0, 1]])  # State cost (penalize CoM deviation more)
        R = 0.1  # Control cost (ZMP deviation)
        
        # Solve discrete-time algebraic Riccati equation
        try:
            P = solve_discrete_are(self.A.T, self.B.T, Q, R)
            
            # Calculate optimal gain
            K = inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            self.feedback_gain = K
            
            # Precompute preview gains
            self.preview_gains = self.calculate_preview_gains(Q, R, P)
            
        except np.linalg.LinAlgError:
            # Fallback if Riccati equation doesn't converge
            print("Warning: LQR design failed, using simple gain")
            self.feedback_gain = np.array([1.0, 1.0])
            self.preview_gains = np.zeros(self.preview_steps)
    
    def calculate_preview_gains(self, Q, R, P):
        """
        Calculate preview gains for future ZMP references
        """
        # This is a simplified calculation
        # In practice, this involves solving for feedforward gains
        gains = np.zeros(self.preview_steps)
        
        # For each preview step k, calculate the gain that maps future reference to current control
        for k in range(self.preview_steps):
            # The preview gain at step k depends on A^k, Q, R, and system matrices
            A_pow = np.linalg.matrix_power(self.A, k)
            
            # Simplified preview gain calculation
            # This is actually a complex calculation in full preview control theory
            gains[k] = np.exp(-0.1 * k)  # Exponential decay weight for preview
        
        return gains
    
    def compute_control(self, current_state, zmp_reference_sequence):
        """
        Compute control input using preview control law
        
        Args:
            current_state: Current state [x_com, x_com_dot]
            zmp_reference_sequence: Sequence of future ZMP references
            
        Returns:
            Control input (desired ZMP for next timestep)
        """
        # Feedback term
        feedback_control = -self.feedback_gain @ current_state
        
        # Preview term (feedforward)
        preview_control = 0.0
        
        # Apply preview gains to future references
        n_refs = min(len(zmp_reference_sequence), self.preview_steps)
        for k in range(n_refs):
            weight = self.preview_gains[k]
            if k < len(zmp_reference_sequence):
                preview_control += weight * zmp_reference_sequence[k]
        
        total_control = feedback_control + preview_control
        
        return total_control[0] if hasattr(total_control, '__len__') else total_control
    
    def generate_walking_pattern_with_preview(self, step_length=0.3, step_duration=1.0, steps=6):
        """
        Generate walking pattern using preview control approach
        """
        total_time = steps * step_duration
        n_points = int(total_time / self.dt)
        time_points = np.linspace(0, total_time, n_points)
        
        # Initialize state and trajectory arrays
        state = np.array([0.0, 0.0])  # [com_pos, com_vel]
        com_positions = np.zeros(n_points)
        com_velocities = np.zeros(n_points)
        zmp_commands = np.zeros(n_points)
        
        # Define ZMP reference pattern (alternating support)
        zmp_reference = np.zeros(n_points)
        for i, t in enumerate(time_points):
            step_no = int(t / step_duration)
            if step_no % 2 == 0:  # Left foot support
                zmp_reference[i] = -0.05  # Slightly left of center
            else:  # Right foot support
                zmp_reference[i] = 0.05   # Slightly right of center
            
            # Add forward progression
            zmp_reference[i] += (step_no * step_length * 0.8)  # Gradual forward movement
        
        # Simulate walking using preview control
        for i in range(n_points - 1):  # -1 to avoid index out of bounds
            # Get preview window of future ZMP references
            start_idx = i
            end_idx = min(i + self.preview_steps, n_points)
            zmp_preview = zmp_reference[start_idx:end_idx]
            
            # Compute control using preview controller
            desired_zmp = self.compute_control(state, zmp_preview)
            
            # Update state using system dynamics
            # x_next = A*x + B*u (where u is the desired ZMP)
            state = self.A @ state + self.B * desired_zmp
            
            # Store results
            com_positions[i+1] = state[0]
            com_velocities[i+1] = state[1]
            zmp_commands[i] = desired_zmp
        
        return {
            'time': time_points,
            'com_position': com_positions,
            'com_velocity': com_velocities,
            'zmp_command': zmp_commands,
            'zmp_reference': zmp_reference,
            'step_times': [i * step_duration for i in range(steps)]
        }

# Example usage of preview controller
preview_walker = PreviewControllerWalking(com_height=0.85, dt=0.01, preview_window=1.0)
preview_walking = preview_walker.generate_walking_pattern_with_preview(
    step_length=0.3,
    step_duration=1.0,
    steps=6
)

print(f"Preview control walking generated for {len(preview_walking['step_times'])} steps")
print(f"Final CoM position: {preview_walking['com_position'][-1]:.3f} m")
print(f"Final CoM velocity: {preview_walking['com_velocity'][-1]:.3f} m/s")
```

## Capture Point-Based Walking

The capture point is a critical concept for dynamic balance recovery:

```python
class CapturePointWalking:
    def __init__(self, com_height=0.85, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)
    
    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point where robot should step to halt CoM motion
        
        Args:
            com_pos: Current CoM position [x, y]
            com_vel: Current CoM velocity [vx, vy]
            
        Returns:
            Capture point position [x, y]
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega
        
        return np.array([cp_x, cp_y])
    
    def generate_stable_step_locations(self, com_trajectory, step_timing):
        """
        Generate step locations based on capture point for stable walking
        
        Args:
            com_trajectory: Dictionary with CoM positions and velocities
            step_timing: Times at which to place feet
            
        Returns:
            Array of step locations
        """
        step_locations = []
        
        for step_time in step_timing:
            # Find closest CoM state to the step time
            time_diffs = np.abs(com_trajectory['time'] - step_time)
            closest_idx = np.argmin(time_diffs)
            
            # Get CoM state at that time
            com_pos = np.array([
                com_trajectory['x'][closest_idx], 
                com_trajectory['y'][closest_idx]
            ])
            com_vel = np.array([
                com_trajectory['vx'][closest_idx] if 'vx' in com_trajectory else 0,
                com_trajectory['vy'][closest_idx] if 'vy' in com_trajectory else 0
            ])
            
            # Calculate capture point
            cp = self.calculate_capture_point(com_pos, com_vel)
            
            # Use capture point as target step location with safety margin
            # Add slight offset for double support and step width
            step_x = cp[0]
            step_y = cp[1] + (self.step_width / 2) * ((len(step_locations) + 1) % 2 * 2 - 1)  # Alternate sides
            
            step_locations.append([step_x, step_y, 0.0])  # z=0 for foot on ground
        
        return np.array(step_locations)
    
    def implement_capture_point_control(self, current_com_pos, current_com_vel, support_feet_pos):
        """
        Implement capture point-based balance control
        
        Args:
            current_com_pos: Current CoM position
            current_com_vel: Current CoM velocity
            support_feet_pos: Positions of support feet
            
        Returns:
            Step location recommendation
        """
        # Calculate current capture point
        current_cp = self.calculate_capture_point(current_com_pos, current_com_vel)
        
        # Determine if capture point is outside support polygon
        support_polygon_valid = self.is_capture_point_stable(current_cp, support_feet_pos)
        
        if not support_polygon_valid:
            # Need to take a step to move support polygon under capture point
            step_location = self.plan_recovery_step(current_cp, support_feet_pos)
            return {
                'action': 'take_step',
                'step_location': step_location,
                'urgency': 'high'
            }
        else:
            # Within stable region, continue normal walking
            return {
                'action': 'continue_normal',
                'step_location': None,
                'urgency': 'low'
            }
    
    def is_capture_point_stable(self, capture_point, support_feet_pos):
        """
        Check if capture point is within support polygon
        """
        if len(support_feet_pos) == 0:
            return False  # No support
        
        if len(support_feet_pos) == 1:
            # Single support - check if CP is near the foot
            foot_pos = support_feet_pos[0][:2]  # Get x,y only
            distance_to_foot = np.linalg.norm(capture_point - foot_pos)
            return distance_to_foot <= 0.1  # Within 10cm of foot (adjust as needed)
        else:
            # Double support - create convex hull and check if CP is inside
            from scipy.spatial import ConvexHull
            # Get x,y coordinates of support feet
            support_xy = np.array([[foot[0], foot[1]] for foot in support_feet_pos]) 
            
            try:
                hull = ConvexHull(support_xy)
                # Check if capture_point is inside the hull
                return self.point_in_convex_hull(capture_point, hull, support_xy)
            except:
                # If hull computation fails, fall back to simpler check
                return self.simple_polygon_check(capture_point, support_xy)
    
    def point_in_convex_hull(self, point, hull, points):
        """
        Check if a point is inside a 2D convex hull
        """
        # For each edge of the convex hull, check if point is on the correct side
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            
            # Calculate cross product to see which side of line point is on
            cross_product = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            
            # For a counter-clockwise hull, all cross products should be positive
            # for the point to be inside
            if cross_product < 0:
                return False
        
        return True
    
    def simple_polygon_check(self, point, vertices):
        """
        Simple check if point is in polygon using ray casting
        """
        x, y = point
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def plan_recovery_step(self, capture_point, current_support_feet):
        """
        Plan step location to move capture point to stability
        """
        # For recovery, step close to the capture point
        # But also consider gait pattern and avoid placing foot too close to current stance foot
        recovery_location = capture_point.copy()
        
        # Make sure it's not too close to any existing feet
        for foot_pos in current_support_feet:
            foot_xy = np.array([foot_pos[0], foot_pos[1]])
            distance = np.linalg.norm(recovery_location - foot_xy)
            
            if distance < 0.3:  # Too close (less than 30cm)
                # Adjust direction away from the close foot
                direction = (recovery_location - foot_xy) / distance
                recovery_location = foot_xy + direction * 0.3
        
        return [recovery_location[0], recovery_location[1], 0.0]

# Example usage
cp_walker = CapturePointWalking(com_height=0.85)

# Simulate an unstable condition
com_position = np.array([0.1, 0.05])  # CoM slightly off position
com_velocity = np.array([0.3, 0.1])   # Moving with some velocity
support_feet = np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]])  # Feet in normal stance

control_action = cp_walker.implement_capture_point_control(com_position, com_velocity, support_feet)

print(f"Capture point control action: {control_action['action']}")
if control_action['step_location'] is not None:
    print(f"Recommended step location: {control_action['step_location']}")
```

## Advanced Walking Patterns

### Dynamic Walking with Momentum

```python
class MomentumBasedWalking:
    def __init__(self, robot_mass=50, com_height=0.85):
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = sqrt(self.gravity / com_height)
    
    def calculate_com_momentum(self, com_pos, com_vel):
        """
        Calculate momentum of the center of mass
        """
        linear_momentum = self.robot_mass * np.array(com_vel)
        
        return {
            'linear_momentum': linear_momentum,
            'momentum_magnitude': np.linalg.norm(linear_momentum)
        }
    
    def generate_momentum_conserving_steps(self, desired_velocity, current_state):
        """
        Generate stepping pattern that conserves momentum where possible
        """
        # Desired momentum
        desired_momentum = self.robot_mass * desired_velocity
        
        # Current momentum
        current_momentum = self.robot_mass * current_state['com_velocity']
        
        # Momentum difference to be corrected through stepping
        momentum_correction = desired_momentum - current_momentum
        
        # Plan steps to achieve desired momentum state
        step_plan = self.plan_momentum_steps(momentum_correction, current_state)
        
        return step_plan
    
    def plan_momentum_steps(self, momentum_correction, current_state):
        """
        Plan steps to correct momentum
        """
        # Calculate required impulse (change in momentum)
        required_impulse = momentum_correction
        
        # Determine how to distribute this impulse over upcoming steps
        # This would involve more complex planning in a real implementation
        
        # For now, return a simple plan based on capture point and momentum
        cp_calc = CapturePointWalking(com_height=self.com_height)
        current_cp = cp_calc.calculate_capture_point(
            current_state['com_position'][:2], 
            current_state['com_velocity'][:2]
        )
        
        # Step location should account for both balance (capture point) and momentum
        step_x = current_cp[0] + 0.3 * (required_impulse[0] / self.robot_mass)  # Factor for momentum correction
        step_y = current_cp[1] + 0.3 * (required_impulse[1] / self.robot_mass)
        
        return {
            'next_step_location': [step_x, step_y, 0.0],
            'required_impulse': required_impulse,
            'planned_velocity_change': required_impulse / self.robot_mass
        }
    
    def implement_momentum_based_control(self, state_reference, current_state):
        """
        Main control function using momentum principles
        """
        # Calculate current momentum
        current_momentum = self.calculate_com_momentum(
            current_state['com_position'], 
            current_state['com_velocity']
        )
        
        # Compare with desired momentum from reference
        desired_momentum = self.robot_mass * state_reference['desired_velocity']
        momentum_error = desired_momentum - current_momentum['linear_momentum']
        
        # Generate appropriate stepping and control commands
        control_output = {
            'momentum_error': momentum_error,
            'correction_strategy': 'momentum_conserving',
            'step_plan': self.plan_momentum_steps(momentum_error, current_state)
        }
        
        return control_output

# Example usage
mom_walker = MomentumBasedWalking(robot_mass=60, com_height=0.85)

# Current walking state
current_state = {
    'com_position': np.array([0.5, 0.0, 0.85]),
    'com_velocity': np.array([0.4, 0.05, 0.0])
}

# Desired state
state_reference = {
    'desired_velocity': np.array([0.5, 0.0, 0.0])  # Want to accelerate forward
}

momentum_control = mom_walker.implement_momentum_based_control(state_reference, current_state)
print(f"Momentum-based control planned step to: {momentum_control['step_plan']['next_step_location']}")
```

## Walking Pattern Visualization

```python
def visualize_walking_pattern(pattern_data):
    """
    Visualize the generated walking pattern
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    time = pattern_data['time']
    
    # Plot 1: CoM trajectory
    axes[0, 0].plot(pattern_data['com_trajectory']['x'], pattern_data['com_trajectory']['y'], 'b-', linewidth=2, label='CoM Trajectory')
    axes[0, 0].plot(pattern_data['left_foot_trajectory']['x'][::10], pattern_data['left_foot_trajectory']['y'][::10], 'ro', markersize=4, label='Left Foot Steps')
    axes[0, 0].plot(pattern_data['right_foot_trajectory']['x'][::10], pattern_data['right_foot_trajectory']['y'][::10], 'go', markersize=4, label='Right Foot Steps')
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('CoM and Foot Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: CoM height (should be mostly constant)
    axes[0, 1].plot(time, pattern_data['com_trajectory']['z'], 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Height (m)')
    axes[0, 1].set_title('CoM Height Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ZMP trajectory and support polygon
    axes[0, 2].plot(pattern_data['zmp_trajectory']['x'], pattern_data['zmp_trajectory']['y'], 'r-', linewidth=2, label='ZMP Trajectory')
    
    # Plot support polygon boundaries for each step
    for poly in pattern_data['support_polygon']:
        vertices = np.array(poly['vertices'])
        vertices_closed = np.vstack([vertices, vertices[0]])  # Close the polygon
        axes[0, 2].plot(vertices_closed[:, 0], vertices_closed[:, 1], '--', alpha=0.7, label=f'Step {poly["step"]} Support' if poly['step'] <= 2 else "")
    
    axes[0, 2].set_xlabel('X Position (m)')
    axes[0, 2].set_ylabel('Y Position (m)')
    axes[0, 2].set_title('ZMP Trajectory vs Support Polygon')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Foot heights over time
    axes[1, 0].plot(time, pattern_data['left_foot_trajectory']['z'], label='Left Foot Height', linewidth=2)
    axes[1, 0].plot(time, pattern_data['right_foot_trajectory']['z'], label='Right Foot Height', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Height (m)')
    axes[1, 0].set_title('Foot Height Profiles Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: CoM velocity profiles
    com_vel_x = np.gradient(pattern_data['com_trajectory']['x'], time[1]-time[0])
    com_vel_y = np.gradient(pattern_data['com_trajectory']['y'], time[1]-time[0])
    axes[1, 1].plot(time, com_vel_x, label='CoM Velocity X', linewidth=2)
    axes[1, 1].plot(time, com_vel_y, label='CoM Velocity Y', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Velocity (m/s)')
    axes[1, 1].set_title('CoM Velocity Profiles')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Step timing and gait phases
    axes[1, 2].eventplot([pattern_data['step_times']], colors=['blue'], linelengths=0.5, label='Step Times')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Events')
    axes[1, 2].set_title('Step Timing Events')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Visualize the walking pattern
visualize_walking_pattern(walking_pattern)
```

## Walking Stability Metrics

```python
class WalkingStabilityAnalyzer:
    def __init__(self, robot_params):
        self.com_height = robot_params.get('com_height', 0.85)
        self.foot_size = robot_params.get('foot_size', [0.25, 0.1])  # [length, width]
        self.gravity = 9.81
        
    def calculate_stability_margins(self, pattern_data):
        """
        Calculate stability margins for the walking pattern
        """
        stability_analysis = {
            'time_to_unstable': [],
            'minimum_zmp_margin': [],
            'capture_point_deviation': [],
            'angular_momentum': []
        }
        
        com_x = pattern_data['com_trajectory']['x']
        com_y = pattern_data['com_trajectory']['y']
        zmp_x = pattern_data['zmp_trajectory']['x']
        zmp_y = pattern_data['zmp_trajectory']['y']
        
        # Calculate velocities and accelerations
        dt = pattern_data['time'][1] - pattern_data['time'][0]
        com_vel_x = np.gradient(com_x, dt)
        com_vel_y = np.gradient(com_y, dt)
        com_acc_x = np.gradient(com_vel_x, dt)
        com_acc_y = np.gradient(com_vel_y, dt)
        
        # Calculate instantaneous stability metrics
        for i in range(len(zmp_x)):
            # Calculate instantaneous capture point
            instant_cp_x = com_x[i] + com_vel_x[i] / sqrt(self.gravity / self.com_height)
            instant_cp_y = com_y[i] + com_vel_y[i] / sqrt(self.gravity / self.com_height)
            
            # Calculate distance from ZMP to support polygon edge
            # For simplicity, assume rectangular support polygon based on foot placement
            current_support_polygon = self.get_current_support_polygon(
                pattern_data, 
                pattern_data['time'][i]
            )
            
            zmp_pos = np.array([zmp_x[i], zmp_y[i]])
            cp_pos = np.array([instant_cp_x, instant_cp_y])
            
            # Calculate distance to polygon boundary
            zmp_margin = self.distance_to_polygon_boundary(zmp_pos, current_support_polygon)
            cp_deviation = self.distance_to_polygon_boundary(cp_pos, current_support_polygon)
            
            stability_analysis['minimum_zmp_margin'].append(zmp_margin)
            stability_analysis['capture_point_deviation'].append(cp_deviation)
            
            # Angular momentum (simplified)
            angular_mom = com_vel_x[i] * com_acc_y[i] - com_vel_y[i] * com_acc_x[i]
            stability_analysis['angular_momentum'].append(angular_mom)
        
        return stability_analysis
    
    def get_current_support_polygon(self, pattern_data, current_time):
        """
        Determine the current support polygon based on foot positions
        """
        # Find which feet are currently in contact with ground
        time_idx = min(int(current_time / 0.01), len(pattern_data['time']) - 1)
        
        left_z = pattern_data['left_foot_trajectory']['z'][time_idx]
        right_z = pattern_data['right_foot_trajectory']['z'][time_idx]
        
        supports = []
        if left_z < 0.01:  # Left foot is down (considering small threshold for contact)
            supports.append([pattern_data['left_foot_trajectory']['x'][time_idx], 
                           pattern_data['left_foot_trajectory']['y'][time_idx]])
        
        if right_z < 0.01:  # Right foot is down
            supports.append([pattern_data['right_foot_trajectory']['x'][time_idx], 
                           pattern_data['right_foot_trajectory']['y'][time_idx]])
        
        # Create support polygon from supporting feet
        if len(supports) == 0:
            return []  # No support
        elif len(supports) == 1:
            # Single support - create a small polygon around the foot
            foot_pos = supports[0]
            return [
                [foot_pos[0] - self.foot_size[0]/2, foot_pos[1] - self.foot_size[1]/2],
                [foot_pos[0] + self.foot_size[0]/2, foot_pos[1] - self.foot_size[1]/2],
                [foot_pos[0] + self.foot_size[0]/2, foot_pos[1] + self.foot_size[1]/2],
                [foot_pos[0] - self.foot_size[0]/2, foot_pos[1] + self.foot_size[1]/2]
            ]
        else:
            # Double support - create polygon spanning both feet
            # Simplified convex hull
            return [
                [supports[0][0] - self.foot_size[0]/4, supports[0][1] - self.foot_size[1]/2],
                [supports[1][0] - self.foot_size[0]/4, supports[1][1] - self.foot_size[1]/2],
                [supports[1][0] + self.foot_size[0]/4, supports[1][1] + self.foot_size[1]/2],
                [supports[0][0] + self.foot_size[0]/4, supports[0][1] + self.foot_size[1]/2]
            ]
    
    def distance_to_polygon_boundary(self, point, polygon):
        """
        Calculate minimum distance from a point to the boundary of a polygon
        """
        if len(polygon) < 2:
            return float('inf')
        
        min_dist = float('inf')
        point = np.array(point)
        
        # Calculate distance to each edge of the polygon
        for i in range(len(polygon)):
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[(i + 1) % len(polygon)])
            
            # Calculate distance from point to line segment p1-p2
            dist = self.point_to_line_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def point_to_line_segment_distance(self, point, line_start, line_end):
        """
        Calculate distance from a point to a line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # Length squared of the line segment
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            # Line segment is actually a point
            return np.linalg.norm(point - line_start)
        
        # Project point_vec onto line_vec to find closest point on the line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        
        # Calculate the closest point on the line segment
        projection = line_start + t * line_vec
        
        # Return the distance to the closest point
        return np.linalg.norm(point - projection)
    
    def evaluate_walking_stability(self, pattern_data):
        """
        Comprehensive stability evaluation of the walking pattern
        """
        stability_metrics = {
            'average_zmp_margin': 0,
            'minimum_zmp_margin': float('inf'),
            'maximum_zmp_margin': float('-inf'),
            'zmp_margin_std': 0,
            'stability_score': 0,  # 0-1 scale, 1 is perfectly stable
            'instability_events': 0,
            'time_in_stable_region': 0
        }
        
        # Calculate stability margins
        stability_analysis = self.calculate_stability_margins(pattern_data)
        
        margins = stability_analysis['minimum_zmp_margin']
        
        if len(margins) > 0:
            stability_metrics['average_zmp_margin'] = np.mean(margins)
            stability_metrics['minimum_zmp_margin'] = np.min(margins)
            stability_metrics['maximum_zmp_margin'] = np.max(margins)
            stability_metrics['zmp_margin_std'] = np.std(margins)
            
            # Calculate stability score (0-1 scale)
            # Penalize low margins and high variance
            avg_margin = stability_metrics['average_zmp_margin']
            std_margin = stability_metrics['zmp_margin_std']
            
            # Score based on average margin and consistency
            margin_score = np.tanh(avg_margin * 10)  # Higher margins are better, capped
            consistency_score = 1.0 / (1.0 + std_margin)  # Lower variance is better
            
            stability_metrics['stability_score'] = 0.7 * margin_score + 0.3 * consistency_score
            
            # Count instability events (negative margins)
            stability_metrics['instability_events'] = np.sum(np.array(margins) < 0)
            stability_metrics['time_in_stable_region'] = np.sum(np.array(margins) > 0.05)  # Stable if > 5cm margin
            stability_metrics['time_in_stable_region'] /= len(margins)  # As fraction of total time
        
        return stability_metrics

# Example analysis
analyzer = WalkingStabilityAnalyzer({'com_height': 0.85})
stability_eval = analyzer.evaluate_walking_stability(walking_pattern)

print("Walking Pattern Stability Analysis:")
print(f"  Average ZMP Margin: {stability_eval['average_zmp_margin']:.3f} m")
print(f"  Minimum ZMP Margin: {stability_eval['minimum_zmp_margin']:.3f} m")
print(f"  Maximum ZMP Margin: {stability_eval['maximum_zmp_margin']:.3f} m")
print(f"  Stability Score: {stability_eval['stability_score']:.3f} (1 = perfectly stable)")
print(f"  Instability Events: {stability_eval['instability_events']}")
print(f"  Time in Stable Region: {stability_eval['time_in_stable_region']*100:.1f}% of total time")
```

## Hands-on Exercise

1. Implement a walking controller that adjusts step timing and location based on external disturbances.

2. Design a gait adaptation algorithm that modifies walking parameters when walking on uneven terrain.

3. Using the educational AI agents, explore how different walking parameters (step length, duration, height) affect the stability and energy efficiency of humanoid locomotion.

The next section will explore how these walking algorithms are implemented in real hardware and the challenges of adapting the algorithms to physical constraints.