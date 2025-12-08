---
sidebar_position: 2
---

# Humanoid Robot Kinematics: Structure and Motion

## Overview

Humanoid robot kinematics is the study of motion in humanoid robots without considering the forces that cause the motion. Understanding kinematics is fundamental to controlling humanoid robots, enabling them to perform complex movements while maintaining balance and achieving task objectives. This section covers the mathematical foundations, implementation approaches, and practical considerations for humanoid robot kinematics.

## Kinematic Structure of Humanoid Robots

### Degrees of Freedom

Humanoid robots typically have many degrees of freedom (DOF) to achieve human-like movement:

- **Head/Neck**: 3-6 DOF (yaw, pitch, roll; optional translation)
- **Trunk**: 6-12 DOF (waist rotation, lateral bending, pitch; optional translation)
- **Each Arm**: 7-9 DOF (shoulder: 3 DOF, elbow: 1 DOF, wrist: 2-3 DOF, optional hand DOF)
- **Each Leg**: 6-7 DOF (hip: 3 DOF, knee: 1 DOF, ankle: 2-3 DOF)

This results in humanoid robots having between 26-40+ total DOF, making their control significantly more complex than simpler robots.

### Kinematic Chains

Humanoid robots have multiple kinematic chains that work together:

```
Base (Trunk/Body)
├── Left Leg Chain
│   ├── Hip (3 DOF: yaw, pitch, roll)
│   ├── Knee (1 DOF: pitch)
│   └── Ankle (2-3 DOF: pitch, roll, optional translation)
├── Right Leg Chain
│   ├── Hip (3 DOF)
│   ├── Knee (1 DOF)
│   └── Ankle (2-3 DOF)
├── Left Arm Chain
│   ├── Shoulder (3 DOF)
│   ├── Elbow (1 DOF)
│   ├── Wrist (2-3 DOF)
│   └── Hand (variable DOF)
├── Right Arm Chain
│   ├── Shoulder (3 DOF)
│   ├── Elbow (1 DOF)
│   ├── Wrist (2-3 DOF)
│   └── Hand (variable DOF)
└── Head/Neck Chain
    ├── Neck (2-3 DOF: pitch, yaw, optional roll)
    └── Eyes (optional DOF for gaze control)
```

## Forward Kinematics

Forward kinematics calculates the position and orientation of end-effectors (hands, feet) given joint angles.

### Mathematical Foundation

For each kinematic chain, forward kinematics can be expressed as:

```
T = T_base * A_1(θ_1) * A_2(θ_2) * ... * A_n(θ_n)
```

Where:
- `T` is the transformation matrix from base to end-effector
- `T_base` is the base transformation
- `A_i(θ_i)` is the transformation matrix for joint i as a function of angle θ_i

### Implementation Example

```python
import numpy as np
from math import sin, cos

class HumanoidFK:
    def __init__(self):
        # Link lengths for a simplified humanoid model
        self.l_upper_leg = 0.45  # meters
        self.l_lower_leg = 0.45  # meters
        self.l_upper_arm = 0.35  # meters
        self.l_lower_arm = 0.35  # meters
    
    def rotation_matrix(self, angle, axis):
        """Create rotation matrix for a given angle around an axis"""
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, cos(angle), -sin(angle)],
                [0, sin(angle), cos(angle)]
            ])
        elif axis == 'y':
            return np.array([
                [cos(angle), 0, sin(angle)],
                [0, 1, 0],
                [-sin(angle), 0, cos(angle)]
            ])
        elif axis == 'z':
            return np.array([
                [cos(angle), -sin(angle), 0],
                [sin(angle), cos(angle), 0],
                [0, 0, 1]
            ])
    
    def transform_matrix(self, angle, axis, translation):
        """Create transformation matrix combining rotation and translation"""
        R = self.rotation_matrix(angle, axis)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T
    
    def leg_forward_kinematics(self, hip_angles, knee_angle, ankle_angles, leg_length=None):
        """
        Calculate leg forward kinematics
        hip_angles: [hip_yaw, hip_pitch, hip_roll]
        knee_angle: knee flexion angle
        ankle_angles: [ankle_pitch, ankle_roll]
        """
        if leg_length is None:
            upper_len = self.l_upper_leg
            lower_len = self.l_lower_leg
        else:
            upper_len, lower_len = leg_length
        
        # Base transformation (relative to body)
        T_base = np.eye(4)  # Simplified as identity
        
        # Hip joint transformations
        T_hip_yaw = self.transform_matrix(hip_angles[0], 'z', [0, 0, 0])
        T_hip_pitch = self.transform_matrix(hip_angles[1], 'y', [0, 0, 0])
        T_hip_roll = self.transform_matrix(hip_angles[2], 'x', [0, 0, 0])
        
        # Upper leg (translation only)
        T_upper_leg = self.transform_matrix(0, 'z', [0, 0, -upper_len])
        
        # Knee joint
        T_knee = self.transform_matrix(knee_angle, 'y', [0, 0, 0])
        
        # Lower leg (translation only)
        T_lower_leg = self.transform_matrix(0, 'z', [0, 0, -lower_len])
        
        # Ankle angles
        T_ankle_pitch = self.transform_matrix(ankle_angles[0], 'y', [0, 0, 0])
        T_ankle_roll = self.transform_matrix(ankle_angles[1], 'x', [0, 0, 0])
        
        # Combine all transformations
        T_total = (T_base @ T_hip_yaw @ T_hip_pitch @ T_hip_roll @ 
                  T_upper_leg @ T_knee @ T_lower_leg @ 
                  T_ankle_pitch @ T_ankle_roll)
        
        return T_total
    
    def arm_forward_kinematics(self, shoulder_angles, elbow_angle, wrist_angles, arm_length=None):
        """
        Calculate arm forward kinematics
        shoulder_angles: [shoulder_yaw, shoulder_pitch, shoulder_roll]
        elbow_angle: elbow flexion angle
        wrist_angles: [wrist_yaw, wrist_pitch, wrist_roll]
        """
        if arm_length is None:
            upper_len = self.l_upper_arm
            lower_len = self.l_lower_arm
        else:
            upper_len, lower_len = arm_length
        
        # Base transformation (relative to body)
        T_base = np.eye(4)  # Simplified as identity
        
        # Shoulder joint transformations
        T_shoulder_yaw = self.transform_matrix(shoulder_angles[0], 'z', [0, 0, 0])
        T_shoulder_pitch = self.transform_matrix(shoulder_angles[1], 'y', [0, 0, 0])
        T_shoulder_roll = self.transform_matrix(shoulder_angles[2], 'x', [0, 0, 0])
        
        # Upper arm (translation only)
        T_upper_arm = self.transform_matrix(0, 'y', [0, 0, -upper_len])
        
        # Elbow joint
        T_elbow = self.transform_matrix(elbow_angle, 'y', [0, 0, 0])
        
        # Lower arm (translation only)
        T_lower_arm = self.transform_matrix(0, 'y', [0, 0, -lower_len])
        
        # Wrist angles
        T_wrist_yaw = self.transform_matrix(wrist_angles[0], 'z', [0, 0, 0])
        T_wrist_pitch = self.transform_matrix(wrist_angles[1], 'y', [0, 0, 0])
        T_wrist_roll = self.transform_matrix(wrist_angles[2], 'x', [0, 0, 0])
        
        # Combine all transformations
        T_total = (T_base @ T_shoulder_yaw @ T_shoulder_pitch @ T_shoulder_roll @ 
                  T_upper_arm @ T_elbow @ T_lower_arm @ 
                  T_wrist_yaw @ T_wrist_pitch @ T_wrist_roll)
        
        return T_total
```

### Whole-Body Forward Kinematics

```python
class WholeBodyFK:
    def __init__(self):
        self.left_arm_fk = HumanoidFK()
        self.right_arm_fk = HumanoidFK()
        self.left_leg_fk = HumanoidFK()
        self.right_leg_fk = HumanoidFK()
    
    def calculate_all_end_effectors(self, joint_angles):
        """
        Calculate positions of all end-effectors given joint angles
        joint_angles: dictionary with joint names and angles
        """
        results = {}
        
        # Calculate left arm end-effector (left hand)
        left_shoulder = [
            joint_angles.get('left_shoulder_yaw', 0),
            joint_angles.get('left_shoulder_pitch', 0), 
            joint_angles.get('left_shoulder_roll', 0)
        ]
        left_elbow = joint_angles.get('left_elbow', 0)
        left_wrist = [
            joint_angles.get('left_wrist_yaw', 0),
            joint_angles.get('left_wrist_pitch', 0),
            joint_angles.get('left_wrist_roll', 0)
        ]
        
        T_left_hand = self.left_arm_fk.arm_forward_kinematics(
            left_shoulder, left_elbow, left_wrist
        )
        results['left_hand'] = T_left_hand
        
        # Calculate right arm end-effector (right hand)
        right_shoulder = [
            joint_angles.get('right_shoulder_yaw', 0),
            joint_angles.get('right_shoulder_pitch', 0),
            joint_angles.get('right_shoulder_roll', 0)
        ]
        right_elbow = joint_angles.get('right_elbow', 0)
        right_wrist = [
            joint_angles.get('right_wrist_yaw', 0),
            joint_angles.get('right_wrist_pitch', 0), 
            joint_angles.get('right_wrist_roll', 0)
        ]
        
        T_right_hand = self.right_arm_fk.arm_forward_kinematics(
            right_shoulder, right_elbow, right_wrist
        )
        results['right_hand'] = T_right_hand
        
        # Calculate left leg end-effector (left foot)
        left_hip = [
            joint_angles.get('left_hip_yaw', 0),
            joint_angles.get('left_hip_pitch', 0),
            joint_angles.get('left_hip_roll', 0)
        ]
        left_knee = joint_angles.get('left_knee', 0)
        left_ankle = [
            joint_angles.get('left_ankle_pitch', 0),
            joint_angles.get('left_ankle_roll', 0)
        ]
        
        T_left_foot = self.left_leg_fk.leg_forward_kinematics(
            left_hip, left_knee, left_ankle
        )
        results['left_foot'] = T_left_foot
        
        # Calculate right leg end-effector (right foot)
        right_hip = [
            joint_angles.get('right_hip_yaw', 0),
            joint_angles.get('right_hip_pitch', 0),
            joint_angles.get('right_hip_roll', 0)
        ]
        right_knee = joint_angles.get('right_knee', 0)
        right_ankle = [
            joint_angles.get('right_ankle_pitch', 0),
            joint_angles.get('right_ankle_roll', 0)
        ]
        
        T_right_foot = self.right_leg_fk.leg_forward_kinematics(
            right_hip, right_knee, right_ankle
        )
        results['right_foot'] = T_right_foot
        
        return results
```

## Inverse Kinematics

Inverse kinematics calculates the required joint angles to achieve desired end-effector positions and orientations. This is more complex than forward kinematics and often requires iterative or analytical solutions.

### Analytical IK for Arms

For simple arm configurations, analytical solutions exist:

```python
import math

class AnalyticalArmIK:
    def __init__(self, upper_arm_length=0.35, lower_arm_length=0.35):
        self.l1 = upper_arm_length  # Upper arm length
        self.l2 = lower_arm_length  # Lower arm length
    
    def solve_3dof_arm(self, target_position, shoulder_position=[0, 0, 0]):
        """
        Solve inverse kinematics for a simplified 3-DOF arm
        Assumes shoulder at origin, with 1 DOF in each of x, y, z planes
        target_position: [x, y, z] in global coordinates
        """
        # Transform target to shoulder-centered coordinates
        x = target_position[0] - shoulder_position[0]
        y = target_position[1] - shoulder_position[1]
        z = target_position[2] - shoulder_position[2]
        
        # Calculate distance from shoulder to target
        r = math.sqrt(x**2 + y**2)
        d = math.sqrt(r**2 + z**2)
        
        # Check if target is reachable
        if d > (self.l1 + self.l2):
            print(f"Target unreachable: {target_position}")
            return None  # Target too far
        elif d < abs(self.l1 - self.l2):
            print(f"Target unreachable: inside workspace")
            return None  # Target too close
        
        # Calculate joint angles using law of cosines
        # Angle at shoulder
        cos_shoulder = (self.l1**2 + d**2 - self.l2**2) / (2 * self.l1 * d)
        angle_shoulder = math.acos(max(-1, min(1, cos_shoulder)))  # Clamp to valid range
        
        # Angle at elbow
        cos_elbow = (self.l1**2 + self.l2**2 - d**2) / (2 * self.l1 * self.l2)
        angle_elbow = math.pi - math.acos(max(-1, min(1, cos_elbow)))
        
        # Shoulder azimuth (rotation about z-axis)
        azimuth = math.atan2(y, x)
        
        # Shoulder elevation (rotation about y-axis)
        elevation = math.atan2(z, r)
        
        # Calculate final joint angles
        shoulder_yaw = azimuth
        shoulder_pitch = elevation + angle_shoulder
        elbow_angle = angle_elbow
        
        return {
            'shoulder_yaw': shoulder_yaw,
            'shoulder_pitch': shoulder_pitch, 
            'elbow_angle': elbow_angle
        }

# Example usage
arm_ik = AnalyticalArmIK()
target = [0.5, 0.0, 0.3]  # 50cm forward, 0cm sideways, 30cm up
joint_angles = arm_ik.solve_3dof_arm(target)

if joint_angles:
    print(f"Shoulder Yaw: {joint_angles['shoulder_yaw']:.3f}")
    print(f"Shoulder Pitch: {joint_angles['shoulder_pitch']:.3f}")
    print(f"Elbow Angle: {joint_angles['elbow_angle']:.3f}")
```

### Iterative IK for Complex Chains

For more complex chains like full arms or legs, iterative methods are more practical:

```python
import numpy as np
from scipy.optimize import minimize

class IterativeIK:
    def __init__(self, fk_solver, max_iterations=100, tolerance=1e-4):
        self.fk_solver = fk_solver
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def jacobian_transpose_method(self, initial_joints, target_position, 
                                  end_effector_func, joint_limits=None):
        """
        Solve inverse kinematics using Jacobian transpose method
        """
        current_joints = np.array(initial_joints)
        
        for iteration in range(self.max_iterations):
            # Calculate current end-effector position
            current_pos = end_effector_func(current_joints)
            
            # Calculate error
            error = target_position - current_pos
            error_norm = np.linalg.norm(error)
            
            # Check if we've reached the target
            if error_norm < self.tolerance:
                print(f"Reached target at iteration {iteration}")
                break
            
            # Calculate Jacobian (simplified finite difference)
            jacobian = self.calculate_jacobian(current_joints, end_effector_func)
            
            # Update joint angles using Jacobian transpose
            joint_delta = 0.01 * jacobian.T @ error  # Learning rate * J^T * error
            current_joints += joint_delta
            
            # Apply joint limits if provided
            if joint_limits:
                for i, (min_limit, max_limit) in enumerate(joint_limits):
                    current_joints[i] = np.clip(current_joints[i], min_limit, max_limit)
        
        return current_joints, error_norm, iteration
    
    def calculate_jacobian(self, joints, end_effector_func, delta_q=1e-5):
        """
        Calculate Jacobian matrix using finite differences
        """
        n_joints = len(joints)
        
        # Get current end-effector position
        current_pos = end_effector_func(joints)
        
        # Initialize Jacobian (3xN for position only, 6xN for pose)
        jacobian = np.zeros((3, n_joints))
        
        # Calculate each column of Jacobian
        for i in range(n_joints):
            # Perturb joint i
            joints_plus = joints.copy()
            joints_minus = joints.copy()
            joints_plus[i] += delta_q
            joints_minus[i] -= delta_q
            
            # Calculate end-effector positions
            pos_plus = end_effector_func(joints_plus)
            pos_minus = end_effector_func(joints_minus)
            
            # Calculate Jacobian column (partial derivatives)
            jacobian[:, i] = (pos_plus - pos_minus) / (2 * delta_q)
        
        return jacobian

# Example usage with a mock forward kinematics function
def mock_arm_fk(joint_angles):
    """
    Mock forward kinematics for demonstration
    Simplified model: x = l1*cos(q1) + l2*cos(q1+q2)
                     y = l1*sin(q1) + l2*sin(q1+q2)
    """
    l1, l2 = 0.35, 0.35  # Arm lengths
    q1, q2 = joint_angles[0], joint_angles[1]
    
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    
    # Return 3D position (z=0 for planar arm)
    return np.array([x, y, 0])

# Solve IK problem
ik_solver = IterativeIK(mock_arm_fk)
initial_joints = [0.1, 0.1]  # Initial joint angles
target = np.array([0.4, 0.3, 0.0])  # Target position

final_joints, final_error, iterations = ik_solver.jacobian_transpose_method(
    initial_joints, target, mock_arm_fk
)

print(f"Final joints: {final_joints}")
print(f"Final error: {final_error}")
print(f"Iterations: {iterations}")
```

## Center of Mass Calculation

The center of mass (CoM) is critical for humanoid robot balance:

```python
class CenterOfMassCalculator:
    def __init__(self):
        # Define link masses and positions relative to joint frames
        self.link_properties = {
            # Format: [mass in kg, [x, y, z] com offset from joint in meters]
            'trunk': [15.0, [0.0, 0.0, 0.1]],  # torso
            'head': [3.0, [0.0, 0.0, 0.05]],   # head
            'left_upper_arm': [2.0, [0.0, 0.0, -0.175]],  # upper arm
            'left_lower_arm': [1.5, [0.0, 0.0, -0.175]],  # lower arm
            'right_upper_arm': [2.0, [0.0, 0.0, -0.175]],
            'right_lower_arm': [1.5, [0.0, 0.0, -0.175]],
            'left_upper_leg': [5.0, [0.0, 0.0, -0.225]],  # upper leg
            'left_lower_leg': [4.0, [0.0, 0.0, -0.225]],  # lower leg
            'right_upper_leg': [5.0, [0.0, 0.0, -0.225]],
            'right_lower_leg': [4.0, [0.0, 0.0, -0.225]],
            'left_foot': [1.0, [0.1, 0.0, -0.05]],  # foot
            'right_foot': [1.0, [0.1, 0.0, -0.05]]
        }
    
    def calculate_com(self, fk_results, trunk_pose):
        """
        Calculate overall center of mass
        fk_results: results from forward kinematics
        trunk_pose: pose of the trunk (base of robot)
        """
        total_mass = 0.0
        weighted_positions = np.array([0.0, 0.0, 0.0])
        
        # Calculate CoM for each link
        for link_name, (mass, local_com_offset) in self.link_properties.items():
            # Get the transform for this link
            if link_name in fk_results:
                T = fk_results[link_name]
                # Calculate CoM position in global frame
                local_com = np.array(local_com_offset + [1])  # homogeneous coordinates
                global_com = T @ local_com
                global_com_pos = global_com[:3]  # Remove homogeneous coordinate
            else:
                # For trunk, use its pose directly
                if link_name == 'trunk':
                    # Calculate trunk CoM from its pose
                    T_trunk = np.eye(4)  # Identity for trunk base
                    T_trunk[:3, :3] = self.rotation_from_pose(trunk_pose['orientation'])
                    T_trunk[:3, 3] = trunk_pose['position']
                    
                    local_com = np.array(local_com_offset + [1])
                    global_com = T_trunk @ local_com
                    global_com_pos = global_com[:3]
                else:
                    # If link not in FK results, skip (e.g., trunk handled separately)
                    continue
            
            # Add to weighted sum
            weighted_positions += mass * global_com_pos
            total_mass += mass
        
        if total_mass > 0:
            overall_com = weighted_positions / total_mass
        else:
            overall_com = np.array([0.0, 0.0, 0.0])
        
        return overall_com, total_mass
    
    def rotation_from_pose(self, orientation_quat):
        """Convert quaternion to rotation matrix"""
        x, y, z, w = orientation_quat
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
```

## Practical Considerations

### Joint Limits and Constraints

```python
class JointConstraintHandler:
    def __init__(self):
        # Define joint limits for a typical humanoid robot
        self.joint_limits = {
            # Hip joints
            'left_hip_yaw': (-0.7, 0.7),      # ~±40 degrees
            'left_hip_pitch': (-0.4, 1.0),    # ~-23 to +57 degrees
            'left_hip_roll': (-0.4, 0.4),     # ~±23 degrees
            'right_hip_yaw': (-0.7, 0.7),
            'right_hip_pitch': (-0.4, 1.0),
            'right_hip_roll': (-0.4, 0.4),
            
            # Knee joints
            'left_knee': (0.0, 2.3),          # ~0 to +132 degrees (flexion only)
            'right_knee': (0.0, 2.3),
            
            # Ankle joints
            'left_ankle_pitch': (-0.4, 0.7),  # ~-23 to +40 degrees
            'left_ankle_roll': (-0.4, 0.4),
            'right_ankle_pitch': (-0.4, 0.7),
            'right_ankle_roll': (-0.4, 0.4),
            
            # Shoulder joints
            'left_shoulder_yaw': (-1.57, 1.57),    # ~±90 degrees
            'left_shoulder_pitch': (-2.0, 1.0),    # ~-115 to +57 degrees
            'left_shoulder_roll': (-2.0, 1.0),
            'right_shoulder_yaw': (-1.57, 1.57),
            'right_shoulder_pitch': (-2.0, 1.0),
            'right_shoulder_roll': (-1.0, 2.0),
            
            # Elbow joints
            'left_elbow': (0.0, 2.5),          # ~0 to +143 degrees (flexion only)
            'right_elbow': (0.0, 2.5),
            
            # Wrist joints
            'left_wrist_yaw': (-0.7, 0.7),     # ~±40 degrees
            'left_wrist_pitch': (-0.7, 0.7),
            'right_wrist_yaw': (-0.7, 0.7),
            'right_wrist_pitch': (-0.7, 0.7)
        }
    
    def check_joint_limits(self, joint_angles):
        """Check if joint angles are within limits"""
        violations = []
        
        for joint_name, angle in joint_angles.items():
            if joint_name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint_name]
                
                if angle < min_limit or angle > max_limit:
                    violations.append({
                        'joint': joint_name,
                        'angle': angle,
                        'min_limit': min_limit,
                        'max_limit': max_limit
                    })
        
        return len(violations) == 0, violations
    
    def apply_joint_limits(self, joint_angles):
        """Apply joint limits by clipping values"""
        limited_angles = joint_angles.copy()
        
        for joint_name, angle in joint_angles.items():
            if joint_name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint_name]
                limited_angles[joint_name] = np.clip(angle, min_limit, max_limit)
        
        return limited_angles
```

## Hands-on Exercise

1. Implement a complete forward kinematics solver for a simplified humanoid model with 12 DOF (6 per leg).

2. Create an inverse kinematics solver that can position a humanoid robot's foot at a given location.

3. Using the educational AI, explore how kinematic constraints affect the range of motion in humanoid robots.

The next section will explore the dynamics of humanoid robots, including how forces affect their motion.