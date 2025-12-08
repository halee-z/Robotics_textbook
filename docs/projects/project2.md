---
sidebar_position: 3
---

# Getting Started with Projects

## Overview

This section provides hands-on projects that integrate the concepts learned throughout the textbook. Each project builds on multiple topics including ROS 2, Vision-Language Models, simulation environments, and humanoid robotics control. These projects are designed to be challenging yet achievable, allowing students to apply theoretical knowledge in practical implementations.

## Project Structure

Each project follows a consistent structure:

1. **Learning Objectives**: Clear goals for what you'll learn
2. **Prerequisites**: Knowledge and tools required
3. **Implementation Steps**: Detailed instructions for implementation
4. **Testing**: How to verify your implementation works
5. **Extensions**: Opportunities to extend the project further

## Project 1: Autonomous Object Retrieval System

### Learning Objectives
- Integrate perception and manipulation for a complete task
- Use VLMs to identify and locate objects
- Implement a complete task pipeline from perception to action
- Combine navigation, manipulation, and planning

### Prerequisites
- Understanding of ROS 2 basics
- Knowledge of robot kinematics
- Familiarity with perception concepts
- Basic control systems knowledge

### Implementation Steps

#### Step 1: Environment Setup
```bash
# Create a workspace for the project
mkdir -p ~/object_retrieval_ws/src
cd ~/object_retrieval_ws

# Build the workspace with necessary packages
colcon build
source install/setup.bash
```

#### Step 2: Perception System
The perception system uses Vision-Language Models to identify objects:

```python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import clip  # Using CLIP for vision-language model

class ObjectRetrievalPerceptor:
    def __init__(self):
        rospy.init_node('object_retrieval_perceptor')
        
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # ROS interface
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.object_pub = rospy.Publisher("/detected_object", Point, queue_size=10)
        
        # Define object vocabulary for the task
        self.object_vocabulary = [
            "red cup", "blue bottle", "green box", 
            "yellow book", "black bag", "white mug"
        ]
        
        self.current_image = None
    
    def image_callback(self, msg):
        # Convert ROS image to PIL Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.current_image = cv_image
        
    def identify_object(self, target_descriptor):
        """
        Identify and locate an object based on a text descriptor
        """
        if self.current_image is None:
            return None
        
        # Preprocess image
        image_input = self.preprocess(self.current_image).unsqueeze(0).to(self.device)
        
        # Create text descriptions
        text_descriptions = [f"a photo of {obj}" for obj in self.object_vocabulary]
        text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(self.device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            
            # Calculate similarities
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            
            # Find the best match
            best_match_idx = similarities[0].argmax().item()
            best_match_score = similarities[0][best_match_idx].item()
            best_match_object = self.object_vocabulary[best_match_idx]
            
            # Only return if it matches the target descriptor and confidence is high
            if target_descriptor.lower() in best_match_object.lower() and best_match_score > 0.7:
                # In a real implementation, you would need to extract the position
                # For this example, we'll return a dummy position
                object_position = Point(x=1.0, y=0.5, z=0.0)
                return object_position, best_match_object, best_match_score
        
        return None, None, 0.0
```

#### Step 3: Navigation System
Implement navigation to approach the object location:

```python
import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class NavigationController:
    def __init__(self):
        rospy.init_node('navigation_controller')
        
        # Action client for move_base
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server")
        
        # Robot pose
        self.current_pose = None
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
    
    def navigate_to(self, target_position):
        """
        Navigate the robot to a target position
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Set the target position
        goal.target_pose.pose.position = target_position
        # Set a default orientation (facing forward)
        goal.target_pose.pose.orientation.w = 1.0
        
        rospy.loginfo(f"Navigating to position: ({target_position.x}, {target_position.y})")
        
        # Send goal to move_base
        self.move_base_client.send_goal(goal)
        
        # Wait for result
        finished_within_time = self.move_base_client.wait_for_result(rospy.Duration(300))  # 5 minutes timeout
        
        if not finished_within_time:
            self.move_base_client.cancel_goal()
            rospy.logerr("Navigation timed out")
            return False
        
        state = self.move_base_client.get_state()
        if state == 3:  # GoalState.SUCCEEDED
            rospy.loginfo("Navigation succeeded")
            return True
        else:
            rospy.logerr(f"Navigation failed with state: {state}")
            return False
```

#### Step 4: Manipulation System
Implement the arm control to grasp the object:

```python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class ManipulationController:
    def __init__(self):
        rospy.init_node('manipulation_controller')
        
        # Joint trajectory publisher for arm
        self.arm_traj_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
        
        # Gripper control
        self.gripper_pub = rospy.Publisher('/gripper_controller/command', String, queue_size=10)
        
        # Action client for more complex trajectories
        self.arm_client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', 
                                                      FollowJointTrajectoryAction)
    
    def grasp_object(self, object_position):
        """
        Execute a grasping motion to pick up an object
        """
        # Approach the object
        approach_poses = self.calculate_approach_poses(object_position)
        
        # Execute approach trajectory
        for pose in approach_poses:
            success = self.move_arm_to_pose(pose)
            if not success:
                rospy.logerr("Failed to move arm to approach position")
                return False
        
        # Close gripper to grasp
        self.close_gripper()
        
        # Lift the object
        lift_trajectory = self.calculate_lift_trajectory()
        success = self.execute_trajectory(lift_trajectory)
        
        if success:
            rospy.loginfo("Successfully grasped the object!")
            return True
        else:
            rospy.logerr("Failed to grasp the object")
            return False
    
    def calculate_approach_poses(self, object_position):
        """
        Calculate a trajectory of poses to approach the object
        """
        # This would contain inverse kinematics calculations
        # For this example, we'll return a simple trajectory
        approach_poses = [
            # Pre-grasp position (slightly above object)
            [object_position.x, object_position.y, object_position.z + 0.1, 0, 0, 0],
            # Grasp position (at object height)
            [object_position.x, object_position.y, object_position.z, 0, 0, 0]
        ]
        return approach_poses
    
    def move_arm_to_pose(self, pose):
        """
        Move the arm to a specific pose
        """
        # Implementation would use inverse kinematics
        # and publish trajectory commands
        trajectory = self.create_trajectory_for_pose(pose)
        return self.execute_trajectory(trajectory)
    
    def create_trajectory_for_pose(self, pose):
        """
        Create a joint trajectory to reach a specific pose
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = ["shoulder_joint", "elbow_joint", "wrist_joint"]  # Example
        
        point = JointTrajectoryPoint()
        # Calculate joint angles for pose using inverse kinematics
        joint_angles = self.inverse_kinematics(pose)  # This would be implemented
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(2.0)  # 2 seconds to reach pose
        
        trajectory.points.append(point)
        return trajectory
    
    def execute_trajectory(self, trajectory):
        """
        Execute a joint trajectory
        """
        self.arm_traj_pub.publish(trajectory)
        rospy.sleep(trajectory.points[-1].time_from_start)  # Wait for completion
        return True  # Simplified success check
    
    def close_gripper(self):
        """
        Close the gripper to grasp the object
        """
        command = String()
        command.data = "close"
        self.gripper_pub.publish(command)
        rospy.sleep(1.0)  # Wait for gripper to close
    
    def calculate_lift_trajectory(self):
        """
        Calculate a trajectory to lift the object
        """
        # Move up by 10cm
        trajectory = JointTrajectory()
        trajectory.joint_names = ["shoulder_joint", "elbow_joint", "wrist_joint"]
        
        # Lift trajectory point
        point = JointTrajectoryPoint()
        # This would be calculated based on current joint angles + lift motion
        point.positions = [0.1, -0.5, 0.2]  # Example values
        point.time_from_start = rospy.Duration(3.0)
        
        trajectory.points.append(point)
        return trajectory
    
    def inverse_kinematics(self, pose):
        """
        Calculate joint angles to reach a specific pose (simplified)
        """
        # In reality, this would be a complex inverse kinematics solver
        # For this example, return dummy values
        return [0.0, 0.0, 0.0]  # Placeholder
```

#### Step 5: System Integration
Create the main project that ties all components together:

```python
#!/usr/bin/env python

import rospy
from object_retrieval_perceptor import ObjectRetrievalPerceptor
from navigation_controller import NavigationController
from manipulation_controller import ManipulationController
from geometry_msgs.msg import Point
import time

class ObjectRetrievalSystem:
    def __init__(self):
        rospy.init_node('object_retrieval_system')
        
        # Initialize all components
        self.perceptor = ObjectRetrievalPerceptor()
        self.navigation = NavigationController()
        self.manipulation = ManipulationController()
        
        # Main control loop rate
        self.rate = rospy.Rate(1)  # 1 Hz
    
    def execute_task(self, object_description):
        """
        Execute the complete object retrieval task:
        1. Identify the object position
        2. Navigate to the object location
        3. Manipulate and retrieve the object
        """
        rospy.loginfo(f"Starting object retrieval for: {object_description}")
        
        # Step 1: Identify object
        rospy.loginfo("Looking for object...")
        for attempt in range(10):  # Try for 10 times
            object_pos, detected_obj, confidence = self.perceptor.identify_object(object_description)
            if object_pos is not None:
                rospy.loginfo(f"Found {detected_obj} with confidence {confidence:.2f} at {object_pos}")
                break
            rospy.sleep(1.0)
        else:
            rospy.logerr(f"Could not find {object_description}")
            return False
        
        # Step 2: Navigate to object
        rospy.loginfo("Navigating to object...")
        # Adjust target position to approach from front, not collide with object
        approach_position = Point(
            x=object_pos.x - 0.5,  # 0.5m in front of object
            y=object_pos.y,
            z=object_pos.z
        )
        
        nav_success = self.navigation.navigate_to(approach_position)
        if not nav_success:
            rospy.logerr("Navigation to object failed")
            return False
        
        # Step 3: Manipulate and retrieve
        rospy.loginfo("Attempting to retrieve object...")
        grasp_success = self.manipulation.grasp_object(object_pos)
        if not grasp_success:
            rospy.logerr("Failed to grasp the object")
            return False
        
        rospy.loginfo("Object retrieval completed successfully!")
        return True
    
    def run(self):
        """
        Run the object retrieval system
        """
        # For this example, we'll retrieve a "red cup"
        target_object = "red cup"
        
        success = self.execute_task(target_object)
        
        if success:
            rospy.loginfo("Task completed successfully!")
        else:
            rospy.logerr("Task failed!")
        
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        system = ObjectRetrievalSystem()
        system.run()
    except rospy.ROSInterruptException:
        pass
```

### Testing the Project

1. **Run the simulation environment** with objects to retrieve
2. **Launch the perception system**: `rosrun object_retrieval object_perceptor.py`
3. **Launch the navigation system**: `rosrun object_retrieval navigation_controller.py`
4. **Launch the manipulation system**: `rosrun object_retrieval manipulation_controller.py`
5. **Run the integrated system**: `rosrun object_retrieval object_retrieval_system.py`

### Extensions

1. Add multiple object retrieval capabilities
2. Implement obstacle avoidance during navigation
3. Add speech interface for object identification
4. Implement a placement station to deliver objects to
5. Add learning capabilities to improve performance over time

## Project 2: Humanoid Robot Dance Choreography

### Learning Objectives
- Use whole-body control for expressive motion
- Sequence complex movements for choreography
- Implement smooth transitions between poses
- Synchronize movements with audio

### Prerequisites
- Understanding of humanoid robot kinematics
- Knowledge of control systems
- Basic understanding of audio processing

### Implementation Steps

#### Step 1: Dance Motion Primitives
```python
import numpy as np

class DanceMotionPrimitives:
    def __init__(self):
        # Define basic dance poses (simplified example)
        self.poses = {
            "ready": self.get_ready_pose(),
            "wave": self.get_wave_pose(),
            "turn": self.get_turn_pose(),
            "step_left": self.get_step_pose(side="left"),
            "step_right": self.get_step_pose(side="right"),
            "raise_arms": self.get_raise_arms_pose(),
            "salsa_step": self.get_salsa_step_pose()
        }
    
    def get_ready_pose(self):
        """Standard standing position"""
        return {
            "left_hip_yaw": 0.0, "left_hip_pitch": -0.3, "left_hip_roll": 0.0,
            "left_knee": 0.6, "left_ankle_pitch": -0.3, "left_ankle_roll": 0.0,
            "right_hip_yaw": 0.0, "right_hip_pitch": -0.3, "right_hip_roll": 0.0,
            "right_knee": 0.6, "right_ankle_pitch": -0.3, "right_ankle_roll": 0.0,
            "left_shoulder_yaw": 0.0, "left_shoulder_pitch": -0.2, "left_shoulder_roll": 0.0,
            "left_elbow": 0.5, "right_shoulder_yaw": 0.0, "right_shoulder_pitch": -0.2,
            "right_shoulder_roll": 0.0, "right_elbow": 0.5,
            "head_yaw": 0.0, "head_pitch": 0.0
        }
    
    def get_wave_pose(self):
        """Position for waving gesture"""
        base = self.get_ready_pose()
        # Modify just the arm joints for waving
        base["left_shoulder_pitch"] = -0.5
        base["left_shoulder_roll"] = 0.5
        base["left_elbow"] = 1.5
        return base
    
    def get_turn_pose(self):
        """Position to facilitate turning motion"""
        base = self.get_ready_pose()
        # Shift weight to right leg for turning
        base["left_hip_pitch"] = -0.2
        base["right_hip_pitch"] = -0.4
        base["left_ankle_pitch"] = -0.1
        base["right_ankle_pitch"] = -0.5
        base["left_hip_roll"] = 0.2
        base["right_hip_roll"] = -0.2
        return base
    
    def get_step_pose(self, side="left"):
        """Position for stepping motion"""
        base = self.get_ready_pose()
        if side == "left":
            # Load weight on right leg, prepare left for step
            base["left_hip_pitch"] = -0.1  # Less weight on stepping leg
            base["right_hip_pitch"] = -0.5  # More weight on support leg
            base["left_knee"] = 0.3  # Slightly bent for step
        else:
            base["right_hip_pitch"] = -0.1
            base["left_hip_pitch"] = -0.5
            base["right_knee"] = 0.3
        return base
    
    def get_raise_arms_pose(self):
        """Position with arms raised"""
        base = self.get_ready_pose()
        base["left_shoulder_pitch"] = 1.0
        base["left_shoulder_roll"] = 0.2
        base["left_elbow"] = 0.2
        base["right_shoulder_pitch"] = 1.0
        base["right_shoulder_roll"] = -0.2
        base["right_elbow"] = 0.2
        return base
    
    def get_salsa_step_pose(self):
        """Salsa-specific stepping pose"""
        base = self.get_ready_pose()
        base["left_hip_pitch"] = -0.2
        base["right_hip_pitch"] = -0.4
        base["left_knee"] = 0.8  # More bent for dynamic movement
        base["right_knee"] = 0.4
        # Add some hip movement for salsa style
        base["left_hip_yaw"] = 0.3
        base["right_hip_yaw"] = -0.3
        return base
    
    def create_transition_trajectory(self, start_pose, end_pose, duration, steps=20):
        """
        Create smooth trajectory between two poses
        """
        trajectory = []
        time_step = duration / steps
        
        for i in range(steps + 1):
            progress = i / steps
            # Linear interpolation between poses
            current_pose = {}
            for joint in start_pose:
                start_val = start_pose[joint]
                end_val = end_pose[joint]
                current_val = start_val + progress * (end_val - start_val)
                current_pose[joint] = current_val
            
            trajectory.append({
                "time": i * time_step,
                "joint_angles": current_pose
            })
        
        return trajectory
```

#### Step 2: Choreography Sequencer
```python
import time
import threading
from dance_motion_primitives import DanceMotionPrimitives

class ChoreographySequencer:
    def __init__(self):
        self.primitives = DanceMotionPrimitives()
        self.current_move = "ready"
        self.is_performing = False
        self.performance_thread = None
        
    def sequence_basic_dance(self):
        """
        Sequence a basic dance routine
        """
        # Define a simple sequence of moves
        sequence = [
            {"move": "wave", "duration": 2.0},
            {"move": "raise_arms", "duration": 1.5},
            {"move": "step_left", "duration": 1.0},
            {"move": "step_right", "duration": 1.0},
            {"move": "salsa_step", "duration": 1.0},
            {"move": "turn", "duration": 2.0},
            {"move": "ready", "duration": 1.0}
        ]
        
        # Repeat the sequence 3 times
        full_sequence = sequence * 3
        
        return full_sequence
    
    def execute_choreography(self, sequence, tempo=120):
        """
        Execute a choreographed sequence of movements
        """
        if self.is_performing:
            rospy.logwarn("Already performing a sequence, skipping request")
            return False
        
        self.is_performing = True
        rospy.loginfo("Starting dance performance")
        
        current_pose = self.primitives.poses["ready"]
        
        for i, move_spec in enumerate(sequence):
            if not self.is_performing:
                rospy.loginfo("Performance stopped by user")
                break
                
            move_name = move_spec["move"]
            duration = move_spec["duration"]
            
            rospy.loginfo(f"Executing move {i+1}/{len(sequence)}: {move_name}")
            
            # Get the target pose
            target_pose = self.primitives.poses[move_name]
            
            # Create and execute transition
            transition = self.primitives.create_transition_trajectory(
                current_pose, target_pose, duration
            )
            
            # Execute the movement
            self.execute_trajectory(transition)
            
            # Update current pose
            current_pose = target_pose
            
            rospy.loginfo(f"Completed move: {move_name}")
        
        rospy.loginfo("Dance performance completed")
        self.is_performing = False
        return True
    
    def execute_trajectory(self, trajectory):
        """
        Execute a trajectory by publishing to joint controllers
        """
        # In a real implementation, this would publish to ROS topics
        # For simulation, we'll just sleep for the duration
        total_duration = trajectory[-1]["time"] if trajectory else 0
        time.sleep(total_duration)
    
    def start_performance(self):
        """
        Start the dance performance in a separate thread
        """
        sequence = self.sequence_basic_dance()
        
        self.performance_thread = threading.Thread(
            target=self.execute_choreography,
            args=(sequence,)
        )
        self.performance_thread.start()
    
    def stop_performance(self):
        """
        Stop the current performance
        """
        self.is_performing = False
        if self.performance_thread:
            self.performance_thread.join()
```

### Testing the Project

1. Load the humanoid robot model in your simulation environment
2. Run the choreography sequencer
3. Start a basic dance performance 
4. Observe the robot executing the programmed dance moves
5. Extend the sequence with additional moves

### Extensions

1. Add audio synchronization capabilities
2. Implement learning from human demonstrations
3. Add interactive elements responding to music
4. Enable real-time performance modifications
5. Implement crowd-responsive behaviors

## Project 3: Educational Tutor Robot

### Learning Objectives
- Implement adaptive tutoring using AI
- Integrate HRI principles for teaching
- Develop personalized learning paths
- Use multimodal interaction (speech, gesture, visual)

### Prerequisites
- Understanding of HRI principles
- Knowledge of AI and machine learning concepts
- Familiarity with educational psychology basics

### Implementation Steps

#### Step 1: Student Model Tracker
```python
class StudentModel:
    def __init__(self, student_id):
        self.student_id = student_id
        self.knowledge_state = {}  # Concepts and mastery levels
        self.learning_style = "balanced"  # visual, auditory, kinesthetic, balanced
        self.motivation_level = 0.7
        self.confusion_indicators = []
        self.progress_history = []
    
    def update_knowledge(self, concept, mastery_level):
        """
        Update the student's mastery level for a concept
        """
        self.knowledge_state[concept] = {
            "mastery": mastery_level,  # 0.0 to 1.0
            "last_attempt": time.time(),
            "attempts": self.knowledge_state.get(concept, {}).get("attempts", 0) + 1
        }
    
    def is_confused(self):
        """
        Determine if the student is currently confused
        """
        # Simple heuristic: if multiple recent attempts failed, consider confused
        recent_confusion = any(c["timestamp"] > time.time() - 300 for c in self.confusion_indicators[-5:])  # Last 5 minutes
        return recent_confusion
    
    def get_difficulty_recommendation(self, concept):
        """
        Recommend difficulty level based on student's mastery
        """
        current_mastery = self.knowledge_state.get(concept, {}).get("mastery", 0.0)
        
        if current_mastery < 0.3:
            return "basic"
        elif current_mastery < 0.7:
            return "intermediate"
        else:
            return "advanced"
```

#### Step 2: Adaptive Tutor Controller
```python
from student_model import StudentModel
from text_to_speech import TextToSpeech
from speech_recognition import SpeechRecognizer

class AdaptiveTutor:
    def __init__(self, robot_name="EducationalBot"):
        self.robot_name = robot_name
        self.current_student = None
        self.concept_repository = self.load_concepts()
        self.tts = TextToSpeech()
        self.speech_rec = SpeechRecognizer()
        
    def load_concepts(self):
        """
        Load educational content repository
        """
        # This would typically load from a database or file
        return {
            "math": {
                "addition": {
                    "basic": "Let's learn addition! Addition is when we combine numbers together.",
                    "intermediate": "Now let's try adding larger numbers with carrying over.",
                    "advanced": "Let's solve complex addition problems with multiple digits."
                },
                "subtraction": {
                    "basic": "Subtraction is taking away one number from another.",
                    "intermediate": "Let's try subtraction with borrowing.",
                    "advanced": "Let's solve complex subtraction word problems."
                }
            },
            "science": {
                "gravity": {
                    "basic": "Gravity is a force that pulls things down.",
                    "intermediate": "Gravity is the force that keeps planets in orbit.",
                    "advanced": "Gravity is described by Einstein's theory of general relativity."
                }
            }
        }
    
    def start_tutoring_session(self, student_id):
        """
        Initialize a tutoring session for a student
        """
        self.current_student = StudentModel(student_id)
        self.introduce_robot()
        
    def introduce_robot(self):
        """
        Introduce the tutoring robot to the student
        """
        introduction = f"Hello! I'm {self.robot_name}, your educational assistant. " \
                      f"I'm here to help you learn. What subject would you like to explore today?"
        self.tts.speak(introduction)
    
    def conduct_lesson(self, subject, concept):
        """
        Conduct an adaptive lesson based on student model
        """
        if not self.current_student:
            self.tts.speak("Please start a session first.")
            return
        
        # Determine appropriate difficulty
        difficulty = self.current_student.get_difficulty_recommendation(concept)
        
        # Get appropriate content
        content = self.concept_repository[subject][concept].get(difficulty, 
                                                              self.concept_repository[subject][concept]["basic"])
        
        # Deliver content with appropriate modalities based on learning style
        self.deliver_content(content, difficulty)
        
        # Assess understanding
        self.assess_understanding(concept)
    
    def deliver_content(self, content, difficulty):
        """
        Deliver content using appropriate modalities
        """
        # Speak the content
        self.tts.speak(content)
        
        # Show visual aids if available
        self.display_visual_content(content)
        
        # Use gestures appropriate to content
        self.use_educational_gestures(content)
    
    def assess_understanding(self, concept):
        """
        Assess if the student understood the concept
        """
        # Ask a question about the concept
        question = self.generate_question(concept)
        
        self.tts.speak(question)
        
        # Listen for student response
        response = self.speech_rec.listen_for_response(timeout=30)
        
        # Evaluate response
        is_correct = self.evaluate_response(response, concept)
        
        # Update student model
        mastery_change = 0.1 if is_correct else -0.1
        current_mastery = self.current_student.knowledge_state.get(concept, {}).get("mastery", 0.0)
        new_mastery = max(0.0, min(1.0, current_mastery + mastery_change))
        
        self.current_student.update_knowledge(concept, new_mastery)
        
        # Provide feedback
        if is_correct:
            self.tts.speak("Great job! You understand this concept well.")
            self.use_positive_gesture()
        else:
            self.tts.speak("That's okay, let me explain this concept again in a different way.")
            self.use_encouraging_gesture()
    
    def generate_question(self, concept):
        """
        Generate an appropriate question for the concept and student level
        """
        # Simplified question generation
        basic_questions = {
            "addition": "What is 2 plus 2?",
            "subtraction": "What is 5 minus 3?",
            "gravity": "What force pulls things down?"
        }
        
        return basic_questions.get(concept, f"Can you explain {concept} to me?")
    
    def evaluate_response(self, response, concept):
        """
        Evaluate if the response is correct
        """
        # Simplified evaluation
        correct_responses = {
            "addition": ["4", "four"],
            "subtraction": ["2", "two"],
            "gravity": ["gravity", "down", "force"]
        }
        
        if concept in correct_responses:
            response_lower = response.lower() if response else ""
            return any(correct in response_lower for correct in correct_responses[concept])
        
        # For open-ended questions, we can't easily evaluate
        return True  # Assume positive for now
    
    def display_visual_content(self, content):
        """
        Display visual content on robot's screen or in environment
        """
        # This would interface with the robot's display system
        # For simulation, we'll just print
        print(f"Displaying visual content: {content[:50]}...")
    
    def use_educational_gestures(self, content):
        """
        Use appropriate gestures to enhance learning
        """
        # Use gestures based on content keywords
        if "addition" in content.lower() or "plus" in content.lower():
            # Use gesture for combining/adding
            pass
        elif "subtraction" in content.lower() or "minus" in content.lower():
            # Use gesture for taking away
            pass
    
    def use_positive_gesture(self):
        """
        Use a positive gesture (like nodding or thumbs up)
        """
        print("Robot performs positive gesture")
    
    def use_encouraging_gesture(self):
        """
        Use an encouraging gesture
        """
        print("Robot performs encouraging gesture")
```

#### Step 3: Main Tutoring System
```python
#!/usr/bin/env python

import rospy
from adaptive_tutor import AdaptiveTutor

class EducationalTutorSystem:
    def __init__(self):
        rospy.init_node('educational_tutor_system')
        
        self.tutor = AdaptiveTutor("LearningBot")
        self.session_active = False
        
        # Create service for starting sessions
        from std_srvs.srv import Trigger, TriggerResponse
        self.session_service = rospy.Service('start_tutoring', Trigger, self.handle_start_session)
        
        rospy.loginfo("Educational Tutor System initialized")
    
    def handle_start_session(self, req):
        """
        Handle request to start a tutoring session
        """
        if not self.session_active:
            self.tutor.start_tutoring_session("student123")
            self.session_active = True
            rospy.loginfo("Tutoring session started")
            return TriggerResponse(success=True, message="Session started")
        else:
            return TriggerResponse(success=False, message="Session already active")
    
    def run(self):
        """
        Run the educational tutoring system
        """
        rospy.loginfo("Educational Tutor System running - waiting for requests")
        rospy.spin()

if __name__ == '__main__':
    try:
        system = EducationalTutorSystem()
        system.run()
    except rospy.ROSInterruptException:
        pass
```

### Testing the Project

1. Deploy the tutoring system on your educational platform
2. Start a tutoring session via the service
3. Interact with the robot tutor through speech/interaction
4. Observe how the system adapts to your responses
5. Check how the student model updates based on interactions

### Extensions

1. Add multiple subject areas with detailed content
2. Implement more sophisticated student modeling
3. Add progress tracking and reporting
4. Include collaborative learning features
5. Add gamification elements to increase engagement

## Conclusion

These projects demonstrate the integration of multiple concepts from the textbook. Each project combines perception, control, interaction, and AI to create meaningful applications. The projects can be extended and customized based on specific educational or research goals.

The key to success in these projects is systematic testing, iterative development, and consideration of human factors in robot design and interaction.