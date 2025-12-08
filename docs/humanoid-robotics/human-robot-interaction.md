---
sidebar_position: 5
---

# Human-Robot Interaction for Social Robotics

## Overview

Human-Robot Interaction (HRI) is a critical field that explores how humans and robots can effectively communicate and collaborate. For humanoid robots, which are designed to interact with humans in social environments, HRI encompasses not only task-oriented interaction but also social communication, emotional engagement, and behavioral adaptation. This section covers the theoretical foundations, technical implementations, and practical considerations for developing socially interactive humanoid robots.

## Fundamentals of Human-Robot Interaction

### Social Robotics Principles

Social robotics is predicated on several foundational principles that guide the design and implementation of social robots:

1. **Anthropomorphic Design**: Leveraging human-like features to facilitate more natural interaction
2. **Turn-Taking**: Implementing conversational and interaction turn-taking norms
3. **Context Awareness**: Understanding and responding to environmental and social context
4. **Emotional Intelligence**: Recognizing, expressing, and managing emotions
5. **Social Norms**: Following culturally-appropriate social behaviors

### HRI Taxonomies

HRI can be classified along several dimensions:

```python
class HRITypes:
    """
    Classification of different types of human-robot interaction
    """
    
    # Interaction Modes
    INTERACTION_MODES = {
        'cooperative': {
            'description': 'Humans and robots work together toward common goals',
            'characteristics': ['shared tasks', 'complementary roles', 'mutual support']
        },
        'collaborative': {
            'description': 'Humans and robots share planning and decision-making',
            'characteristics': ['joint planning', 'bidirectional communication', 'equal partnership']
        },
        'assistive': {
            'description': 'Robots provide assistance to humans',
            'characteristics': ['service-oriented', 'human-initiated', 'supportive']
        },
        'companion': {
            'description': 'Robots provide companionship and social interaction',
            'characteristics': ['social presence', 'emotional connection', 'long-term interaction']
        }
    }
    
    # Interaction Modalities
    MODALITIES = {
        'verbal': {
            'channel': 'auditory',
            'components': ['speech recognition', 'natural language processing', 'text-to-speech']
        },
        'nonverbal': {
            'channel': 'visual',
            'components': ['gestures', 'facial expressions', 'body posture', 'eye contact']
        },
        'physical': {
            'channel': 'haptic',
            'components': ['touch', 'proximity', 'force feedback', 'physical guidance']
        },
        'multimodal': {
            'channel': 'combined',
            'components': ['integration of multiple channels', 'context-aware']
        }
    }
    
    # Proxemics (Personal Space)
    PROXEMIC_ZONES = {
        'intimate': {
            'distance': (0, 0.45),
            'typical_uses': ['whispering', 'embracing', 'touching'],
            'robot_behavior': 'reserved for special applications'
        },
        'personal': {
            'distance': (0.45, 1.2),
            'typical_uses': ['conversations with friends', 'individual interactions'],
            'robot_behavior': 'primary interaction zone'
        },
        'social': {
            'distance': (1.2, 3.6),
            'typical_uses': ['business encounters', 'group conversations'],
            'robot_behavior': 'formal interaction zone'
        },
        'public': {
            'distance': (3.6, float('inf')),
            'typical_uses': ['public speaking', 'general awareness'],
            'robot_behavior': 'monitoring and attention'
        }
    }

# Example implementation of proxemics for robot navigation
class ProxemicController:
    def __init__(self, robot_name="SocialRobot"):
        self.robot_name = robot_name
        self.human_positions = {}
        self.cultural_profiles = {}  # Store cultural preferences
        self.personal_space = 0.8  # Default personal distance for this robot
    
    def calculate_comfortable_distance(self, human_id, cultural_background="default"):
        """
        Calculate appropriate social distance based on cultural background
        """
        cultural_factors = {
            'mediterranean': 0.6,  # Closer distances
            'north_american': 0.8,  # Standard distance
            'east_asian': 1.0,     # More distance preferred
            'middle_eastern': 0.7  # Moderate distance
        }
        
        factor = cultural_factors.get(cultural_background, 1.0)
        return self.personal_space * factor
    
    def update_human_position(self, human_id, position, orientation):
        """
        Update internal model of human positions and orientations
        """
        self.human_positions[human_id] = {
            'position': np.array(position),
            'orientation': np.array(orientation),
            'gaze_direction': None,
            'last_seen': time.time()
        }
    
    def calculate_navigation_target(self, human_id, desired_interaction_level="normal"):
        """
        Calculate where robot should position itself relative to human
        """
        if human_id not in self.human_positions:
            return None
        
        human_pos = self.human_positions[human_id]['position']
        human_orient = self.human_positions[human_id]['orientation']
        
        # Calculate appropriate distance based on cultural profile
        cultural_profile = self.cultural_profiles.get(human_id, "default")
        distance = self.calculate_comfortable_distance(human_id, cultural_profile)
        
        # Calculate target position relative to human orientation
        # Position robot at appropriate angle to human's forward direction
        if desired_interaction_level == "greeting":
            # Closer for greeting, but not invading space
            distance = max(distance * 0.8, 0.5)
            angle_offset = 0  # Face directly towards human
        elif desired_interaction_level == "attentive":
            # Standard distance, slight side angle
            angle_offset = np.pi / 6  # 30 degrees
        elif desired_interaction_level == "observational":
            # Further away, less engaged positioning
            distance = distance * 1.2
            angle_offset = np.pi / 3  # 60 degrees
        
        # Calculate target position
        target_x = human_pos[0] + distance * np.cos(human_orient[2] + angle_offset)
        target_y = human_pos[1] + distance * np.sin(human_orient[2] + angle_offset)
        target_z = human_pos[2]  # Same height as human
        
        return np.array([target_x, target_y, target_z])

# Example usage
proxemic_ctrl = ProxemicController("CompanionBot")
target_pos = proxemic_ctrl.calculate_navigation_target(
    "human_001", 
    desired_interaction_level="greeting"
)
if target_pos is not None:
    print(f"Recommended position for greeting: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
```

## Social Cues and Non-Verbal Communication

### Gaze Behavior

Gaze is one of the most important non-verbal communication channels in HRI:

```python
class GazeController:
    def __init__(self, robot_face_tracking):
        self.face_tracker = robot_face_tracking
        self.current_gaze_target = None
        self.gaze_patterns = self.initialize_gaze_patterns()
        self.attention_buffer = AttentionBuffer(size=10)
        
    def initialize_gaze_patterns(self):
        """
        Initialize different gaze patterns for various social contexts
        """
        return {
            'attentive': {
                'duration': (0.8, 2.0),
                'transition_time': 0.2,
                'saccadic_movement': True
            },
            'social': {
                'duration': (0.5, 1.5),
                'transition_time': 0.3,
                'distribution': 'group_aware'
            },
            'task_oriented': {
                'duration': (0.3, 1.0), 
                'transition_time': 0.1,
                'focus': 'relevant_objects'
            },
            'exploratory': {
                'duration': (0.2, 0.8),
                'transition_time': 0.05,
                'movement': 'random_walk'
            }
        }
    
    def calculate_gaze_target(self, interaction_context):
        """
        Calculate where the robot should look based on interaction context
        """
        active_humans = interaction_context.get('humans', [])
        objects_of_interest = interaction_context.get('objects', [])
        current_task = interaction_context.get('task', 'social')
        
        if current_task == 'greeting':
            # Look at the person being greeted
            if active_humans:
                target_person = self.identify_primary_interactant(active_humans)
                return self.look_at_person(target_person)
                
        elif current_task == 'group_interaction':
            # Distribute gaze among group members
            return self.distribute_gaze_among_group(active_humans)
            
        elif current_task == 'task_execution':
            # Look at task-relevant objects
            if objects_of_interest:
                target_object = self.select_relevant_object(objects_of_interest)
                return self.look_at_object(target_object)
        
        elif current_task == 'storytelling':
            # Alternate between audience and objects being discussed
            if active_humans:
                # Look at main audience member
                primary_person = self.identify_primary_interactant(active_humans)
                return self.look_at_person(primary_person)
        
        # Default: look at primary human or most salient stimulus
        if active_humans:
            primary_human = self.identify_primary_interactant(active_humans)
            return self.look_at_person(primary_human)
        else:
            # Look at center of visual field or most interesting object
            return self.calculate_foveal_point()
    
    def identify_primary_interactant(self, humans):
        """
        Determine which human is the primary focus of interaction
        """
        if len(humans) == 1:
            return humans[0]
        
        # Use multiple cues to determine primary interactant:
        # - Proximity to robot
        # - Orientation toward robot
        # - Recent speaking activity
        # - Attention indicators
        
        scores = {}
        robot_pos = np.array([0, 0, 0])  # Robot's position
        
        for human in humans:
            score = 0.0
            
            # Distance factor (closer = higher priority)
            distance = np.linalg.norm(np.array(human['position']) - robot_pos)
            if distance < 0.1:  # Invalid distance
                score += 0
            else:
                score += 1.0 / (distance + 0.1)  # Prevent division by zero
            
            # Orientation factor (facing robot = higher priority)
            if 'orientation' in human:
                robot_to_human = (np.array(human['position']) - robot_pos) / distance
                human_facing = np.array(human['orientation'])[:2]  # 2D projection
                alignment = np.dot(robot_to_human, human_facing)
                score += max(0, alignment)  # Only positive alignment contributes
            
            # Speaking factor (if applicable)
            if human.get('speaking', False):
                score += 2.0  # Bonus for speakers
            
            # Attention factor (if being tracked as attention focus)
            if human['id'] in self.attention_buffer:
                # Recently attended = higher priority
                attention_recentness = self.attention_buffer.get_recency(human['id'])
                score += attention_recentness * 0.5
            
            scores[human['id']] = score
        
        # Return human with highest score
        if scores:
            primary_id = max(scores, key=scores.get)
            return next((h for h in humans if h['id'] == primary_id), humans[0])
        else:
            return humans[0] if humans else None
    
    def generate_smooth_gaze_trajectory(self, current_gaze, target_gaze, duration):
        """
        Generate smooth trajectory from current gaze to target gaze
        """
        # Use sinusoidal interpolation for smooth acceleration/deceleration
        n_points = int(duration / 0.01)  # 100Hz for smooth motion
        times = np.linspace(0, duration, n_points)
        
        # Sinusoidal interpolation (smoother than linear)
        progress = 0.5 * (1 - np.cos(np.pi * times / duration))
        
        trajectory = []
        for prog in progress:
            intermediate_gaze = current_gaze + prog * (target_gaze - current_gaze)
            trajectory.append(intermediate_gaze)
        
        return trajectory
    
    def update_attention_model(self, current_attention):
        """
        Update internal attention model based on who is paying attention to robot
        """
        # Track attention patterns to improve future social interactions
        self.attention_buffer.update(current_attention)
    
    def express_attention(self, attention_type):
        """
        Use gaze to express different types of attention
        """
        if attention_type == "focused":
            # Prolonged gaze with minimal micro-movements
            self.set_gaze_stillness(0.9)
        elif attention_type == "acknowledging":
            # Brief gaze followed by social glance
            self.perform_acknowledging_gaze()
        elif attention_type == "curious":
            # Slight head tilt with focused gaze
            self.perform_curious_gaze()
        elif attention_type == "attentive":
            # Balanced gaze with normal micro-movements
            self.set_gaze_stillness(0.5)

class AttentionBuffer:
    def __init__(self, size=10):
        self.size = size
        self.buffer = {}  # id -> (timestamp, score)
    
    def update(self, attention_dict):
        """
        Update buffer with current attention status
        attention_dict: {human_id: attention_score}
        """
        current_time = time.time()
        for human_id, score in attention_dict.items():
            self.buffer[human_id] = (current_time, score)
        
        # Prune old entries
        cutoff_time = current_time - 30  # Remove entries older than 30 seconds
        to_remove = [hid for hid, (t, _) in self.buffer.items() if t < cutoff_time]
        for hid in to_remove:
            del self.buffer[hid]
    
    def get_recency(self, human_id):
        """
        Get recency score for a human (0-1 scale, 1 = most recent)
        """
        if human_id not in self.buffer:
            return 0.0
        
        timestamp, _ = self.buffer[human_id]
        time_diff = time.time() - timestamp
        # Recency decays exponentially over time
        return max(0.0, np.exp(-time_diff / 10.0))  # Half-life of 10 seconds
    
    def __contains__(self, human_id):
        return human_id in self.buffer
```

### Gesture Recognition and Generation

```python
class GestureController:
    def __init__(self):
        self.gesture_repertoire = self.initialize_gesture_repertoire()
        self.gesture_sequences = self.initialize_gesture_sequences()
        self.gesture_recognizer = GestureRecognizer()
        self.gesture_generator = GestureGenerator()
        
    def initialize_gesture_repertoire(self):
        """
        Initialize comprehensive gesture repertoire
        """
        return {
            # Emotive gestures
            "greeting": {
                "type": "emotive",
                "name": "hand_wave",
                "parameters": {
                    "arm": "right",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                    "duration": 2.0,
                    "cultural_variants": ["western_wave", "japanese_bow", "indian_namaste"]
                },
                "contexts": ["first_encounter", "farewell", "attention_getting"]
            },
            "acknowledgment": {
                "type": "emotive", 
                "name": "head_nod",
                "parameters": {
                    "amplitude": 0.3,
                    "frequency": 1.0, 
                    "duration": 1.0,
                    "head_tilt_degree": 0.1
                },
                "contexts": ["listening", "agreement", "confirmation"]
            },
            "emphasis": {
                "type": "emotive",
                "name": "hand_gesture",
                "parameters": {
                    "type": "open_palm",
                    "amplitude": 0.8,
                    "duration": 1.5,
                    "direction": "forward"
                },
                "contexts": ["storytelling", "instruction", "highlighting"]
            },
            "regulation": {
                "type": "regulatory",
                "name": "attention_direct",
                "parameters": {
                    "gaze_shift": True,
                    "arm_point": True,
                    "duration": 2.0
                },
                "contexts": ["turn_taking", "object_reference", "direction_guidance"]
            },
            "adaptive": {
                "type": "adaptive",
                "name": "comfort_adjustment",
                "parameters": {
                    "posture_shift": True,
                    "distance_adjust": True,
                    "duration": 1.0
                },
                "contexts": ["long_interaction", "space_comfort", "fatigue_reduction"]
            }
        }
    
    def select_relevant_gesture(self, context, urgency_level="normal", cultural_background="default"):
        """
        Select appropriate gesture based on context and cultural background
        """
        relevant_gestures = []
        
        # Filter gestures by context
        for gesture_name, gesture_def in self.gesture_repertoire.items():
            if context in gesture_def["contexts"]:
                # Apply cultural filtering
                if cultural_background in gesture_def["parameters"].get("cultural_variants", [cultural_background]):
                    relevant_gestures.append((gesture_name, gesture_def))
        
        if not relevant_gestures:
            # Default to acknowledgment gesture
            return "acknowledgment"
        
        # Apply priority/urgency filtering
        if urgency_level == "high":
            # Prioritize simple, clear gestures
            high_priority = [g for g in relevant_gestures 
                           if g[1]["type"] in ["emotive", "regulatory"]]
            if high_priority:
                relevant_gestures = high_priority
        elif urgency_level == "low":
            # Allow more complex/detailed gestures
            pass  # All relevant gestures are acceptable
        
        # Select based on additional heuristics
        # For now, return the first relevant gesture
        return relevant_gestures[0][0]
    
    def generate_gesture_sequence(self, gesture_name, intensity=0.8):
        """
        Generate a complete gesture sequence for execution
        """
        if gesture_name not in self.gesture_repertoire:
            raise ValueError(f"Unknown gesture: {gesture_name}")
        
        gesture_def = self.gesture_repertoire[gesture_name]
        parameters = gesture_def["parameters"]
        
        # Generate detailed movement sequence
        if gesture_def["name"] == "hand_wave":
            return self.generate_hand_wave_sequence(parameters, intensity)
        elif gesture_def["name"] == "head_nod":
            return self.generate_head_nod_sequence(parameters, intensity)
        elif gesture_def["name"] == "hand_gesture":
            return self.generate_hand_gesture_sequence(parameters, intensity)
        else:
            return self.generate_generic_sequence(gesture_def, parameters, intensity)
    
    def generate_hand_wave_sequence(self, params, intensity):
        """
        Generate sequence for hand waving gesture
        """
        duration = params["duration"] * (2.0 - intensity)  # Faster for higher intensity
        amplitude = params["amplitude"] * intensity
        frequency = params["frequency"]
        
        # Generate wave motion trajectory
        dt = 0.01  # 100Hz control
        time_points = np.arange(0, duration, dt)
        
        # Sine wave motion for natural wave
        motion_x = amplitude * np.sin(2 * np.pi * frequency * time_points)
        motion_y = amplitude * 0.5 * np.cos(2 * np.pi * frequency * time_points)  # Secondary motion
        
        trajectory = []
        for i, t in enumerate(time_points):
            waypoint = {
                "time": t,
                "right_arm": {
                    "shoulder_roll": motion_y[i] * 0.3,
                    "elbow_flex": motion_x[i] * 0.5,
                    "wrist_yaw": motion_x[i] * 0.2
                },
                "head": {
                    "yaw": motion_x[i] * 0.1  # Subtle head motion for engagement
                }
            }
            trajectory.append(waypoint)
        
        return trajectory
    
    def generate_head_nod_sequence(self, params, intensity):
        """
        Generate sequence for head nodding gesture
        """
        duration = params["duration"]
        amplitude = params["amplitude"] * intensity
        head_tilt = params["head_tilt_degree"] * intensity
        
        dt = 0.01
        time_points = np.arange(0, duration, dt)
        
        # Smooth nod motion (like a spring)
        motion = amplitude * np.sin(np.pi * time_points / duration)
        
        trajectory = []
        for i, t in enumerate(time_points):
            waypoint = {
                "time": t,
                "head": {
                    "pitch": motion[i],
                    "yaw": 0,
                    "roll": head_tilt * np.sin(2 * np.pi * t * 1.5) if t < duration/2 else 0
                }
            }
            trajectory.append(waypoint)
        
        return trajectory
    
    def execute_gesture(self, gesture_name, intensity=0.8, blocking=True):
        """
        Execute a gesture on the physical robot
        """
        try:
            sequence = self.generate_gesture_sequence(gesture_name, intensity)
            
            if blocking:
                # Execute the entire sequence synchronously
                for waypoint in sequence:
                    self.move_to_waypoint(waypoint)
                    if 'time' in waypoint:
                        time.sleep(waypoint['time'] - time.time() % 0.01)  # Sync to control rate
            else:
                # Execute asynchronously
                self.execute_sequence_async(sequence)
            
            return True
        except Exception as e:
            print(f"Gesture execution failed: {e}")
            return False
    
    def move_to_waypoint(self, waypoint):
        """
        Move robot joints to reach specified waypoint
        """
        # In a real implementation, this would interface with the robot's
        # joint controllers to reach the specified positions
        pass
    
    def execute_sequence_async(self, sequence):
        """
        Execute gesture sequence asynchronously
        """
        # In a real implementation, this would run the sequence
        # in a separate thread or process
        pass

class GestureRecognizer:
    """
    Recognizes human gestures and interprets their meaning
    """
    def __init__(self):
        self.known_gestures = self.initialize_known_gestures()
        self.recognition_model = self.load_recognition_model()
    
    def initialize_known_gestures(self):
        """
        Initialize the set of known human gestures
        """
        return {
            "wave": {
                "motion_pattern": ["arm_raised", "repetitive_swing"],
                "kinematic_signature": ["shoulder_rotation", "elbow_flexion"],
                "temporal_constraints": {"duration": (0.5, 3.0), "frequency": (1.0, 4.0)},
                "meaning": "greeting_attention"
            },
            "point": {
                "motion_pattern": ["arm_extended", "finger_extension"],
                "kinematic_signature": ["shoulder_yaw", "elbow_extension"],
                "temporal_constraints": {"duration": (0.2, 1.0), "velocity": "high"},
                "meaning": "object_reference_direction"
            },
            "beckon": {
                "motion_pattern": ["arm_outstretched", "curved_finger_motion"],
                "kinematic_signature": ["shoulder_abduction", "finger_flexion"],
                "temporal_constraints": {"duration": (1.0, 5.0), "rhythm": "slow_repetitive"},
                "meaning": "come_here_approach"
            },
            "stop": {
                "motion_pattern": ["palm_facing_outward", "arm_extended"],
                "kinematic_signature": ["shoulder_flexion", "wrist_extension"],
                "temporal_constraints": {"duration": (0.5, 2.0), "motion": "minimal"},
                "meaning": "stop_wait_halt"
            }
        }
    
    def recognize_gesture(self, skeleton_data, confidence_threshold=0.7):
        """
        Recognize gesture from human skeleton data
        
        Args:
            skeleton_data: Joint positions and movements over time
            confidence_threshold: Minimum confidence for recognition
            
        Returns:
            Recognized gesture name and confidence score
        """
        if len(skeleton_data) < 5:  # Need sufficient data points
            return None, 0.0
        
        # Extract motion features from skeleton data
        motion_features = self.extract_motion_features(skeleton_data)
        
        # Compare against known gestures
        best_match = None
        best_confidence = 0.0
        
        for gesture_name, gesture_def in self.known_gestures.items():
            confidence = self.match_gesture_pattern(motion_features, gesture_def)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = gesture_name
        
        if best_confidence >= confidence_threshold:
            return best_match, best_confidence
        else:
            return None, best_confidence
    
    def extract_motion_features(self, skeleton_data):
        """
        Extract relevant motion features from skeleton data
        """
        # Calculate velocities and accelerations
        velocities = np.gradient([s['joints']['right_wrist']['position'] for s in skeleton_data], axis=0)
        accelerations = np.gradient(velocities, axis=0)
        
        # Extract key joint positions over time
        features = {
            'right_wrist_trajectory': [s['joints']['right_wrist']['position'] for s in skeleton_data],
            'right_elbow_trajectory': [s['joints']['right_elbow']['position'] for s in skeleton_data],
            'shoulder_hip_alignment': [],  # Calculate alignment over time
            'motion_dynamics': {
                'velocity_magnitude': np.linalg.norm(velocities, axis=2),
                'acceleration_magnitude': np.linalg.norm(accelerations, axis=2)
            }
        }
        
        return features
    
    def match_gesture_pattern(self, motion_features, gesture_definition):
        """
        Match extracted motion features to gesture definition
        """
        # Simplified matching algorithm
        # In practice, this would use ML models or complex pattern matching
        
        # Check temporal constraints
        duration_ok = (len(motion_features['right_wrist_trajectory']) * 0.1  # Assuming 10 FPS
                      >= gesture_definition['temporal_constraints']['duration'][0] and
                      len(motion_features['right_wrist_trajectory']) * 0.1
                      <= gesture_definition['temporal_constraints']['duration'][1])
        
        # Check motion characteristics
        velocity_profile = motion_features['motion_dynamics']['velocity_magnitude']
        avg_velocity = np.mean(velocity_profile)
        velocity_ok = (avg_velocity >= 0.1)  # Has significant motion
        
        # Calculate similarity score
        score = 0.0
        if duration_ok: score += 0.4
        if velocity_ok: score += 0.3
        
        # Add other matching criteria...
        
        return min(1.0, score)  # Normalize to 0-1 range

class GestureGenerator:
    """
    Generates appropriate robot responses to human gestures
    """
    def __init__(self):
        self.response_mapping = self.initialize_response_mapping()
    
    def initialize_response_mapping(self):
        """
        Initialize mapping from human gestures to robot responses
        """
        return {
            "wave": {
                "appropriate_responses": ["greeting", "acknowledgment"],
                "social_rules": {"reciprocity": True, "timeliness": 0.5, "intensity_match": True},
                "cultural_modifiers": {
                    "japanese": {"bow_instead_of_wave": True},
                    "middle_eastern": {"handshake_followup": True}
                }
            },
            "point": {
                "appropriate_responses": ["regulation", "emphasis"],
                "social_rules": {"attention_following": True, "validation": True},
                "cultural_modifiers": {}
            },
            "beckon": {
                "appropriate_responses": ["approach", "acknowledgment"], 
                "social_rules": {"obedience": True, "gratitude": True},
                "cultural_modifiers": {}
            },
            "stop": {
                "appropriate_responses": ["acknowledgment", "stillness"],
                "social_rules": {"compliance": True, "patience": True},
                "cultural_modifiers": {}
            }
        }
    
    def generate_response(self, human_gesture, cultural_background="default", relationship="stranger"):
        """
        Generate appropriate robot response to human gesture
        """
        if human_gesture not in self.response_mapping:
            return "acknowledgment"  # Default response
        
        response_info = self.response_mapping[human_gesture]
        
        # Apply cultural modifiers
        if cultural_background in response_info.get("cultural_modifiers", {}):
            cultural_mods = response_info["cultural_modifiers"][cultural_background]
            if "handshake_followup" in cultural_mods:
                # Add handshake after acknowledgment
                return ["acknowledgment", "handshake"]
            elif "bow_instead_of_wave" in cultural_mods:
                return ["greeting_bow"]
        
        # Select primary response based on relationship
        if relationship == "close_acquaintance":
            primary_response = response_info["appropriate_responses"][0]  # Usually more enthusiastic
        else:
            primary_response = response_info["appropriate_responses"][0]  # Standard response
        
        return primary_response
```

## Conversational AI and Natural Language Interaction

### Social Dialogue Management

```python
class SocialDialogueManager:
    def __init__(self):
        self.conversation_state = {
            'topic_history': [],
            'entity_memory': {},
            'user_personality': {},
            'relationship_level': 'stranger',
            'current_intent': 'greeting',
            'turn_count': 0
        }
        
        self.dialogue_policy = SocialDialoguePolicy()
        self.language_generator = SocialLanguageGenerator()
        self.implicit_meaning_interpreter = ImplicitMeaningInterpreter()
        
    def process_user_input(self, user_input, user_context=None):
        """
        Process user input and generate appropriate social response
        """
        # Classify user intent
        intent = self.classify_intent(user_input)
        
        # Extract entities and update memory
        entities = self.extract_entities(user_input)
        self.update_entity_memory(entities, user_input)
        
        # Infer implicit meaning
        implicit_meaning = self.implicit_meaning_interpreter.interpret(user_input)
        
        # Update conversation state
        self.conversation_state['current_intent'] = intent
        self.conversation_state['turn_count'] += 1
        self.conversation_state['topic_history'].append({
            'speaker': 'user',
            'text': user_input,
            'intent': intent,
            'entities': entities,
            'implicit_meaning': implicit_meaning,
            'timestamp': time.time()
        })
        
        # Generate response based on state and policy
        response = self.generate_response(intent, entities, user_context)
        
        # Update conversation state with our response
        self.conversation_state['topic_history'].append({
            'speaker': 'robot',
            'text': response,
            'timestamp': time.time()
        })
        
        return response
    
    def classify_intent(self, text):
        """
        Classify the intent behind user input
        """
        # Use rule-based classification as a starting point
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
            return "greeting"
        elif any(word in text_lower for word in ["how are you", "how do you do", "what's up", "how's it going"]):
            return "wellbeing_inquiry"
        elif any(word in text_lower for word in ["bye", "goodbye", "see you", "farewell", "take care"]):
            return "farewell"
        elif any(word in text_lower for word in ["what", "how", "when", "where", "who", "why", "can you", "could you"]):
            return "information_request"
        elif any(word in text_lower for word in ["please", "thank you", "thanks", "appreciate", "grateful"]):
            return "politeness"
        elif any(word in text_lower for word in ["yes", "yeah", "yep", "sure", "ok", "okay", "alright"]):
            return "acknowledgment_affirmation"
        elif any(word in text_lower for word in ["no", "nope", "nah", "not", "never"]):
            return "acknowledgment_negation"
        elif any(word in text_lower for word in ["story", "tell me", "about", "happen", "experience"]):
            return "narrative_request"
        elif any(word in text_lower for word in ["help", "assist", "aid", "support"]):
            return "assistance_request"
        else:
            return "general_conversation"
    
    def extract_entities(self, text):
        """
        Extract named entities from text (simplified version)
        """
        # In a real implementation, this would use NLP libraries like spaCy or transformers
        # For this example, we'll do simple pattern matching
        
        import re
        
        # Common entity patterns
        patterns = {
            'person': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Names
            'location': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Places (would have better detection in reality)
            'time': r'\b(?:today|tomorrow|yesterday|\d{1,2}(?::\d{2})?(?:\s*(?:am|pm))?)\b',
            'object': r'\b(?:robot|person|computer|table|chair|book|phone|car|house|tree)\b'
        }
        
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Add the text itself as a potential entity for topic discussion
        entities['topic'] = [text[:50]]  # First 50 chars as topic placeholder
        
        return entities
    
    def update_entity_memory(self, entities, text):
        """
        Update robot's memory of entities mentioned
        """
        for entity_type, entity_list in entities.items():
            if entity_type not in self.conversation_state['entity_memory']:
                self.conversation_state['entity_memory'][entity_type] = []
            
            for entity in entity_list:
                if entity not in self.conversation_state['entity_memory'][entity_type]:
                    self.conversation_state['entity_memory'][entity_type].append(entity)
    
    def generate_response(self, intent, entities, user_context):
        """
        Generate appropriate response based on intent, entities, and context
        """
        # Select response strategy based on intent and context
        if intent == "greeting":
            return self.generate_greeting_response(user_context)
        elif intent == "wellbeing_inquiry":
            return self.generate_wellbeing_response()
        elif intent == "farewell":
            return self.generate_farewell_response()
        elif intent == "information_request":
            return self.generate_information_response(entities, user_context)
        elif intent == "assistance_request":
            return self.generate_assistance_response(entities)
        elif intent == "narrative_request":
            return self.generate_narrative_response(entities)
        else:
            return self.generate_general_response(entities, user_context)
    
    def generate_greeting_response(self, user_context=None):
        """
        Generate an appropriate greeting
        """
        import random
        
        # Get time-based greeting
        current_hour = time.localtime().tm_hour
        if 5 <= current_hour < 12:
            time_greeting = "Good morning"
        elif 12 <= current_hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= current_hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Hello"
        
        # Get user-appropriate greeting
        if user_context and user_context.get('name'):
            name = user_context['name']
            greeting_patterns = [
                f"{time_greeting}, {name}! It's wonderful to see you again.",
                f"Hello {name}! How can I assist you today?",
                f"Greetings, {name}! I hope you're having a pleasant day."
            ]
        else:
            greeting_patterns = [
                f"{time_greeting}! I'm your friendly humanoid assistant.",
                f"Hello there! I'm happy to meet you.",
                f"Good day! I'm here to help with whatever you need."
            ]
        
        return random.choice(greeting_patterns)
    
    def generate_wellbeing_response(self):
        """
        Generate response to wellbeing inquiries
        """
        import random
        
        responses = [
            "I'm doing well, thank you for asking! My systems are all functioning optimally.",
            "I'm quite good, thank you! Always excited to learn and interact with humans.",
            "All systems running smoothly here! How are you doing today?",
            "I'm in good spirits! Though as a robot, I don't experience wellbeing like humans do."
        ]
        
        return random.choice(responses)
    
    def generate_farewell_response(self):
        """
        Generate appropriate farewell
        """
        import random
        
        farewells = [
            "Goodbye! I hope we can chat again soon.",
            "Take care and have a wonderful day!",
            "Until next time! It was great talking with you.",
            "Farewell! Remember, I'm always here if you need assistance."
        ]
        
        return random.choice(farewells)
    
    def generate_information_response(self, entities, user_context):
        """
        Generate informative response to questions
        """
        # In a real system, this would query a knowledge base
        # For this example, provide generic informative responses
        
        topic = entities.get('topic', ['general'])[0] if entities.get('topic') else 'general'
        
        if any(keyword in topic.lower() for keyword in ["robot", "humanoid", "technology"]):
            return ("I'm a humanoid robot designed to assist and interact with humans. " +
                   "I can help with various tasks, answer questions, and engage in conversations. " +
                   "My systems include sensors for perception and actuators for movement.")
        
        elif any(keyword in topic.lower() for keyword in ["name", "call you"]):
            return "I'm called SocialBot, though you can give me a nickname if you like!"
        
        else:
            return ("I'd be happy to discuss that topic with you! What specifically would you like to know? " +
                   "While I can provide information on many subjects, I'm particularly good at topics related to robotics, technology, and social interaction.")

class SocialDialoguePolicy:
    """
    Policy for managing turn-taking and social rules in conversation
    """
    def __init__(self):
        self.turn_taking_rules = {
            'pause_duration': (0.5, 1.5),  # Pause before responding
            'interrupt_handling': 'polite_wait',
            'attention_shift': 'acknowledge_then_redirect',
            'topic_transition': 'graceful_with_context'
        }
        
        self.social_norms = {
            'greeting_return': True,
            'politeness_reciprocity': True,
            'personal_space_respect': True,
            'cultural_sensitivity': True
        }
    
    def manage_turn_taking(self, user_finished_speaking, robot_has_response):
        """
        Manage turn transitions in conversation
        """
        if user_finished_speaking:
            # Wait appropriate pause time before responding
            pause_duration = random.uniform(
                self.turn_taking_rules['pause_duration'][0],
                self.turn_taking_rules['pause_duration'][1]
            )
            time.sleep(pause_duration)
            return True  # Robot should speak
        else:
            return False  # Wait for user to finish

class SocialLanguageGenerator:
    """
    Generate natural, contextually appropriate language
    """
    def __init__(self):
        self.lexicon = self.build_social_lexicon()
        self.language_templates = self.define_language_templates()
    
    def build_social_lexicon(self):
        """
        Build lexicon of socially appropriate language
        """
        return {
            'greetings': ['hello', 'hi', 'greetings', 'good day', 'how do you do'],
            'politeness_markers': ['please', 'thank you', 'excuse me', 'pardon', 'you\'re welcome'],
            'response_hedges': ['well', 'actually', 'umm', 'let me see', 'that\'s interesting'],
            'acknowledgments': ['yes', 'right', 'correct', 'indeed', 'exactly'],
            'social_bridges': ['so', 'well', 'anyway', 'now', 'then']
        }
    
    def define_language_templates(self):
        """
        Define templates for different types of responses
        """
        return {
            'greeting': [
                "{time_greeting}, {user_name}! How can I assist you today?",
                "Hello {user_name}! I'm delighted to meet you.",
                "Greetings! {user_name}, what brings you to talk with me?"
            ],
            'information_request': [
                "I'd be happy to explain {topic} in more detail.",
                "About {topic}: {information}",
                "That's an interesting question about {topic}. {explanation}"
            ],
            'error_handling': [
                "I apologize, I didn't quite understand that. Could you rephrase?",
                "I'm sorry, that's outside my knowledge base. Can I help with something else?",
                "Could you clarify what you mean by '{unclear_term}'?"
            ]
        }
    
    def generate_contextual_response(self, intent, entities, context):
        """
        Generate response that fits the social context
        """
        # This would use the templates and lexicon to create appropriate responses
        # For brevity, we'll return basic contextual strings
        
        if intent == "greeting" and context.get("is_return_greeting"):
            return "Hello again! It's good to see you."
        elif intent == "wellbeing_inquiry":
            return "I'm functioning well, thank you for asking!"
        else:
            return "I'd be happy to continue our conversation."

class ImplicitMeaningInterpreter:
    """
    Interpret the implicit meaning behind user utterances
    """
    def __init__(self):
        self.implicature_patterns = {
            'indirect_requests': {
                'pattern': r"can you.*(\.|\?)$",
                'interpretation': 'assistance_request'
            },
            'social_bonding': {
                'pattern': r"how was your.*day|what did you.*today",
                'interpretation': 'social_connection_attempt'
            },
            'sarcasm_detection': {
                'pattern': r"(oh really|sure thing|of course).*(\.|\?)$",
                'interpretation': 'potentially_sarcastic'
            }
        }
    
    def interpret(self, text):
        """
        Interpret implicit meanings in user text
        """
        import re
        interpretations = []
        
        for pattern_name, pattern_info in self.implicature_patterns.items():
            if re.search(pattern_info['pattern'], text.lower()):
                interpretations.append({
                    'type': pattern_name,
                    'interpretation': pattern_info['interpretation'],
                    'confidence': 0.8
                })
        
        # Add general sentiment if no specific implicatures found
        if not interpretations:
            # Simple sentiment analysis
            sentiment = self.basic_sentiment_analysis(text)
            interpretations.append({
                'type': 'sentiment',
                'interpretation': sentiment,
                'confidence': 0.6
            })
        
        return interpretations
    
    def basic_sentiment_analysis(self, text):
        """
        Very basic sentiment analysis for example purposes
        """
        positive_words = ["good", "great", "excellent", "wonderful", "fantastic", "awesome", "amazing"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "stupid"]
        
        pos_count = sum(1 for word in positive_words if word.lower() in text.lower())
        neg_count = sum(1 for word in negative_words if word.lower() in text.lower())
        
        if pos_count > neg_count:
            return "positive_sentiment"
        elif neg_count > pos_count:
            return "negative_sentiment"
        else:
            return "neutral_sentiment"
```

## Emotional Intelligence and Expression

### Emotion Recognition and Response

```python
class EmotionalIntelligenceSystem:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.emotion_regulator = EmotionRegulator()
        self.emotional_expression_controller = EmotionalExpressionController()
        
        # Emotional state tracking
        self.robot_emotional_state = {
            'current_emotion': 'neutral',
            'intensity': 0.0,
            'duration': 0.0,
            'trigger': None
        }
        
        # Emotional memory
        self.emotional_interactions = []
    
    def analyze_user_emotion(self, user_data):
        """
        Analyze user's emotional state from multiple modalities
        
        Args:
            user_data: Dictionary containing face expression, voice tone, 
                      physiological signals, and behavioral cues
                      
        Returns:
            Dictionary with detected emotions and confidence scores
        """
        emotions = {}
        
        # Analyze facial expressions
        if 'face_expression' in user_data:
            face_emotions = self.emotion_detector.analyze_facial_expression(
                user_data['face_expression']
            )
            emotions.update(face_emotions)
        
        # Analyze voice prosody
        if 'voice_data' in user_data:
            voice_emotions = self.emotion_detector.analyze_voice_emotion(
                user_data['voice_data']
            )
            for emo, score in voice_emotions.items():
                if emo in emotions:
                    # Combine scores with weighted average
                    emotions[emo] = 0.7 * emotions[emo] + 0.3 * score
                else:
                    emotions[emo] = score
        
        # Analyze behavioral patterns
        if 'behavioral_data' in user_data:
            behavior_emotions = self.emotion_detector.analyze_behavioral_emotion(
                user_data['behavioral_data']
            )
            for emo, score in behavior_emotions.items():
                if emo in emotions:
                    emotions[emo] = max(emotions[emo], score)  # Take maximum
                else:
                    emotions[emo] = score
        
        return emotions
    
    def respond_to_user_emotion(self, user_emotions, interaction_context):
        """
        Generate appropriate emotional response to user's emotional state
        """
        # Determine dominant user emotion
        if not user_emotions:
            return self.maintain_neutral_response()
        
        dominant_emotion = max(user_emotions, key=user_emotions.get)
        emotion_intensity = user_emotions[dominant_emotion]
        
        # Set robot's emotional response
        response_emotion = self.determine_response_emotion(
            dominant_emotion, emotion_intensity, interaction_context
        )
        
        # Generate expressive behavior
        response_behavior = self.emotional_expression_controller.generate_expression(
            response_emotion, emotion_intensity
        )
        
        # Update internal state
        self.update_emotional_state(response_emotion, emotion_intensity)
        
        return response_behavior
    
    def determine_response_emotion(self, user_emotion, intensity, context):
        """
        Determine appropriate emotional response based on user emotion and context
        """
        # Empathy-based responses
        empathy_map = {
            'happy': 'joyful',
            'sad': 'compassionate', 
            'angry': 'concerned',
            'afraid': 'reassuring',
            'surprised': 'curious',
            'disgusted': 'neutral'
        }
        
        # Context-sensitive adjustments
        if context.get('relationship', 'stranger') == 'close_friend':
            responses = {
                'happy': 'joyful',
                'sad': 'empathetic',
                'angry': 'supportive',
                'afraid': 'protective'
            }
        else:
            responses = empathy_map
        
        return responses.get(user_emotion, 'neutral')
    
    def update_emotional_state(self, emotion, intensity):
        """
        Update robot's emotional state
        """
        self.robot_emotional_state = {
            'current_emotion': emotion,
            'intensity': intensity,
            'duration': time.time(),
            'trigger': 'user_interaction'
        }
        
        # Add to emotional memory
        self.emotional_interactions.append({
            'emotion': emotion,
            'intensity': intensity,
            'timestamp': time.time(),
            'trigger_source': 'user_interaction'
        })
        
        # Maintain only recent emotional interactions (last 10)
        if len(self.emotional_interactions) > 10:
            self.emotional_interactions = self.emotional_interactions[-10:]

class EmotionDetector:
    """
    Detect emotions from multiple modalities
    """
    def __init__(self):
        # In a real system, these would be ML models
        self.facial_expression_model = self.load_model('facial_expression')
        self.voice_emotion_model = self.load_model('voice_emotion')
        self.behavioral_emotion_model = self.load_model('behavioral_emotion')
    
    def load_model(self, model_type):
        """
        Load emotion detection model (placeholder)
        """
        return f"Loaded {model_type} model"
    
    def analyze_facial_expression(self, face_data):
        """
        Analyze emotions from facial expressions
        """
        # In a real implementation, this would process facial landmarks/features
        # and classify emotions using a trained model
        
        # For this example, return mock emotions based on simplified analysis
        import random
        
        emotions = {
            'happy': random.uniform(0.0, 1.0) * 0.3,  # Base happiness
            'sad': random.uniform(0.0, 1.0) * 0.2,
            'angry': random.uniform(0.0, 1.0) * 0.1,
            'surprised': random.uniform(0.0, 1.0) * 0.1,
            'fearful': random.uniform(0.0, 1.0) * 0.1,
            'disgusted': random.uniform(0.0, 1.0) * 0.1
        }
        
        # Boost one emotion based on facial features
        # This would be more sophisticated with real facial data
        dominant_emotion = random.choice(list(emotions.keys()))
        emotions[dominant_emotion] = min(1.0, emotions[dominant_emotion] + 0.4)
        
        return emotions
    
    def analyze_voice_emotion(self, voice_data):
        """
        Analyze emotions from vocal prosody
        """
        # Analyze pitch, rhythm, intensity, etc.
        import random
        
        # Simulate analysis based on voice features
        pitch_variation = voice_data.get('pitch_variance', 0.5)
        speaking_rate = voice_data.get('rate', 150)  # words per minute
        
        emotions = {}
        
        # Higher pitch variance often indicates excitement/happiness
        emotions['happy'] = min(1.0, pitch_variation * 2.0)
        
        # Faster speaking rate can indicate anxiety/anger
        rate_excitement = max(0, (speaking_rate - 120) / 80.0)
        emotions['excited'] = min(1.0, rate_excitement)
        emotions['angry'] = min(0.8, rate_excitement * 0.7)
        
        # Lower pitch can indicate sadness
        avg_pitch = voice_data.get('avg_pitch', 150)
        sadness_factor = max(0, (100 - avg_pitch) / 50.0)
        emotions['sad'] = min(1.0, sadness_factor)
        
        return emotions
    
    def analyze_behavioral_emotion(self, behavior_data):
        """
        Analyze emotions from behavioral patterns
        """
        # Analyze movement patterns, posture, gesture frequency, etc.
        import random
        
        emotions = {}
        
        # Rapid movements might indicate excitement/anger
        movement_intensity = behavior_data.get('movement_intensity', 0.5)
        emotions['excited'] = movement_intensity
        emotions['angry'] = movement_intensity * 0.6
        
        # Leaning forward might indicate interest/engagement
        forward_lean = behavior_data.get('forward_lean', 0.0)
        emotions['interested'] = min(1.0, forward_lean * 3.0)
        
        # Stillness might indicate calmness or boredom
        stillness = behavior_data.get('stillness_index', 0.3)
        emotions['calm'] = min(1.0, stillness * 2.0)
        
        return emotions

class EmotionRegulator:
    """
    Regulate emotional responses based on social and contextual rules
    """
    def __init__(self):
        self.regulation_rules = {
            'intensity_modulation': self.modulate_intensity,
            'context_alignment': self.align_to_context,
            'social_norms_compliance': self.comply_with_norms,
            'relationship_sensitivity': self.adjust_for_relationship
        }
    
    def regulate_emotional_response(self, detected_emotion, user_emotion, context):
        """
        Regulate emotional response based on multiple factors
        """
        regulated_emotion = detected_emotion.copy()
        
        # Apply regulation rules
        for rule_name, rule_func in self.regulation_rules.items():
            regulated_emotion = rule_func(regulated_emotion, user_emotion, context)
        
        return regulated_emotion
    
    def modulate_intensity(self, emotion, user_emotion, context):
        """
        Adjust emotional intensity based on social appropriateness
        """
        # Don't mirror too intense emotions directly
        for emo, intensity in emotion.items():
            # Reduce intensity if user shows very strong emotion
            user_intensity = user_emotion.get(emo, 0.0)
            if user_intensity > 0.8:
                emotion[emo] = min(intensity, user_intensity * 0.75)
        
        return emotion
    
    def align_to_context(self, emotion, user_emotion, context):
        """
        Align emotional expression with contextual appropriateness
        """
        # In formal contexts, reduce emotional intensity
        if context.get('formality_level', 'casual') == 'formal':
            for emo in emotion:
                emotion[emo] *= 0.6  # Reduce by 40%
        
        # During sad events, don't show happiness
        if context.get('event_type') == 'mourning':
            emotion['happy'] = min(emotion['happy'], 0.2)
            emotion['sad'] = max(emotion['sad'], 0.6)
        
        return emotion

class EmotionalExpressionController:
    """
    Control emotional expression through various modalities
    """
    def __init__(self):
        self.expression_modalities = {
            'facial': self.control_facial_expression,
            'vocal': self.control_vocal_tone,
            'gestural': self.control_gestural_expression,
            'postural': self.control_postural_expression
        }
    
    def generate_expression(self, emotion, intensity):
        """
        Generate emotional expression across modalities
        """
        expression = {}
        
        for modality, control_func in self.expression_modalities.items():
            expression[modality] = control_func(emotion, intensity)
        
        return expression
    
    def control_facial_expression(self, emotion, intensity):
        """
        Generate facial expression parameters
        """
        expression_params = {
            'eyebrow_position': 0.0,
            'eye_openness': 1.0,
            'mouth_shape': 'neutral',
            'jaw_position': 0.0,
            'cheek_raising': 0.0
        }
        
        if emotion == 'happy':
            expression_params.update({
                'eyebrow_position': intensity * 0.3,
                'eye_openness': 1.0 - intensity * 0.1,
                'mouth_shape': 'smile',
                'cheek_raising': intensity * 0.8
            })
        elif emotion == 'sad':
            expression_params.update({
                'eyebrow_position': -intensity * 0.4,
                'eye_openness': 1.0 - intensity * 0.2,
                'mouth_shape': 'frown',
                'jaw_position': -intensity * 0.2
            })
        elif emotion == 'angry':
            expression_params.update({
                'eyebrow_position': -intensity * 0.6,
                'eye_openness': 1.0 + intensity * 0.3,
                'mouth_shape': 'tight',
                'jaw_position': intensity * 0.1
            })
        elif emotion == 'surprised':
            expression_params.update({
                'eyebrow_position': intensity * 0.8,
                'eye_openness': 1.0 + intensity * 0.5,
                'mouth_shape': 'open',
                'jaw_position': intensity * 0.3
            })
        elif emotion == 'fearful':
            expression_params.update({
                'eyebrow_position': intensity * 0.6,
                'eye_openness': 1.0 + intensity * 0.4,
                'mouth_shape': 'tense',
                'cheek_raising': intensity * 0.2
            })
        elif emotion == 'disgusted':
            expression_params.update({
                'eyebrow_position': -intensity * 0.2,
                'eye_openness': 1.0 - intensity * 0.1,
                'mouth_shape': 'grimace',
                'jaw_position': -intensity * 0.1
            })
        
        return expression_params
    
    def control_vocal_tone(self, emotion, intensity):
        """
        Generate vocal expression parameters
        """
        voice_params = {
            'pitch': 1.0,
            'volume': 1.0,
            'speed': 1.0,
            'quality': 'neutral'
        }
        
        if emotion == 'happy':
            voice_params.update({
                'pitch': 1.0 + intensity * 0.2,
                'volume': 1.0 + intensity * 0.1,
                'speed': 1.0 + intensity * 0.15,
                'quality': 'warm'
            })
        elif emotion == 'sad':
            voice_params.update({
                'pitch': 1.0 - intensity * 0.3,
                'volume': 1.0 - intensity * 0.2,
                'speed': 1.0 - intensity * 0.25,
                'quality': 'soothing'
            })
        elif emotion == 'angry':
            voice_params.update({
                'pitch': 1.0 - intensity * 0.1,
                'volume': 1.0 + intensity * 0.4,
                'speed': 1.0 + intensity * 0.3,
                'quality': 'firm'
            })
        elif emotion == 'fearful':
            voice_params.update({
                'pitch': 1.0 + intensity * 0.4,
                'volume': 1.0 - intensity * 0.1,
                'speed': 1.0 + intensity * 0.2,
                'quality': 'tremulous'
            })
        elif emotion == 'surprised':
            voice_params.update({
                'pitch': 1.0 + intensity * 0.5,
                'volume': 1.0 + intensity * 0.2,
                'speed': 1.0 + intensity * 0.1,
                'quality': 'sharp'
            })
        
        return voice_params
    
    def control_gestural_expression(self, emotion, intensity):
        """
        Generate gestural expression parameters
        """
        gesture_params = {
            'amplitude': 0.5,
            'frequency': 1.0,
            'symmetry': 0.8,
            'rhythm': 'smooth'
        }
        
        if emotion == 'happy':
            gesture_params.update({
                'amplitude': 0.5 + intensity * 0.4,
                'frequency': 1.0 + intensity * 0.3,
                'rhythm': 'bouncy'
            })
        elif emotion == 'sad':
            gesture_params.update({
                'amplitude': 0.5 - intensity * 0.3,
                'frequency': 1.0 - intensity * 0.4,
                'rhythm': 'slow'
            })
        elif emotion == 'angry':
            gesture_params.update({
                'amplitude': 0.5 + intensity * 0.5,
                'frequency': 1.0 + intensity * 0.6,
                'rhythm': 'jerky'
            })
        elif emotion == 'excited':
            gesture_params.update({
                'amplitude': 0.5 + intensity * 0.6,
                'frequency': 1.0 + intensity * 0.5,
                'rhythm': 'rapid'
            })
        
        return gesture_params
    
    def control_postural_expression(self, emotion, intensity):
        """
        Generate postural expression parameters
        """
        posture_params = {
            'head_tilt': 0.0,
            'shoulder_position': 0.0,
            'chest_raised': 0.0,
            'posture_tension': 0.5
        }
        
        if emotion == 'happy':
            posture_params.update({
                'head_tilt': 0.1,
                'shoulder_position': 0.1,
                'chest_raised': 0.3 * intensity,
                'posture_tension': 0.4
            })
        elif emotion == 'sad':
            posture_params.update({
                'head_tilt': -0.3 * intensity,
                'shoulder_position': -0.4 * intensity,
                'chest_raised': -0.2 * intensity,
                'posture_tension': 0.3
            })
        elif emotion == 'confident':
            posture_params.update({
                'head_tilt': 0.1,
                'shoulder_position': 0.2,
                'chest_raised': 0.4 * intensity,
                'posture_tension': 0.6
            })
        elif emotion == 'submissive':
            posture_params.update({
                'head_tilt': -0.2,
                'shoulder_position': -0.1,
                'chest_raised': -0.1,
                'posture_tension': 0.3
            })
        
        return posture_params
```

## Cultural Adaptation in HRI

### Cultural Sensitivity and Adaptation

```python
class CulturalAdapter:
    """
    Adapt robot behavior to different cultural contexts
    """
    def __init__(self):
        self.cultural_models = self.load_cultural_models()
        self.user_cultural_profile = {}
        self.cultural_learning_engine = CulturalLearningEngine()
    
    def load_cultural_models(self):
        """
        Load cultural behavior models based on Hofstede's dimensions and other frameworks
        """
        return {
            'japanese': {
                'power_distance': 0.7,      # High respect for authority
                'individualism': 0.2,       # Collectivist culture
                'masculinity': 0.5,        # Moderate emphasis on achievement
                'uncertainty_avoidance': 0.8,  # High avoidance of uncertainty
                'long_term_orientation': 0.9,  # Very long-term oriented
                'indulgence': 0.2,         # Restrained society
                
                # Specific behavioral adaptations
                'bowing_preferred': True,
                'direct_eye_contact': False,  # Less direct eye contact
                'personal_space': 1.0,      # Larger personal space
                'formality_level': 'high',
                'touch_aversion': True,
                'gift_giving_rituals': True,
                'face_saving': Very important
            },
            'norwegian': {
                'power_distance': 0.2,      # Low power distance
                'individualism': 0.8,       # Highly individualist
                'masculinity': 0.4,        # Moderate masculinity
                'uncertainty_avoidance': 0.5,  # Moderate uncertainty avoidance
                'long_term_orientation': 0.4,  # Short-term oriented
                'indulgence': 0.7,         # More indulgent society
                
                # Specific behavioral adaptations
                'bowing_preferred': False,
                'direct_eye_contact': True,  # Direct eye contact expected
                'personal_space': 0.8,      # Moderate personal space
                'formality_level': 'low',
                'touch_acceptance': True,
                'egalitarian_interaction': True,
                'informal_address': Preferred
            },
            'saudi_arabian': {
                'power_distance': 0.8,      # Very hierarchical
                'individualism': 0.3,       # Somewhat collectivist
                'masculinity': 0.7,        # Achievement-oriented
                'uncertainty_avoidance': 0.9,  # Very high uncertainty avoidance
                'long_term_orientation': 0.5,  # Moderate long-term orientation
                'indulgence': 0.1,         # Very restrained society
                
                # Specific behavioral adaptations
                'gender_interaction_norms': True,  # Different norms for male/female interaction
                'formal_address': Required,
                'head_nod_acceptable': True,
                'handshake_protocol': Important,
                'religious_sensitivity': Critical,
                'hospitality_important': True
            }
        }
    
    def adapt_behavior_to_culture(self, user_culture, interaction_context):
        """
        Adapt robotic behavior based on user's cultural background
        """
        if user_culture not in self.cultural_models:
            # Default to Western cultural model if unknown
            user_culture = 'western_generic'
        
        cultural_profile = self.cultural_models[user_culture]
        adapted_behavior = {}
        
        # Adapt greeting behavior
        adapted_behavior['greeting'] = self.adapt_greeting(cultural_profile)
        
        # Adapt personal space
        adapted_behavior['personal_distance'] = self.adapt_personal_space(cultural_profile)
        
        # Adapt communication style
        adapted_behavior['communication_style'] = self.adapt_communication_style(cultural_profile)
        
        # Adapt physical interaction
        adapted_behavior['physical_interaction'] = self.adapt_physical_interaction(cultural_profile)
        
        # Adapt emotional expression
        adapted_behavior['emotional_expression'] = self.adapt_emotional_expression(cultural_profile)
        
        return adapted_behavior
    
    def adapt_greeting(self, cultural_profile):
        """
        Adapt greeting based on cultural norms
        """
        if cultural_profile.get('bowing_preferred', False):
            return {
                'type': 'bow',
                'angle': 15 if cultural_profile.get('formality_level') == 'high' else 30,
                'duration': 2.0,
                'accompanying_gesture': 'hands_together'
            }
        elif cultural_profile.get('handshake_protocol'):
            return {
                'type': 'handshake',
                'firmness': 'gentle',
                'duration': 3.0,
                'eye_contact': cultural_profile.get('direct_eye_contact', True)
            }
        else:
            return {
                'type': 'wave',
                'distance': 'arm_length',
                'formality': cultural_profile.get('formality_level', 'medium')
            }
    
    def adapt_personal_space(self, cultural_profile):
        """
        Adapt required personal space based on culture
        """
        base_distance = 0.8  # Default meter distance
        cultural_factor = cultural_profile.get('personal_space', 1.0)
        
        return base_distance * cultural_factor
    
    def adapt_communication_style(self, cultural_profile):
        """
        Adapt communication style based on cultural dimensions
        """
        style = {}
        
        # Adapt to power distance
        if cultural_profile['power_distance'] > 0.6:
            style['formality_level'] = 'high'
            style['deference_level'] = 'high'
            style['title_usage'] = True
        else:
            style['formality_level'] = 'low'
            style['deference_level'] = 'low'
            style['title_usage'] = False
        
        # Adapt to uncertainty avoidance
        if cultural_profile['uncertainty_avoidance'] > 0.7:
            style['structure_preference'] = True
            style['clear_instructions'] = True
            style['rule_following'] = Emphasized
        else:
            style['structure_preference'] = False
            style['flexibility'] = Valued
        
        # Adapt to communication directness
        if cultural_profile['individualism'] < 0.5:  # Collectivist
            style['indirect_communication'] = Preferred
            style['group_focus'] = True
            style['face_saving'] = Important
        else:
            style['direct_communication'] = Preferred
            style['individual_focus'] = True
        
        return style
    
    def adapt_physical_interaction(self, cultural_profile):
        """
        Adapt physical interaction based on cultural norms
        """
        interaction = {}
        
        if cultural_profile.get('touch_aversion', False):
            interaction['physical_contact'] = 'minimal'
            interaction['handshake_preference'] = 'light'
            interaction['personal_space_increase'] = 0.2  # 20cm extra distance
        else:
            interaction['physical_contact'] = 'normal'
            interaction['handshake_preference'] = 'firm'
            interaction['personal_space_increase'] = 0.0
        
        # Gender interaction considerations
        if cultural_profile.get('gender_interaction_norms', False):
            interaction['gender_sensitive_interaction'] = True
            interaction['separate_greeting_protocols'] = True
        
        return interaction
    
    def adapt_emotional_expression(self, cultural_profile):
        """
        Adapt emotional expression based on cultural values
        """
        expression = {}
        
        if cultural_profile.get('indulgence', 0.5) < 0.5:
            # Restrained society - moderate emotional expression
            expression['expression_intensity'] = 0.6  # Less intense emotions
            expression['joy_expressions'] = 'controlled'
            expression['anger_expressions'] = 'suppressed'
        else:
            # Indulgent society - more open emotional expression
            expression['expression_intensity'] = 0.8  # More intense emotions
            expression['joy_expressions'] = 'outward'
            expression['anger_expressions'] = 'expressed'
        
        # Long-term orientation considerations
        if cultural_profile.get('long_term_orientation', 0.5) > 0.7:
            expression['practical_focus'] = True
            expression['achievement_based_emotions'] = Emphasized
            expression['patient_emotions'] = Valued
        else:
            expression['adventure_focus'] = True
            expression['immediate_gratification'] = Considered
        
        return expression

class CulturalLearningEngine:
    """
    Learn and adapt cultural preferences over time through interaction
    """
    def __init__(self):
        self.cultural_preference_model = {}
        self.interaction_history = []
        self.cultural_calibration_updates = 0
    
    def update_cultural_model(self, user_id, interaction_outcome):
        """
        Update cultural model based on interaction outcomes
        """
        user_history = [ih for ih in self.interaction_history if ih['user_id'] == user_id]
        
        # Analyze interaction success/failure patterns
        successful_interactions = [
            ih for ih in user_history 
            if ih.get('outcome', {}).get('satisfaction', 0) > 0.7
        ]
        
        if len(successful_interactions) > 5:  # Sufficient data
            # Update cultural preferences based on successful patterns
            self.calibrate_cultural_preferences(user_id, successful_interactions)
            self.cultural_calibration_updates += 1
    
    def calibrate_cultural_preferences(self, user_id, successful_interactions):
        """
        Calibrate cultural preferences based on successful interactions
        """
        # Analyze what behaviors were most successful
        successful_behaviors = {}
        
        for interaction in successful_interactions:
            behavior = interaction.get('robot_behavior', {})
            outcome = interaction.get('outcome', {})
            
            for behavior_type, value in behavior.items():
                if behavior_type not in successful_behaviors:
                    successful_behaviors[behavior_type] = []
                
                # Weight by satisfaction level
                satisfaction = outcome.get('satisfaction', 0.5)
                successful_behaviors[behavior_type].append((value, satisfaction))
        
        # Update user's cultural profile based on successful behaviors
        if user_id not in self.cultural_preference_model:
            self.cultural_preference_model[user_id] = {}
        
        for behavior_type, value_satisfaction_pairs in successful_behaviors.items():
            # Calculate weighted average of successful values
            total_value = 0
            total_weight = 0
            
            for value, satisfaction in value_satisfaction_pairs:
                total_value += value * satisfaction
                total_weight += satisfaction
            
            if total_weight > 0:
                calibrated_value = total_value / total_weight
                self.cultural_preference_model[user_id][behavior_type] = calibrated_value
```

## Hands-on Exercise

1. Design a social interaction system for a humanoid robot that can recognize when a human is paying attention and respond appropriately with gaze and gestures.

2. Implement a cultural adaptation module that adjusts the robot's behavior based on the detected cultural background of the user.

3. Using the educational AI agents, explore how different personality traits could be incorporated into the humanoid robot's interaction style to make it more relatable to different types of users.

The next section will explore advanced topics in human-robot collaboration and teaming, building on the social interaction foundations established in this section.