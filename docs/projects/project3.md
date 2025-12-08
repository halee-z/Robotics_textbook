---
sidebar_position: 3
---

# Project 3: Socially Assistive Robot for Elderly Care

## Overview

This project focuses on developing a socially assistive humanoid robot designed to provide companionship, cognitive stimulation, and basic assistance to elderly individuals. The robot will engage in conversations, remind users about medications, suggest activities, and monitor for signs of distress or health issues, all while maintaining a friendly and approachable demeanor.

## Learning Objectives

- Integrate multiple AI systems (conversational AI, computer vision, behavioral analysis)
- Implement long-term human-robot interaction strategies
- Create adaptive systems that learn user preferences over time
- Combine physical assistance with social interaction
- Address ethical and privacy considerations in care robotics

## Prerequisites

- Understanding of HRI principles
- Knowledge of conversational AI and NLP
- Familiarity with computer vision and perception
- Basic understanding of healthcare and aging needs

## Implementation Steps

### Step 1: System Architecture Design

```python
import datetime
import json
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

class InteractionMode(Enum):
    COMPANION = "companion"
    ASSISTANT = "assistant"
    CARE_MONITOR = "care_monitor"

class HealthAlert(Enum):
    NORMAL = "normal"
    LOW_ACTIVITY = "low_activity"
    UNUSUAL_BEHAVIOR = "unusual_behavior"
    EMERGENCY = "emergency"

@dataclass
class UserPreference:
    """Store user preferences for interaction"""
    favorite_topics: List[str]
    preferred_interaction_mode: InteractionMode
    daily_routine_times: Dict[str, datetime.time]  # wake, meal, activity times
    privacy_settings: Dict[str, bool]  # what data to share
    communication_style: str  # formal, casual, etc.

@dataclass
class HealthMetrics:
    """Track user health indicators"""
    activity_level: float  # 0.0 to 1.0
    sleep_quality: float  # 0.0 to 1.0
    social_engagement: float  # 0.0 to 1.0
    mood_indicators: List[str]  # happy, sad, anxious, etc.
    medication_adherence: float  # 0.0 to 1.0
    health_alerts: List[Tuple[HealthAlert, datetime.datetime]]

class SocialCareSystem:
    def __init__(self, robot_name: str = "CareBot"):
        self.robot_name = robot_name
        self.current_user: Optional[str] = None
        self.user_model: Optional[UserPreference] = None
        self.health_tracker = HealthMetrics(
            activity_level=0.5,
            sleep_quality=0.5,
            social_engagement=0.5,
            mood_indicators=[],
            medication_adherence=0.5,
            health_alerts=[]
        )
        self.interaction_history = []
        self.daily_goals = self.initialize_daily_goals()
        
        # System state
        self.current_mode = InteractionMode.COMPANION
        self.time_of_day = self.get_time_of_day()
        
    def initialize_daily_goals(self):
        """Initialize daily goals based on user preferences and time"""
        return {
            "interaction_time": 30,  # minutes of interaction
            "physical_activity": 15,  # minutes of suggested activity
            "social_connection": 1,  # number of social interactions
            "cognitive_stimulation": 2  # number of cognitive tasks
        }
    
    def get_time_of_day(self):
        """Determine time of day for appropriate interaction"""
        current_hour = datetime.datetime.now().hour
        if 5 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 17:
            return "afternoon"
        elif 17 <= current_hour < 21:
            return "evening"
        else:
            return "night"
    
    def load_user_profile(self, user_id: str):
        """Load user-specific preferences and history"""
        # In a real implementation, this would load from a database
        self.current_user = user_id
        
        # Default preferences for new user
        self.user_model = UserPreference(
            favorite_topics=["family", "garden", "cooking", "music"],
            preferred_interaction_mode=InteractionMode.COMPANION,
            daily_routine_times={
                "wake": datetime.time(7, 0),
                "breakfast": datetime.time(8, 0),
                "lunch": datetime.time(12, 0),
                "dinner": datetime.time(18, 0),
                "bedtime": datetime.time(22, 0)
            },
            privacy_settings={
                "health_data": True,
                "location": True,
                "conversations": False
            },
            communication_style="warm"
        )
        
        return self.user_model
    
    def update_interaction_history(self, interaction_type: str, content: str, response: str):
        """Record interaction for learning and adaptation"""
        interaction = {
            "timestamp": datetime.datetime.now(),
            "type": interaction_type,
            "content": content,
            "response": response,
            "mode": self.current_mode.value,
            "user_reaction": "positive"  # Would be determined in real system
        }
        self.interaction_history.append(interaction)
        
        # Keep only recent interactions (last 100)
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
```

### Step 2: Context-Aware Interaction System

```python
class ContextAwareInteraction:
    def __init__(self, care_system: SocialCareSystem):
        self.care_system = care_system
        self.conversation_manager = ConversationManager()
        self.activity_manager = ActivityManager()
        self.reminder_system = ReminderSystem()
        
    def determine_interaction_context(self):
        """Determine appropriate interaction based on multiple factors"""
        context = {
            "time_of_day": self.care_system.time_of_day,
            "user_state": self.assess_user_state(),
            "health_metrics": self.care_system.health_tracker,
            "daily_goals_progress": self.calculate_goals_progress(),
            "user_preferences": self.care_system.user_model
        }
        
        return context
    
    def assess_user_state(self):
        """Assess user's current state from multiple sensors"""
        # In a real system, this would integrate data from:
        # - Camera (facial expression, posture)
        # - Voice analysis (tone, energy)
        # - Activity sensors (movement, engagement)
        # - Calendar/integration (scheduled activities)
        
        # For simulation, return a random state
        import random
        states = ["resting", "active", "tired", "engaged", "withdrawn"]
        moods = ["happy", "neutral", "sad", "anxious", "excited"]
        
        return {
            "activity_level": random.uniform(0.3, 0.9),
            "mood": random.choice(moods),
            "engagement_level": random.uniform(0.2, 0.8),
            "energy_level": random.uniform(0.2, 0.9)
        }
    
    def calculate_goals_progress(self):
        """Calculate progress toward daily goals"""
        # For simplicity, return random progress
        import random
        return {
            "interaction_time": random.uniform(0, 45),  # minutes
            "physical_activity": random.uniform(0, 20),  # minutes
            "social_connection": random.randint(0, 3),  # count
            "cognitive_stimulation": random.randint(0, 5)  # count
        }
    
    def select_interaction(self, context):
        """Select appropriate interaction based on context"""
        # Adjust mode based on user state and needs
        if context["user_state"]["mood"] == "sad":
            self.care_system.current_mode = InteractionMode.COMPANION
        elif datetime.datetime.now().time().hour in [8, 12, 18]:  # meal times
            self.care_system.current_mode = InteractionMode.ASSISTANT
        elif context["health_metrics"].activity_level < 0.3:
            self.care_system.current_mode = InteractionMode.CARE_MONITOR
        
        # Select interaction type based on mode and context
        if self.care_system.current_mode == InteractionMode.COMPANION:
            return self.select_companion_interaction(context)
        elif self.care_system.current_mode == InteractionMode.ASSISTANT:
            return self.select_assistant_interaction(context)
        else:  # CARE_MONITOR
            return self.select_care_monitoring_interaction(context)
    
    def select_companion_interaction(self, context):
        """Select companion-style interaction"""
        user_energy = context["user_state"]["energy_level"]
        user_engagement = context["user_state"]["engagement_level"]
        
        if user_energy > 0.7 and user_engagement > 0.6:
            # High energy, high engagement - suggest activity
            return self.activity_manager.suggest_mental_activity()
        elif user_engagement < 0.4:
            # Low engagement - try to connect
            return self.conversation_manager.start_connection_dialogue(
                context["user_preferences"].favorite_topics
            )
        else:
            # Normal engagement - continue conversation
            return self.conversation_manager.continue_conversation()
    
    def select_assistant_interaction(self, context):
        """Select assistant-style interaction"""
        current_time = datetime.datetime.now().time()
        
        # Check for scheduled reminders
        if current_time.hour == 8:  # Breakfast time
            return self.reminder_system.remind_meal_time("breakfast")
        elif current_time.hour == 12:  # Lunch time
            return self.reminder_system.remind_meal_time("lunch")
        elif current_time.hour == 18:  # Dinner time
            return self.reminder_system.remind_meal_time("dinner")
        else:
            # General assistant interaction
            return self.conversation_manager.assistant_interaction()
    
    def select_care_monitoring_interaction(self, context):
        """Select care monitoring interaction"""
        activity_level = context["health_metrics"].activity_level
        
        if activity_level < 0.3:
            # Low activity - encourage movement
            return self.activity_manager.suggest_light_activity()
        else:
            # Normal activity - continue monitoring
            return self.conversation_manager.check_in_interaction()

class ConversationManager:
    def __init__(self):
        self.conversation_context = []
        self.knowledge_base = self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load conversation starters and topics"""
        return {
            "greetings": [
                "Good morning! How did you sleep?",
                "Hello! How are you feeling today?",
                "Good afternoon! I hope you're having a pleasant day."
            ],
            "family_talk": [
                "Tell me about your family.",
                "What are your grandchildren up to these days?",
                "I'd love to hear about your children."
            ],
            "activities": [
                "Would you like to try a puzzle?",
                "I could read you a story if you'd like.",
                "How about a simple game to keep your mind active?"
            ],
            "health_check": [
                "How are you feeling today?",
                "Are you comfortable?",
                "Is there anything I can get for you?"
            ]
        }
    
    def start_connection_dialogue(self, favorite_topics):
        """Start conversation based on user's favorite topics"""
        import random
        
        # Select from user's favorite topics
        topic = random.choice(favorite_topics) if favorite_topics else "general"
        
        if topic == "family":
            return random.choice(self.knowledge_base["family_talk"])
        elif topic == "music":
            return "Would you like me to play some music? What genre do you enjoy?"
        elif topic == "garden":
            return "I'd love to hear about your garden. What are you growing?"
        elif topic == "cooking":
            return "What's your favorite recipe? I'd enjoy hearing about it."
        else:
            return random.choice(self.knowledge_base["greetings"])
    
    def continue_conversation(self):
        """Continue an ongoing conversation"""
        import random
        return random.choice([
            "That's interesting! Could you tell me more?",
            "I see. How does that make you feel?",
            "That reminds me of something similar..."
        ])
    
    def assistant_interaction(self):
        """Provide assistant-style help"""
        return "Is there anything I can help you with today?"
    
    def check_in_interaction(self):
        """Check on user's wellbeing"""
        import random
        return random.choice(self.knowledge_base["health_check"])
```

### Step 3: Activity and Reminder Systems

```python
class ActivityManager:
    def __init__(self):
        self.mental_activities = {
            "trivia": "Would you like to try a trivia question? It's good for the mind!",
            "word_games": "How about a word puzzle? I can think of a word for you to guess.",
            "memory_games": "Let's play a memory game. I'll name some items, and you repeat them back.",
            "story_telling": "Would you like me to tell you a story, or perhaps you have a story to share?"
        }
        
        self.physical_activities = {
            "arm_exercises": "Let's do some gentle arm exercises. Can you raise your arms slowly?",
            "seated_march": "Let's do some seated marching. Lift your knees alternately.",
            "stretching": "How about some gentle stretching? Move your arms slowly in circles.",
            "breathing": "Let's do some deep breathing exercises. Inhale slowly, then exhale."
        }
    
    def suggest_mental_activity(self):
        """Suggest cognitive stimulation activity"""
        import random
        activity = random.choice(list(self.mental_activities.keys()))
        return self.mental_activities[activity]
    
    def suggest_light_activity(self):
        """Suggest light physical activity"""
        import random
        activity = random.choice(list(self.physical_activities.keys()))
        return self.physical_activities[activity]

class ReminderSystem:
    def __init__(self):
        self.medication_schedule = {}  # Will be populated with user schedule
        self.appointment_schedule = {}
        
    def remind_meal_time(self, meal_type):
        """Remind user about meal time"""
        return f"It's time for {meal_type}! Would you like me to help you get ready?"
    
    def remind_medication(self, medication_name):
        """Remind user to take medication"""
        return f"It's time for your {medication_name}. Would you like me to bring it to you?"
    
    def suggest_social_interaction(self):
        """Suggest social activity"""
        return "Would you like to call a family member or friend today?"
    
    def schedule_reminders(self, user_preferences):
        """Set up daily reminders based on user schedule"""
        # In a real system, this would use a scheduler
        pass
```

### Step 4: Health Monitoring and Adaptation

```python
class HealthMonitoringSystem:
    def __init__(self, care_system: SocialCareSystem):
        self.care_system = care_system
        self.wellness_indicators = {
            "activity": 0.5,
            "mood": 0.5,
            "socialization": 0.5,
            "cognitive_function": 0.5
        }
        
    def update_health_metrics(self):
        """Update health metrics based on interactions and sensor data"""
        # In a real system, this would process sensor data
        # For simulation, update based on interaction patterns
        
        # Update activity level based on physical activities suggested
        physical_activity_count = self.count_recent_interactions("physical_activity")
        self.care_system.health_tracker.activity_level = min(1.0, 
            self.care_system.health_tracker.activity_level + (physical_activity_count * 0.1))
        
        # Update social engagement based on conversation frequency
        social_count = self.count_recent_interactions("conversation")
        self.care_system.health_tracker.social_engagement = min(1.0,
            self.care_system.health_tracker.social_engagement + (social_count * 0.05))
        
        # Update mood based on conversation sentiment (simplified)
        mood_trend = self.analyze_conversation_sentiment()
        if mood_trend == "positive":
            self.care_system.health_tracker.mood_indicators.append("happy")
        elif mood_trend == "negative":
            self.care_system.health_tracker.mood_indicators.append("sad")
        # else neutral, no update
        
        # Check for unusual patterns
        self.check_for_anomalies()
    
    def count_recent_interactions(self, interaction_type, hours=24):
        """Count recent interactions of a specific type"""
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        count = sum(1 for interaction in self.care_system.interaction_history
                   if interaction["type"] == interaction_type and 
                   interaction["timestamp"] > cutoff)
        return count
    
    def analyze_conversation_sentiment(self):
        """Analyze recent conversation sentiment (simplified)"""
        # In a real system, this would use NLP sentiment analysis
        # For simulation, return a random sentiment
        import random
        return random.choice(["positive", "neutral", "negative"])
    
    def check_for_anomalies(self):
        """Check for unusual patterns that might indicate health issues"""
        # Check for unusually low activity
        if self.care_system.health_tracker.activity_level < 0.2:
            self.care_system.health_tracker.health_alerts.append(
                (HealthAlert.LOW_ACTIVITY, datetime.datetime.now())
            )
            self.trigger_low_activity_protocol()
        
        # Check for unusual behavior patterns
        # This would involve more complex pattern recognition in a real system
    
    def trigger_low_activity_protocol(self):
        """Trigger protocol for when activity is unusually low"""
        # Increase check-ins
        self.increase_monitoring_frequency()
        
        # Consider contacting caregivers
        self.assess_need_for_caregiver_contact()
    
    def increase_monitoring_frequency(self):
        """Increase monitoring when health indicators are concerning"""
        # In a real system, this might mean more frequent check-ins
        pass
    
    def assess_need_for_caregiver_contact(self):
        """Determine if caregiver should be contacted"""
        # In a real system, this would consider multiple factors
        # and possibly contact emergency contacts
        pass
    
    def generate_daily_report(self):
        """Generate daily report for user and caregivers"""
        report = {
            "date": datetime.date.today().isoformat(),
            "user_id": self.care_system.current_user,
            "activity_level": self.care_system.health_tracker.activity_level,
            "sleep_quality": self.care_system.health_tracker.sleep_quality,
            "social_engagement": self.care_system.health_tracker.social_engagement,
            "mood_summary": self.get_mood_summary(),
            "medication_adherence": self.care_system.health_tracker.medication_adherence,
            "health_alerts": self.format_health_alerts(),
            "daily_goals_met": self.calculate_daily_goals_met(),
            "interaction_summary": self.get_interaction_summary()
        }
        
        return report
    
    def get_mood_summary(self):
        """Summarize mood indicators"""
        recent_moods = self.care_system.health_tracker.mood_indicators[-10:]  # Last 10 indicators
        if not recent_moods:
            return "No mood data available"
        
        mood_counts = {}
        for mood in recent_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        # Return most common mood
        most_common_mood = max(mood_counts, key=mood_counts.get)
        return f"Mostly {most_common_mood} (based on {len(recent_moods)} observations)"
    
    def format_health_alerts(self):
        """Format health alerts for the report"""
        alerts = self.care_system.health_tracker.health_alerts[-5:]  # Last 5 alerts
        formatted = []
        for alert, timestamp in alerts:
            formatted.append({
                "type": alert.value,
                "timestamp": timestamp.isoformat()
            })
        return formatted
    
    def calculate_daily_goals_met(self):
        """Calculate which daily goals were met"""
        progress = self.care_system.calculate_goals_progress()
        goals = self.care_system.daily_goals
        
        met_goals = {}
        for goal, target in goals.items():
            if isinstance(target, (int, float)):
                met_goals[goal] = progress.get(goal, 0) >= target * 0.8  # 80% of target
        
        return met_goals
    
    def get_interaction_summary(self):
        """Summarize recent interactions"""
        # Count interaction types
        interaction_types = {}
        for interaction in self.care_system.interaction_history[-20:]:  # Last 20 interactions
            type_ = interaction["type"]
            interaction_types[type_] = interaction_types.get(type_, 0) + 1
        
        return interaction_types
```

### Step 5: Main System Integration

```python
#!/usr/bin/env python

import rospy
import time
from social_care_system import SocialCareSystem
from context_aware_interaction import ContextAwareInteraction
from health_monitoring_system import HealthMonitoringSystem

class SocialCareRobot:
    def __init__(self):
        rospy.init_node('social_care_robot')
        
        # Initialize system components
        self.care_system = SocialCareSystem("ElderCare Assistant")
        self.interaction_system = ContextAwareInteraction(self.care_system)
        self.health_system = HealthMonitoringSystem(self.care_system)
        
        # Initialize with default user
        self.care_system.load_user_profile("elderly_user_001")
        
        # Set up ROS interfaces
        from std_msgs.msg import String
        self.speech_publisher = rospy.Publisher('/tts_speech', String, queue_size=10)
        self.display_publisher = rospy.Publisher('/display_text', String, queue_size=10)
        
        # System state
        self.running = True
        self.interaction_interval = 300  # 5 minutes between regular interactions
        self.last_interaction_time = time.time()
        
        rospy.loginfo("Social Care Robot system initialized")
    
    def run_interaction_cycle(self):
        """Main interaction cycle"""
        while not rospy.is_shutdown() and self.running:
            current_time = time.time()
            
            # Check if it's time for regular interaction
            if current_time - self.last_interaction_time > self.interaction_interval:
                self.execute_regular_interaction()
                self.last_interaction_time = current_time
            
            # Update health metrics
            self.health_system.update_health_metrics()
            
            # Check for scheduled events (reminders, etc.)
            self.check_scheduled_events()
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
    
    def execute_regular_interaction(self):
        """Execute a regular interaction with the user"""
        try:
            # Determine context
            context = self.interaction_system.determine_interaction_context()
            
            # Select and execute interaction
            interaction = self.interaction_system.select_interaction(context)
            
            # Publish interaction to speech system
            speech_msg = String()
            speech_msg.data = interaction
            self.speech_publisher.publish(speech_msg)
            
            # Update interaction history
            self.care_system.update_interaction_history(
                interaction_type="proactive",
                content="Regular check-in",
                response=interaction
            )
            
            rospy.loginfo(f"Executed interaction: {interaction[:50]}...")
            
        except Exception as e:
            rospy.logerr(f"Error in interaction cycle: {e}")
    
    def check_scheduled_events(self):
        """Check for and execute scheduled events (reminders, etc.)"""
        current_time = datetime.datetime.now().time()
        
        # Check for medication times (simplified)
        medication_times = [datetime.time(9, 0), datetime.time(21, 0)]  # 9 AM and 9 PM
        if current_time in medication_times:
            self.execute_medication_reminder()
    
    def execute_medication_reminder(self):
        """Execute medication reminder interaction"""
        try:
            reminder = "It's time for your evening medication. Would you like me to bring it to you?"
            
            # Publish reminder
            speech_msg = String()
            speech_msg.data = reminder
            self.speech_publisher.publish(speech_msg)
            
            # Record interaction
            self.care_system.update_interaction_history(
                interaction_type="medication_reminder",
                content="Medication time reminder",
                response=reminder
            )
            
            rospy.loginfo("Executed medication reminder")
            
        except Exception as e:
            rospy.logerr(f"Error in medication reminder: {e}")
    
    def generate_daily_report(self):
        """Generate and publish daily report"""
        try:
            report = self.health_system.generate_daily_report()
            
            # In a real system, this would be sent to caregivers
            # For now, just log the report
            rospy.loginfo(f"Generated daily report for {report['user_id']}")
            rospy.loginfo(f"Activity level: {report['activity_level']}")
            rospy.loginfo(f"Social engagement: {report['social_engagement']}")
            
            # Save report
            self.save_daily_report(report)
            
        except Exception as e:
            rospy.logerr(f"Error generating daily report: {e}")
    
    def save_daily_report(self, report):
        """Save daily report to persistent storage"""
        import json
        from datetime import date
        
        filename = f"daily_report_{date.today().isoformat()}_{report['user_id']}.json"
        
        # In a real system, this would save to a database
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def run(self):
        """Run the social care robot system"""
        rospy.loginfo("Starting Social Care Robot interaction cycle")
        
        try:
            self.run_interaction_cycle()
        except KeyboardInterrupt:
            rospy.loginfo("Social Care Robot shutting down")
            self.running = False

if __name__ == '__main__':
    try:
        robot = SocialCareRobot()
        robot.run()
    except rospy.ROSInterruptException:
        pass
```

### Step 6: Safety and Privacy System

```python
class SafetyAndPrivacySystem:
    def __init__(self, care_system: SocialCareSystem):
        self.care_system = care_system
        self.privacy_controls = {
            "health_data_sharing": False,
            "location_tracking": True,
            "conversation_logging": False,
            "emergency_contacts": [],
            "data_retention_days": 30
        }
        self.safety_protocols = SafetyProtocols()
        
    def enforce_privacy_settings(self, data_type: str, data: dict):
        """Enforce privacy settings on data before processing or storage"""
        if data_type == "health_data" and not self.privacy_controls["health_data_sharing"]:
            # Anonymize health data or refuse processing
            return self.anonymize_health_data(data)
        elif data_type == "location" and not self.privacy_controls["location_tracking"]:
            # Don't process location data
            return None
        elif data_type == "conversations" and not self.privacy_controls["conversation_logging"]:
            # Don't log conversations
            return None
        
        return data
    
    def anonymize_health_data(self, health_data: dict):
        """Remove identifying information from health data"""
        anonymized = health_data.copy()
        
        # Remove any direct identifiers
        for key in ["user_id", "name", "address"]:
            anonymized.pop(key, None)
        
        # Add random identifier
        import random
        anonymized["anonymous_id"] = f"anon_{random.randint(1000, 9999)}"
        
        return anonymized
    
    def trigger_safety_protocol(self, alert_type: HealthAlert):
        """Trigger appropriate safety protocol based on alert type"""
        return self.safety_protocols.handle_alert(alert_type)
    
    def check_emergency_conditions(self):
        """Check if emergency protocols should be activated"""
        # Check for emergency health alerts
        recent_alerts = [alert for alert, timestamp in self.care_system.health_tracker.health_alerts
                        if alert == HealthAlert.EMERGENCY and 
                        (datetime.datetime.now() - timestamp).seconds < 300]  # Last 5 minutes
        
        if recent_alerts:
            self.activate_emergency_protocol()
            return True
        
        # Check for other emergency indicators
        # This would involve more complex checks in a real system
        return False
    
    def activate_emergency_protocol(self):
        """Activate emergency response protocols"""
        # In a real system, this would:
        # 1. Contact emergency services
        # 2. Notify family members
        # 3. Guide user through emergency procedures
        # 4. Document the event
        
        rospy.logerr("EMERGENCY PROTOCOL ACTIVATED")
        
        # For simulation, just log the event
        self.log_emergency_event()
    
    def log_emergency_event(self):
        """Log emergency event for review"""
        emergency_log = {
            "timestamp": datetime.datetime.now(),
            "user_id": self.care_system.current_user,
            "event_type": "emergency_protocol_activated",
            "health_state": self.care_system.health_tracker.__dict__.copy()
        }
        
        # Save to emergency log file
        import json
        filename = f"emergency_log_{datetime.date.today().isoformat()}.json"
        with open(filename, 'a') as f:
            f.write(json.dumps(emergency_log) + "\n")

class SafetyProtocols:
    def __init__(self):
        self.protocol_map = {
            HealthAlert.NORMAL: self.normal_operation,
            HealthAlert.LOW_ACTIVITY: self.low_activity_check,
            HealthAlert.UNUSUAL_BEHAVIOR: self.unusual_behavior_check,
            HealthAlert.EMERGENCY: self.emergency_response
        }
    
    def handle_alert(self, alert_type: HealthAlert):
        """Handle different types of health alerts"""
        return self.protocol_map[alert_type]()
    
    def normal_operation(self):
        """Continue normal operation"""
        return {"action": "continue_normal_operation", "severity": "none"}
    
    def low_activity_check(self):
        """Check and respond to low activity"""
        return {
            "action": "increase_monitoring", 
            "severity": "low",
            "recommendation": "Suggest light physical activity"
        }
    
    def unusual_behavior_check(self):
        """Check and respond to unusual behavior"""
        return {
            "action": "check_in_with_user",
            "severity": "medium", 
            "recommendation": "Engage in wellness conversation"
        }
    
    def emergency_response(self):
        """Handle emergency situation"""
        return {
            "action": "activate_emergency_protocols",
            "severity": "high",
            "recommendation": "Contact emergency services"
        }
```

### Step 7: User Experience and Evaluation

```python
class UserExperienceEvaluator:
    def __init__(self, care_system: SocialCareSystem):
        self.care_system = care_system
        self.engagement_metrics = {
            "interaction_frequency": 0,
            "response_quality": 0.0,
            "user_satisfaction": 0.0,
            "system_adaptability": 0.0
        }
        
    def evaluate_interaction_effectiveness(self, interaction_result: dict):
        """Evaluate effectiveness of an interaction"""
        # Calculate engagement based on user response
        user_response = interaction_result.get("user_response", "")
        interaction_type = interaction_result.get("type", "")
        
        # Simple engagement metric based on response length and positiveness
        engagement_score = self.calculate_engagement_score(user_response, interaction_type)
        
        # Update metrics
        self.engagement_metrics["interaction_frequency"] += 1
        self.update_average("response_quality", engagement_score)
        
        return engagement_score
    
    def calculate_engagement_score(self, user_response: str, interaction_type: str):
        """Calculate engagement score based on user response"""
        if not user_response:
            return 0.1  # Very low engagement if no response
        
        # Length of response indicates engagement level
        length_factor = min(1.0, len(user_response) / 100)  # Normalize for 100 characters
        
        # Positive words indicate good engagement
        positive_words = ["yes", "ok", "good", "great", "thank", "like", "love", "happy"]
        positive_factor = sum(1 for word in positive_words if word in user_response.lower())
        positive_factor = min(1.0, positive_factor / 2)  # Max 2 positive indicators = 1.0
        
        # Calculate overall score
        score = 0.4 * length_factor + 0.6 * positive_factor
        return min(1.0, score)  # Cap at 1.0
    
    def update_average(self, metric_name: str, new_value: float):
        """Update running average for a metric"""
        # Simple exponential moving average
        alpha = 0.1  # Smoothing factor
        current_avg = self.engagement_metrics[metric_name]
        new_avg = alpha * new_value + (1 - alpha) * current_avg
        self.engagement_metrics[metric_name] = new_avg
    
    def generate_user_experience_report(self):
        """Generate detailed UX report"""
        report = {
            "period": "daily",
            "user_id": self.care_system.current_user,
            "engagement_metrics": self.engagement_metrics.copy(),
            "interaction_types": self.analyze_interaction_types(),
            "adaptive_changes": self.count_adaptive_changes(),
            "user_feedback_indicators": self.extract_user_feedback()
        }
        
        return report
    
    def analyze_interaction_types(self):
        """Analyze distribution of interaction types"""
        type_counts = {}
        for interaction in self.care_system.interaction_history[-50:]:  # Last 50 interactions
            type_ = interaction["type"]
            type_counts[type_] = type_counts.get(type_, 0) + 1
        
        return type_counts
    
    def count_adaptive_changes(self):
        """Count how often system adapted to user needs"""
        # Count changes in interaction mode, topic, etc.
        mode_changes = 0
        prev_mode = None
        
        for interaction in self.care_system.interaction_history[-20:]:
            curr_mode = interaction.get("mode", "unknown")
            if prev_mode and prev_mode != curr_mode:
                mode_changes += 1
            prev_mode = curr_mode
        
        return mode_changes
    
    def extract_user_feedback(self):
        """Extract positive/negative indicators from interactions"""
        positive_indicators = 0
        negative_indicators = 0
        
        for interaction in self.care_system.interaction_history[-30:]:
            response = interaction.get("response", "").lower()
            
            positive_words = ["thank", "good", "great", "love", "happy", "helpful"]
            negative_words = ["no", "don't", "stop", "leave", "tired", "bored"]
            
            positive_indicators += sum(1 for word in positive_words if word in response)
            negative_indicators += sum(1 for word in negative_words if word in response)
        
        return {
            "positive_indicators": positive_indicators,
            "negative_indicators": negative_indicators,
            "overall_tone": "positive" if positive_indicators > negative_indicators else "negative"
        }
```

## Testing the Project

1. **Setup Environment**: Deploy the system in a safe, controlled environment
2. **Configure User Profile**: Set up user preferences and daily routines
3. **Monitor Interactions**: Observe how the robot adapts to user behavior
4. **Check Health Tracking**: Verify that activity and engagement metrics update
5. **Test Safety Systems**: Ensure emergency protocols work correctly
6. **Evaluate User Experience**: Collect feedback on interaction quality

## Extensions

1. **Multi-user Support**: Extend to support multiple residents in care facilities
2. **Advanced Health Monitoring**: Integrate with medical devices and health systems
3. **Family Connection**: Add interfaces to keep family members informed
4. **Cultural Adaptation**: Adapt interaction styles for different cultural backgrounds
5. **Learning Improvements**: Implement more sophisticated learning algorithms to improve over time

## Ethical Considerations

1. **Privacy**: Ensure all personal data is protected and shared only with proper consent
2. **Autonomy**: Maintain the user's ability to control interactions and data sharing
3. **Dignity**: Design interactions that respect the user's dignity and individuality
4. **Transparency**: Make the robot's capabilities and limitations clear to users
5. **Emergency Response**: Ensure proper handling of emergency situations while considering user preferences
6. **Social Isolation**: Use the robot to enhance, not replace, human interaction

This project demonstrates the integration of multiple complex systems required for social care robotics, addressing both technical and humanistic challenges in creating supportive and respectful AI companions for elderly care.