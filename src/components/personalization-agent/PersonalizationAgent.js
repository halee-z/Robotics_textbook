import React, { useState } from 'react';
import styles from './PersonalizationAgent.module.css';
import { personalizationService } from './api/personalizationService';

const PersonalizationAgent = ({ chapterContent, onContentUpdate, chapterTitle = null }) => {
  const [isPersonalizing, setIsPersonalizing] = useState(false);
  const [preferences, setPreferences] = useState({
    level: 'intermediate',  // beginner, intermediate, advanced
    interests: [],
    goals: [],
    learningStyle: 'comprehensive'  // comprehensive, concise, example-focused
  });
  const [showPreferences, setShowPreferences] = useState(false);
  const [error, setError] = useState(null);

  const togglePreferences = () => {
    setShowPreferences(!showPreferences);
  };

  const handlePreferenceChange = (prefType, value) => {
    setPreferences(prev => ({
      ...prev,
      [prefType]: value
    }));
  };

  const handleInterestToggle = (interest) => {
    setPreferences(prev => {
      const newInterests = prev.interests.includes(interest)
        ? prev.interests.filter(i => i !== interest)
        : [...prev.interests, interest];
      return { ...prev, interests: newInterests };
    });
  };

  const handleGoalToggle = (goal) => {
    setPreferences(prev => {
      const newGoals = prev.goals.includes(goal)
        ? prev.goals.filter(g => g !== goal)
        : [...prev.goals, goal];
      return { ...prev, goals: newGoals };
    });
  };

  const personalizeContent = async () => {
    setIsPersonalizing(true);
    setError(null);

    try {
      // Use the service to personalize content
      const result = await personalizationService.personalizeContent(
        chapterContent,
        preferences,
        chapterTitle
      );

      if (onContentUpdate) {
        onContentUpdate(result.personalized_content);
      }

      setIsPersonalizing(false);
      setShowPreferences(false);
    } catch (err) {
      console.error('Error personalizing content:', err);
      setError('Failed to personalize content. Please try again.');
      setIsPersonalizing(false);
    }
  };

  const interestOptions = [
    'ROS 2 Development',
    'Vision-Language Models',
    'Humanoid Locomotion',
    'Human-Robot Interaction',
    'Simulation Environments',
    'Robot Control Systems'
  ];

  const goalOptions = [
    'Understand Fundamentals',
    'Build Applications',
    'Conduct Research',
    'Implement Algorithms',
    'Develop Systems',
    'Troubleshoot Issues'
  ];

  return (
    <div className={styles.personalizationContainer}>
      <div className={styles.header}>
        <h3>Personalize Your Learning Experience</h3>
        <button
          className={styles.toggleButton}
          onClick={togglePreferences}
        >
          {showPreferences ? 'Hide Preferences' : 'Personalize Content'}
        </button>
      </div>

      {showPreferences && (
        <div className={styles.preferencesPanel}>
          <div className={styles.preferenceSection}>
            <h4>Knowledge Level</h4>
            <div className={styles.radioGroup}>
              {['beginner', 'intermediate', 'advanced'].map(level => (
                <label key={level} className={styles.radioLabel}>
                  <input
                    type="radio"
                    name="level"
                    value={level}
                    checked={preferences.level === level}
                    onChange={(e) => handlePreferenceChange('level', e.target.value)}
                    className={styles.radioButton}
                  />
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </label>
              ))}
            </div>
          </div>

          <div className={styles.preferenceSection}>
            <h4>Your Interests</h4>
            <div className={styles.checkboxGroup}>
              {interestOptions.map(interest => (
                <label key={interest} className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={preferences.interests.includes(interest)}
                    onChange={() => handleInterestToggle(interest)}
                    className={styles.checkbox}
                  />
                  {interest}
                </label>
              ))}
            </div>
          </div>

          <div className={styles.preferenceSection}>
            <h4>Learning Goals</h4>
            <div className={styles.checkboxGroup}>
              {goalOptions.map(goal => (
                <label key={goal} className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={preferences.goals.includes(goal)}
                    onChange={() => handleGoalToggle(goal)}
                    className={styles.checkbox}
                  />
                  {goal}
                </label>
              ))}
            </div>
          </div>

          <div className={styles.preferenceSection}>
            <h4>Learning Style</h4>
            <div className={styles.radioGroup}>
              {['comprehensive', 'concise', 'example-focused'].map(style => (
                <label key={style} className={styles.radioLabel}>
                  <input
                    type="radio"
                    name="learningStyle"
                    value={style}
                    checked={preferences.learningStyle === style}
                    onChange={(e) => handlePreferenceChange('learningStyle', e.target.value)}
                    className={styles.radioButton}
                  />
                  {style === 'comprehensive' ? 'Comprehensive' :
                   style === 'concise' ? 'Concise' : 'Example-Focused'}
                </label>
              ))}
            </div>
          </div>

          <button
            className={styles.personalizeButton}
            onClick={personalizeContent}
            disabled={isPersonalizing}
          >
            {isPersonalizing ? 'Personalizing...' : 'Apply Personalization'}
          </button>

          {error && <div className={styles.error}>{error}</div>}
        </div>
      )}

      {isPersonalizing && (
        <div className={styles.progressIndicator}>
          <div className={styles.spinner}></div>
          <p>Adjusting content to match your preferences...</p>
        </div>
      )}
    </div>
  );
};

export default PersonalizationAgent;