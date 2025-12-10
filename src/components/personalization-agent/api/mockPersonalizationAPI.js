// Mock API for Personalization Agent - to be replaced with actual API server in production

// In-memory storage for user profiles (in a real app, this would be in a database)
const userProfiles = {
  'default_user': {
    user_id: 'default_user',
    preferences: {
      level: 'intermediate',
      interests: ['ROS 2 Development', 'Humanoid Locomotion'],
      goals: ['Build Applications'],
      learning_style: 'example-focused'
    },
    learning_history: {
      completed_chapters: ['ROS Fundamentals', 'VLM Introduction'],
      quiz_scores: [85, 92, 78],
      time_spent: 3600
    }
  }
};

// Function to adjust content based on preferences
const adjustContent = (content, preferences) => {
  let adjustedContent = content;
  const adjustments = [];

  // Adjust content based on level
  if (preferences.level === 'beginner') {
    adjustedContent = `**BEGINNER LEVEL**: ${adjustedContent} (We've simplified the language and concepts for easier understanding)`;
    adjustments.push('simplified_language');
  } else if (preferences.level === 'advanced') {
    adjustedContent = `**ADVANCED LEVEL**: ${adjustedContent} (We've added deeper technical details and advanced concepts)`;
    adjustments.push('added_depth');
  }

  // Adjust content based on learning style
  if (preferences.learning_style === 'concise') {
    adjustedContent = `**CONCISE VERSION**: ${adjustedContent} (We've shortened and summarized the content)`;
    adjustments.push('concise_format');
  } else if (preferences.learning_style === 'example-focused') {
    adjustedContent = `${adjustedContent} **EXAMPLES**: Here are examples related to your interests: ${preferences.interests.join(', ')}.`;
    adjustments.push('added_examples');
  }

  // Add specific content based on interests and goals
  if (preferences.interests && preferences.interests.length > 0) {
    adjustedContent += ` **RELATED INTERESTS**: This content covers topics related to: ${preferences.interests.join(', ')}.`;
    adjustments.push('added_interest_content');
  }

  if (preferences.goals && preferences.goals.length > 0) {
    adjustedContent += ` **FOR YOUR GOALS**: This content will help you achieve: ${preferences.goals.join(', ')}.`;
    adjustments.push('goal_aligned_content');
  }

  return {
    personalized_content: adjustedContent,
    metadata: {
      processing_time: 0.2,
      adjustments_made: adjustments,
      confidence: 0.9
    }
  };
};

// Mock API functions
export const mockPersonalizationAPI = {
  // Personalize content based on user preferences
  personalizeContent: (content, preferences, chapterTitle = null) => {
    // Simulate API processing time
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(adjustContent(content, preferences));
      }, 300); // Simulate 300ms processing time
    });
  },

  // Get user's profile/preferences
  getUserProfile: (userId) => {
    // Simulate API processing time
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (userProfiles[userId]) {
          resolve(userProfiles[userId]);
        } else {
          reject(new Error('User not found'));
        }
      }, 200); // Simulate 200ms processing time
    });
  },

  // Update user's preferences
  updateUserPreferences: (userId, preferences) => {
    // Simulate API processing time
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (userProfiles[userId]) {
          userProfiles[userId].preferences = { ...userProfiles[userId].preferences, ...preferences };
          resolve({
            status: 'success',
            message: 'Preferences updated successfully'
          });
        } else {
          reject(new Error('User not found'));
        }
      }, 200); // Simulate 200ms processing time
    });
  },

  // Initialize a default user if one doesn't exist
  initializeUser: (userId) => {
    if (!userProfiles[userId]) {
      userProfiles[userId] = {
        user_id: userId,
        preferences: {
          level: 'intermediate',
          interests: [],
          goals: [],
          learning_style: 'comprehensive'
        },
        learning_history: {
          completed_chapters: [],
          quiz_scores: [],
          time_spent: 0
        }
      };
    }
  }
};