// api/personalizationService.js

import { mockPersonalizationAPI } from './mockPersonalizationAPI';

// Always use mock API in development to avoid connection issues
const USE_MOCK_API = true;

const API_BASE_URL = '/api'; // This will be mapped to the actual API endpoint

export const personalizationService = {
  // Personalize content based on user preferences
  async personalizeContent(content, preferences, chapterTitle = null) {
    if (USE_MOCK_API || process.env.NODE_ENV !== 'production') {
      // Use mock API in development to avoid connection issues
      return mockPersonalizationAPI.personalizeContent(content, preferences, chapterTitle);
    } else {
      const response = await fetch(`${API_BASE_URL}/personalization/personalize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          preferences,
          chapter_title: chapterTitle
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to personalize content: ${response.status} ${response.statusText}`);
      }

      return response.json();
    }
  },

  // Get user's profile/preferences
  async getUserProfile(userId) {
    if (USE_MOCK_API || process.env.NODE_ENV !== 'production') {
      // Use mock API in development to avoid connection issues
      return mockPersonalizationAPI.getUserProfile(userId);
    } else {
      const response = await fetch(`${API_BASE_URL}/users/${userId}/profile`);

      if (!response.ok) {
        throw new Error(`Failed to get user profile: ${response.status} ${response.statusText}`);
      }

      return response.json();
    }
  },

  // Update user's preferences
  async updateUserPreferences(userId, preferences) {
    if (USE_MOCK_API || process.env.NODE_ENV !== 'production') {
      // Use mock API in development to avoid connection issues
      return mockPersonalizationAPI.updateUserPreferences(userId, preferences);
    } else {
      const response = await fetch(`${API_BASE_URL}/users/${userId}/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(preferences)
      });

      if (!response.ok) {
        throw new Error(`Failed to update user preferences: ${response.status} ${response.statusText}`);
      }

      return response.json();
    }
  }
};