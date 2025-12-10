// api/quizService.js

import { mockQuizAPI } from './mockQuizAPI';

// Always use mock API in development to avoid connection issues
const USE_MOCK_API = true;

const API_BASE_URL = '/api'; // This will be mapped to the actual API endpoint

export const quizService = {
  // Start a new quiz session
  async startQuiz(difficulty, topic = null) {
    if (USE_MOCK_API || process.env.NODE_ENV !== 'production') {
      // Use mock API in development to avoid connection issues
      return mockQuizAPI.startQuiz(difficulty, topic);
    } else {
      const response = await fetch(`${API_BASE_URL}/quiz/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          difficulty,
          topic
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to start quiz: ${response.status} ${response.statusText}`);
      }

      return response.json();
    }
  },

  // Submit an answer for a question
  async submitAnswer(sessionId, questionId, answer) {
    if (USE_MOCK_API || process.env.NODE_ENV !== 'production') {
      // Use mock API in development to avoid connection issues
      return mockQuizAPI.submitAnswer(sessionId, questionId, answer);
    } else {
      const response = await fetch(`${API_BASE_URL}/quiz/submit_answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          question_id: questionId,
          answer
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to submit answer: ${response.status} ${response.statusText}`);
      }

      return response.json();
    }
  },

  // Get quiz results
  async getResults(sessionId) {
    if (USE_MOCK_API || process.env.NODE_ENV !== 'production') {
      // Use mock API in development to avoid connection issues
      return mockQuizAPI.getResults(sessionId);
    } else {
      const response = await fetch(`${API_BASE_URL}/quiz/results/${sessionId}`);

      if (!response.ok) {
        throw new Error(`Failed to get results: ${response.status} ${response.statusText}`);
      }

      return response.json();
    }
  }
};