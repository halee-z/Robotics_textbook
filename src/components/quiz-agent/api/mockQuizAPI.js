// Mock API for Quiz Agent - to be replaced with actual API server in production
// This simulates the API endpoints documented in /docs/api/overview.md

// In-memory storage for quiz sessions (in a real app, this would be in a database)
let quizSessions = {};

// Generate a unique session ID
const generateSessionId = () => {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

// Sample questions organized by difficulty - these should come from the actual textbook content
const sampleQuestions = {
  'beginner': [
    {
      id: 1,
      question: "True or False: Humanoid robots must have a human-like appearance to be considered as such.",
      type: "TF",
      correctAnswer: false,
      explanation: "Humanoid robots are characterized by having similar capabilities to humans (like bipedal locomotion and manipulation) rather than strictly resembling humans in appearance.",
      topic: 'Introduction to AI & Humanoid Robotics'
    },
    {
      id: 2,
      question: "In ROS 2, what is the purpose of a 'launch file'?",
      type: "MCQ",
      options: [
        "To compile source code",
        "To start multiple nodes with a single command",
        "To debug runtime errors",
        "To visualize robot data"
      ],
      correctAnswer: 1,
      explanation: "Launch files allow you to start multiple nodes with a single command, making it easier to manage complex systems.",
      topic: 'ROS 2 Fundamentals'
    },
    {
      id: 3,
      question: "What does RMW stand for in ROS 2?",
      type: "short",
      correctAnswer: "ROS Middleware",
      explanation: "RMW stands for ROS Middleware, which abstracts the underlying DDS implementation in ROS 2.",
      topic: 'ROS 2 Fundamentals'
    }
  ],
  'normal': [
    {
      id: 1,
      question: "Which of the following is NOT a common application of Vision-Language Models in robotics?",
      type: "MCQ",
      options: [
        "Object recognition and manipulation",
        "Natural language command interpretation",
        "Path planning in unknown environments",
        "Human-robot interaction"
      ],
      correctAnswer: 2,
      explanation: "While VLMs enhance perception and communication, path planning in unknown environments typically relies on mapping and navigation algorithms rather than vision-language models.",
      topic: 'Vision-Language Models in Robotics'
    },
    {
      id: 2,
      question: "True or False: CLIP models can associate images with text descriptions without task-specific fine-tuning.",
      type: "TF",
      correctAnswer: true,
      explanation: "CLIP models are trained on image-text pairs and can match images to text descriptions without needing fine-tuning for specific tasks.",
      topic: 'Vision-Language Models in Robotics'
    },
    {
      id: 3,
      question: "What are the main challenges in humanoid robotics?",
      type: "short",
      correctAnswer: "Balance and stability, real-time decision making, human-robot interaction, and energy efficiency",
      explanation: "Key challenges in humanoid robotics include balance and stability, real-time decision making, human-robot interaction, and energy efficiency.",
      topic: 'Introduction to AI & Humanoid Robotics'
    }
  ],
  'high': [
    {
      id: 1,
      question: "Which simulation environment is developed by Open Robotics and widely used for robotics simulation?",
      type: "MCQ",
      options: [
        "Unity Robotics",
        "Isaac Sim",
        "Gazebo",
        "Webots"
      ],
      correctAnswer: 2,
      explanation: "Gazebo, developed by Open Robotics, is one of the most popular simulation environments for robotics, offering realistic physics simulation and sensor modeling.",
      topic: 'Simulation Environments'
    },
    {
      id: 2,
      question: "What does 'ZMP' stand for in humanoid locomotion?",
      type: "short",
      correctAnswer: "Zero Moment Point",
      explanation: "Zero Moment Point is a crucial concept in humanoid robotics that helps ensure stable walking patterns.",
      topic: 'Humanoid Control Systems'
    },
    {
      id: 3,
      question: "Which control method is commonly used for maintaining balance in humanoid robots?",
      type: "MCQ",
      options: [
        "PID Control",
        "Model Predictive Control",
        "Fuzzy Logic Control",
        "Neural Network Control"
      ],
      correctAnswer: 1,
      explanation: "Model Predictive Control (MPC) is widely used in humanoid robotics for balance control due to its ability to handle constraints and predict future states.",
      topic: 'Humanoid Control Systems'
    },
    {
      id: 4,
      question: "Which of the following is a key principle of effective human-robot interaction?",
      type: "MCQ",
      options: [
        "Minimizing robot autonomy",
        "Ensuring robot behavior is predictable and understandable",
        "Making robots as human-like as possible",
        "Reducing robot response time"
      ],
      correctAnswer: 1,
      explanation: "Predictable and understandable behavior is crucial for effective human-robot interaction, as it builds trust and allows humans to anticipate robot actions.",
      topic: 'Human-Robot Interaction'
    }
  ]
};

// Utility function to shuffle an array
const shuffleArray = (array) => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
};

// Get a random subset of questions based on difficulty
const getQuestionsForDifficulty = (difficulty) => {
  const allQuestions = sampleQuestions[difficulty] || [];
  const shuffledQuestions = shuffleArray(allQuestions);
  
  // Select number of questions based on difficulty
  let questionCount;
  switch(difficulty) {
    case 'beginner':
      questionCount = Math.min(3, shuffledQuestions.length);
      break;
    case 'normal':
      questionCount = Math.min(4, shuffledQuestions.length);
      break;
    case 'high':
      questionCount = Math.min(5, shuffledQuestions.length);
      break;
    default:
      questionCount = Math.min(3, shuffledQuestions.length);
  }
  
  return shuffledQuestions.slice(0, questionCount);
};

// Mock API functions
export const mockQuizAPI = {
  // Start a new quiz session
  startQuiz: (difficulty, topic = null) => {
    const sessionId = generateSessionId();
    const questions = getQuestionsForDifficulty(difficulty);
    
    // Select the first question
    const currentQuestion = questions[0];
    
    // Create quiz session object
    const quizSession = {
      session_id: sessionId,
      difficulty: difficulty,
      questions: questions,
      total_questions: questions.length,
      current_question_index: 0,
      answers: [],
      score: 0,
      started_at: new Date().toISOString()
    };
    
    // Store the session
    quizSessions[sessionId] = quizSession;
    
    // Return the first question
    return {
      session_id: sessionId,
      total_questions: questions.length,
      current_question: currentQuestion
    };
  },

  // Submit an answer for a question
  submitAnswer: (sessionId, questionId, answer) => {
    const session = quizSessions[sessionId];
    if (!session) {
      throw new Error('Quiz session not found');
    }

    // Find the current question
    const currentQuestion = session.questions[session.current_question_index];
    if (!currentQuestion) {
      throw new Error('Current question not found');
    }

    // Check if the answer is correct
    let isCorrect = false;
    if (currentQuestion.type === 'MCQ') {
      isCorrect = answer === currentQuestion.correctAnswer;
    } else if (currentQuestion.type === 'TF') {
      isCorrect = answer === currentQuestion.correctAnswer;
    } else if (currentQuestion.type === 'short') {
      // Simple string comparison (could be enhanced with fuzzy matching)
      isCorrect = answer.toLowerCase().includes(currentQuestion.correctAnswer.toLowerCase());
    }

    // Update score if correct
    if (isCorrect) {
      session.score += 1;
    }

    // Save the answer
    const answerRecord = {
      question_id: currentQuestion.id,
      question: currentQuestion.question,
      user_answer: answer,
      correct_answer: currentQuestion.correctAnswer,
      is_correct: isCorrect,
      explanation: currentQuestion.explanation
    };
    
    session.answers.push(answerRecord);

    // Move to the next question
    session.current_question_index += 1;

    // Prepare response
    const response = {
      is_correct: isCorrect,
      explanation: currentQuestion.explanation,
      score: session.score,
      total_questions: session.total_questions
    };

    // If there's a next question, include it
    if (session.current_question_index < session.questions.length) {
      const nextQuestion = session.questions[session.current_question_index];
      response.next_question = nextQuestion;
    }

    // Update the session
    quizSessions[sessionId] = session;

    return response;
  },

  // Get quiz results
  getResults: (sessionId) => {
    const session = quizSessions[sessionId];
    if (!session) {
      throw new Error('Quiz session not found');
    }

    return {
      session_id: sessionId,
      score: session.score,
      total_questions: session.total_questions,
      percentage: Math.round((session.score / session.total_questions) * 100),
      difficulty: session.difficulty,
      answers: session.answers.map(ans => ({
        question_id: ans.question_id,
        question: ans.question,
        user_answer: ans.user_answer,
        correct_answer: ans.correct_answer,
        is_correct: ans.is_correct,
        explanation: ans.explanation
      }))
    };
  },

  // Clean up old sessions (optional, for memory management)
  cleanup: () => {
    // In a real implementation, you might want to remove old sessions
    // For now, just clear all sessions
    quizSessions = {};
  }
};