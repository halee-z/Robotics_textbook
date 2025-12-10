import React, { useState, useEffect } from 'react';
import styles from './QuizAgent.module.css';
import { quizService } from './api/quizService';

const QuizAgent = () => {
  const [quizState, setQuizState] = useState('setup'); // 'setup', 'question', 'feedback', 'results'
  const [difficulty, setDifficulty] = useState('beginner');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [userAnswer, setUserAnswer] = useState('');
  const [feedback, setFeedback] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);
  const [quizQuestions, setQuizQuestions] = useState([]);
  const [selectedOption, setSelectedOption] = useState('');
  const [answers, setAnswers] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState(null);

  const startQuiz = async () => {
    try {
      const response = await quizService.startQuiz(difficulty);
      setSessionId(response.session_id);
      setQuizQuestions(response.questions || [response.current_question]); // Handle both formats
      setCurrentQuestion(response.current_question);
      setTotalQuestions(response.total_questions);

      if (response.current_question) {
        setQuizState('question');
        setCurrentQuestionIndex(0);
        setScore(0);
        setAnswers([]);

        // Initialize state based on question type
        if (response.current_question.type === 'MCQ') {
          setSelectedOption('');
        } else if (response.current_question.type === 'TF') {
          setSelectedOption('');
        } else if (response.current_question.type === 'short') {
          setUserAnswer('');
        }
      }
    } catch (error) {
      console.error('Error starting quiz:', error);
      setFeedback('Failed to start quiz. Please try again.');
      setShowFeedback(true);
    }
  };

  const handleAnswerSubmit = async () => {
    if (quizState === 'question' && sessionId && currentQuestion) {
      try {
        // Prepare the answer based on question type
        let answerValue;
        if (currentQuestion.type === 'MCQ') {
          answerValue = parseInt(selectedOption);
        } else if (currentQuestion.type === 'TF') {
          answerValue = selectedOption === 'true';
        } else if (currentQuestion.type === 'short') {
          answerValue = userAnswer;
        }

        const response = await quizService.submitAnswer(sessionId, currentQuestion.id, answerValue);

        // Update score
        if (response.is_correct) {
          setScore(prev => prev + 1);
        }

        // Save answer
        const newAnswer = {
          questionId: currentQuestion.id,
          question: currentQuestion.question,
          userAnswer: currentQuestion.type === 'MCQ' ? currentQuestion.options[answerValue] :
                      currentQuestion.type === 'TF' ? (answerValue ? 'True' : 'False') : userAnswer,
          correctAnswer: currentQuestion.correctAnswer,
          isCorrect: response.is_correct,
          explanation: response.explanation,
          topic: currentQuestion.topic
        };

        setAnswers(prev => [...prev, newAnswer]);

        // Show feedback
        setFeedback(response.is_correct ? "Correct! " + response.explanation : "Incorrect. " + response.explanation);
        setShowFeedback(true);
        setQuizState('feedback');

        // Update to next question if provided by API
        if (response.next_question) {
          setCurrentQuestion(response.next_question);
          setCurrentQuestionIndex(prev => prev + 1);
        } else if (response.score !== undefined && response.total_questions !== undefined) {
          // If the API response indicates the end of the quiz
          if (response.score + (response.is_correct ? 1 : 0) >= response.total_questions) {
            // Get final results
            setTimeout(async () => {
              try {
                const results = await quizService.getResults(sessionId);
                setAnswers(results.answers);
                setScore(results.score);
                setQuizState('results');
              } catch (error) {
                console.error('Error getting results:', error);
                setFeedback('Error getting quiz results.');
                setShowFeedback(true);
              }
            }, 1000);
          }
        }
      } catch (error) {
        console.error('Error submitting answer:', error);
        setFeedback('Error submitting answer. Please try again.');
        setShowFeedback(true);
      }
    } else if (quizState === 'feedback') {
      // Move to next question or end quiz if not handled by API
      if (currentQuestionIndex < totalQuestions - 1) {
        setSelectedOption('');
        setUserAnswer('');
        setShowFeedback(false);
        setQuizState('question');
      } else {
        // Get final results from API
        try {
          const results = await quizService.getResults(sessionId);
          setAnswers(results.answers);
          setScore(results.score);
          setQuizState('results');
        } catch (error) {
          console.error('Error getting results:', error);
          setFeedback('Error getting quiz results.');
          setShowFeedback(true);
        }
      }
    }
  };

  const resetQuiz = () => {
    setQuizState('setup');
    setSubject('');
    setDifficulty('beginner');
    setCurrentQuestionIndex(0);
    setScore(0);
    setUserAnswer('');
    setFeedback('');
    setShowFeedback(false);
    setQuizQuestions([]);
    setSelectedOption('');
    setAnswers([]);
  };

  return (
    <div className={styles.quizContainer}>
      <h2>Interactive Quiz Agent 🎯</h2>

      {quizState === 'setup' && (
        <div className={styles.setupSection}>
          <h3>Welcome to the Quiz Agent! 🎯</h3>
          <p>Test your knowledge of the Educational AI & Humanoid Robotics textbook with our interactive quizzes.</p>

          <div className={styles.inputGroup}>
            <label htmlFor="difficulty-select">Select Difficulty Level:</label>
            <select
              id="difficulty-select"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              className={styles.selectInput}
            >
              <option value="beginner">Beginner</option>
              <option value="normal">Normal</option>
              <option value="high">High</option>
            </select>
          </div>

          <div className={styles.difficultyExplanation}>
            <p><strong>Beginner:</strong> Basic concepts from the textbook</p>
            <p><strong>Normal:</strong> Core principles and applications</p>
            <p><strong>High:</strong> Advanced topics and implementation details</p>
          </div>

          <button
            onClick={startQuiz}
            className={styles.startButton}
          >
            Start Quiz
          </button>
        </div>
      )}
      
      {quizState === 'question' && currentQuestion && (
        <div className={styles.questionSection}>
          <div className={styles.questionHeader}>
            <span className={styles.questionCounter}>
              Question {currentQuestionIndex + 1} of {totalQuestions}
            </span>
            <span className={styles.scoreDisplay}>Score: {score}</span>
          </div>

          <h3 className={styles.questionText}>{currentQuestion.question}</h3>

          {currentQuestion.type === 'MCQ' && currentQuestion.options && (
            <div className={styles.optionsContainer}>
              {currentQuestion.options.map((option, index) => (
                <div
                  key={index}
                  className={`${styles.option} ${selectedOption == index ? styles.selectedOption : ''}`}
                  onClick={() => setSelectedOption(index.toString())}
                >
                  <span className={styles.optionLetter}>{String.fromCharCode(65 + index)}.</span>
                  <span className={styles.optionText}>{option}</span>
                </div>
              ))}
            </div>
          )}

          {currentQuestion.type === 'TF' && (
            <div className={styles.optionsContainer}>
              <div
                className={`${styles.option} ${selectedOption === 'true' ? styles.selectedOption : ''}`}
                onClick={() => setSelectedOption('true')}
              >
                <span className={styles.optionText}>True</span>
              </div>
              <div
                className={`${styles.option} ${selectedOption === 'false' ? styles.selectedOption : ''}`}
                onClick={() => setSelectedOption('false')}
              >
                <span className={styles.optionText}>False</span>
              </div>
            </div>
          )}

          {currentQuestion.type === 'short' && (
            <div className={styles.shortAnswerContainer}>
              <textarea
                value={userAnswer}
                onChange={(e) => setUserAnswer(e.target.value)}
                placeholder="Type your answer here..."
                className={styles.shortAnswerInput}
                rows="3"
              />
            </div>
          )}

          <button
            onClick={handleAnswerSubmit}
            disabled={
              (currentQuestion.type === 'MCQ' && selectedOption === '') ||
              (currentQuestion.type === 'TF' && selectedOption === '') ||
              (currentQuestion.type === 'short' && userAnswer.trim() === '')
            }
            className={styles.submitButton}
          >
            Submit Answer
          </button>
        </div>
      )}
      
      {quizState === 'feedback' && (
        <div className={styles.feedbackSection}>
          <h3>Feedback</h3>
          <p className={showFeedback ? styles.feedbackVisible : styles.feedbackHidden}>
            {feedback}
          </p>
          <button 
            onClick={handleAnswerSubmit}
            className={styles.continueButton}
          >
            {currentQuestionIndex < quizQuestions.length - 1 ? 'Next Question' : 'See Results'}
          </button>
        </div>
      )}
      
      {quizState === 'results' && (
        <div className={styles.resultsSection}>
          <h3>Quiz Completed!</h3>
          <p className={styles.finalScore}>Your final score: {score}/{quizQuestions.length}</p>
          <p className={styles.percentageScore}>Percentage: {Math.round((score / quizQuestions.length) * 100)}%</p>

          <div className={styles.performanceSummary}>
            <h4>Performance Summary</h4>
            <p>Difficulty Level: <span className={styles.difficultyBadge}>{difficulty}</span></p>

            {score === quizQuestions.length ? (
              <p className={styles.perfectScore}>Perfect! You have a strong understanding of this topic.</p>
            ) : score >= quizQuestions.length * 0.7 ? (
              <p>Good job! You have a solid understanding with room for improvement.</p>
            ) : (
              <p>Keep studying! Review the material to strengthen your knowledge.</p>
            )}
          </div>

          <div className={styles.answersReview}>
            <h4>Review Your Answers:</h4>
            {answers.map((answer, index) => (
              <div key={index} className={`${styles.answerItem} ${answer.isCorrect ? styles.correctAnswer : styles.incorrectAnswer}`}>
                <p><strong>Topic:</strong> {answer.topic}</p>
                <p><strong>Question:</strong> {answer.question}</p>
                <p><strong>Your Answer:</strong> {answer.userAnswer}</p>
                {!answer.isCorrect && <p><strong>Correct Answer:</strong> {typeof answer.correctAnswer === 'boolean' ? (answer.correctAnswer ? 'True' : 'False') : (typeof answer.correctAnswer === 'number' ? quizQuestions.find(q => q.id === answer.questionId)?.options[answer.correctAnswer] : answer.correctAnswer)}</p>}
                <p><strong>Explanation:</strong> {answer.explanation}</p>
              </div>
            ))}
          </div>

          <button
            onClick={resetQuiz}
            className={styles.restartButton}
          >
            Take Another Quiz
          </button>
        </div>
      )}
    </div>
  );
};

export default QuizAgent;