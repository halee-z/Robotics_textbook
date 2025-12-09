import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import { searchBookContent } from './contentService';

const CHATBOT_STORAGE_KEY = 'bookChatbotHistory';

const BookChatbot = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState(() => {
    // Try to load messages from localStorage
    if (typeof window !== 'undefined') {
      const savedMessages = localStorage.getItem(CHATBOT_STORAGE_KEY);
      if (savedMessages) {
        try {
          const parsed = JSON.parse(savedMessages);
          return parsed;
        } catch (e) {
          console.error('Error parsing saved chat history:', e);
          // Return default welcome message if parsing fails
          return [
            { id: 1, text: "Hello! I'm your AI assistant for the Educational AI & Humanoid Robotics textbook. Ask me anything about the book content!", sender: 'bot' }
          ];
        }
      }
    }

    // Default initial message
    return [
      { id: 1, text: "Hello! I'm your AI assistant for the Educational AI & Humanoid Robotics textbook. Ask me anything about the book content!", sender: 'bot' }
    ];
  });

  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem(CHATBOT_STORAGE_KEY, JSON.stringify(messages));
      } catch (e) {
        console.error('Error saving chat history:', e);
      }
    }
  }, [messages]);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Generate bot response based on book content
      const botResponse = await generateBotResponse(inputValue);

      const botMessage = {
        id: Date.now() + 1,
        text: botResponse,
        sender: 'bot'
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an issue processing your question. Please try again.",
        sender: 'bot'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to clear chat history
  const clearChatHistory = () => {
    const welcomeMessage = {
      id: 1,
      text: "Hello! I'm your AI assistant for the Educational AI & Humanoid Robotics textbook. Ask me anything about the book content!",
      sender: 'bot'
    };
    setMessages([welcomeMessage]);

    // Clear from localStorage as well
    if (typeof window !== 'undefined') {
      localStorage.removeItem(CHATBOT_STORAGE_KEY);
    }
  };

  // Function to generate responses based on book content
  const generateBotResponse = async (question) => {
    // Search the book content for relevant information
    const relevantContent = searchBookContent(question);

    if (relevantContent.length > 0) {
      // Combine relevant paragraphs to form the response
      const response = relevantContent.join(' ... ');

      // If the response is too long, truncate it
      if (response.length > 500) {
        return response.substring(0, 500) + '... [Content truncated for readability]';
      }

      return response;
    } else {
      // If no specific content found, provide a general response
      return "I'm an AI assistant trained on the Educational AI & Humanoid Robotics textbook. I couldn't find specific information about your query in the book content. Please try rephrasing your question or ask about a specific topic like ROS 2, Vision-Language Models, Humanoid Robotics, Simulation Environments, or Human-Robot Interaction.";
    }
  };

  if (!isOpen) return null;

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h3>Book Assistant</h3>
        <div className="header-actions">
          <button
            onClick={clearChatHistory}
            className="clear-history-button"
            title="Clear chat history"
          >
            🗑️
          </button>
          <button onClick={onClose} className="close-button">✕</button>
        </div>
      </div>
      <div className="chatbot-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender}`}
          >
            <div className="message-bubble">
              {message.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <div className="message-bubble">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="chatbot-input-form">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about the book..."
          disabled={isLoading}
          rows="3"
        />
        <button type="submit" disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default BookChatbot;