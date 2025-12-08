import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [selectedAgent, setSelectedAgent] = useState('general');
  const [sessionId, setSessionId] = useState(null);
  const [availableAgents, setAvailableAgents] = useState([]);
  const messagesEndRef = useRef(null);

  // Fetch available agents on component mount
  useEffect(() => {
    fetchAgents();
  }, []);

  const fetchAgents = async () => {
    try {
      const response = await axios.get('/api/agents/available');
      setAvailableAgents(response.data.agents);
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    // Add user message to the chat
    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText(''); // Clear input

    try {
      // Send request to backend
      const response = await axios.post('/api/chat', {
        message: inputText,
        session_id: sessionId,
        agent_type: selectedAgent
      });

      // Update session ID if new
      if (!sessionId) {
        setSessionId(response.data.session_id);
      }

      // Add agent response to the chat
      const agentMessage = {
        id: Date.now() + 1,
        text: response.data.response,
        sender: 'agent',
        agentType: selectedAgent,
        confidence: response.data.confidence,
        sources: response.data.sources,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'agent',
        agentType: 'system',
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Educational AI Assistant</h2>
        <div className="agent-selector">
          <label htmlFor="agent-select">Ask the expert about:</label>
          <select 
            id="agent-select"
            value={selectedAgent} 
            onChange={(e) => setSelectedAgent(e.target.value)}
          >
            <option value="general">General Robotics</option>
            {availableAgents.map(agent => (
              <option key={agent.type} value={agent.type}>
                {agent.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h3>Welcome to the Educational AI & Humanoid Robotics Assistant!</h3>
            <p>Ask me anything about ROS 2, Vision-Language Models, simulation, or humanoid robotics. 
               Select a specialist agent above for more focused expertise.</p>
          </div>
        ) : (
          messages.map(message => (
            <div 
              key={message.id} 
              className={`message ${message.sender === 'user' ? 'user-message' : 'agent-message'}`}
            >
              <div className="message-header">
                <span className="sender">
                  {message.sender === 'user' ? 'You' : 
                   message.agentType === 'ros' ? 'ROS 2 Expert' :
                   message.agentType === 'vlm' ? 'VLM Specialist' :
                   message.agentType === 'simulation' ? 'Simulation Expert' :
                   message.agentType === 'control' ? 'Control Systems Expert' :
                   'Educational AI'}
                </span>
                {message.confidence !== undefined && (
                  <span className="confidence">Confidence: {(message.confidence * 100).toFixed(0)}%</span>
                )}
              </div>
              <div className="message-content">
                {message.text}
              </div>
              {message.sources && message.sources.length > 0 && (
                <div className="message-sources">
                  <strong>Sources:</strong> {message.sources.join(', ')}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question about robotics here..."
          rows="3"
        />
        <button onClick={handleSendMessage} disabled={!inputText.trim()}>
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;