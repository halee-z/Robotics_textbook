import React, { useState } from 'react';
import ChatInterface from '../chat/ChatInterface';
import RobotStatus from '../components/RobotStatus';
import SimulationControl from '../components/SimulationControl';

const MainPage = () => {
  const [activeTab, setActiveTab] = useState('chat'); // chat, robot, simulation

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatInterface />;
      case 'robot':
        return <RobotStatus />;
      case 'simulation':
        return <SimulationControl />;
      default:
        return <ChatInterface />;
    }
  };

  return (
    <div className="main-page">
      <header className="main-header">
        <h1>Educational AI & Humanoid Robotics Platform</h1>
        <p>Interactive learning environment for humanoid robotics education</p>
      </header>

      <nav className="main-nav">
        <button 
          className={activeTab === 'chat' ? 'active' : ''}
          onClick={() => setActiveTab('chat')}
        >
          Educational AI Chat
        </button>
        <button 
          className={activeTab === 'robot' ? 'active' : ''}
          onClick={() => setActiveTab('robot')}
        >
          Robot Status & Control
        </button>
        <button 
          className={activeTab === 'simulation' ? 'active' : ''}
          onClick={() => setActiveTab('simulation')}
        >
          Simulation Environment
        </button>
      </nav>

      <main className="main-content">
        {renderContent()}
      </main>

      <footer className="main-footer">
        <p>AI-Native Textbook for Educational AI & Humanoid Robotics</p>
        <p>© {new Date().getFullYear()} - All rights reserved</p>
      </footer>
    </div>
  );
};

export default MainPage;