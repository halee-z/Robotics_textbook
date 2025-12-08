import React, { useState, useEffect } from 'react';
import axios from 'axios';

const RobotStatus = () => {
  const [robotStatus, setRobotStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRobotStatus();
    // Set up periodic updates
    const interval = setInterval(fetchRobotStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchRobotStatus = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/robot/status');
      setRobotStatus(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch robot status');
      console.error('Error fetching robot status:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderJointStatus = () => {
    if (!robotStatus || !robotStatus.joint_angles) return null;

    return (
      <div className="joint-status">
        <h3>Joint Angles</h3>
        <div className="joint-grid">
          {Object.entries(robotStatus.joint_angles).map(([joint, angle]) => (
            <div key={joint} className="joint-item">
              <span className="joint-name">{joint}:</span>
              <span className="joint-angle">{angle.toFixed(3)} rad</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loading) {
    return <div className="robot-status">Loading robot status...</div>;
  }

  if (error) {
    return (
      <div className="robot-status error">
        <h2>Robot Status: Error</h2>
        <p>{error}</p>
        <button onClick={fetchRobotStatus}>Retry</button>
      </div>
    );
  }

  return (
    <div className="robot-status">
      <h2>Humanoid Robot Status</h2>
      
      <div className="status-summary">
        <div className={`connection-status ${robotStatus.is_connected ? 'connected' : 'disconnected'}`}>
          <h3>Connection Status</h3>
          <p>{robotStatus.is_connected ? 'Connected' : 'Disconnected'}</p>
        </div>
        
        <div className="battery-status">
          <h3>Battery Level</h3>
          <div className="battery-bar">
            <div 
              className="battery-fill" 
              style={{ width: `${(robotStatus.battery_level || 0) * 100}%` }}
            />
          </div>
          <p>{Math.round((robotStatus.battery_level || 0) * 100)}%</p>
        </div>
      </div>

      {renderJointStatus()}

      <div className="active-agents">
        <h3>Active Subagents</h3>
        <ul>
          {robotStatus.active_agents && robotStatus.active_agents.length > 0 ? (
            robotStatus.active_agents.map(agent => (
              <li key={agent}>{agent}</li>
            ))
          ) : (
            <li>None active</li>
          )}
        </ul>
      </div>

      <div className="robot-controls">
        <h3>Quick Controls</h3>
        <div className="control-buttons">
          <button onClick={() => sendRobotCommand('wave_hand')}>Wave Hand</button>
          <button onClick={() => sendRobotCommand('walk_forward')}>Walk Forward</button>
          <button onClick={() => sendRobotCommand('balance')}>Balance</button>
          <button onClick={() => sendRobotCommand('speak', { text: "Hello, I am your educational robot assistant" })}>Say Hello</button>
        </div>
      </div>
    </div>
  );
};

// Function to send robot commands (would need to be defined in a parent component or context)
const sendRobotCommand = async (command, params = null) => {
  try {
    const response = await axios.post('/api/robot/control', {
      command,
      parameters: params
    });
    
    if (response.data.status === 'success') {
      console.log('Command executed successfully:', response.data.message);
    } else {
      console.warn('Command not successful:', response.data.message);
    }
  } catch (error) {
    console.error('Error sending robot command:', error);
  }
};

export default RobotStatus;