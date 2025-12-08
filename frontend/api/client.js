// api/client.js - Frontend API client for the Educational AI & Humanoid Robotics platform

class RoboticsApiClient {
  constructor(baseURL = '/api') {
    this.baseURL = baseURL;
  }

  // Chat and Educational AI endpoints
  async getChatResponse(message, sessionId = null, agentType = 'general') {
    const response = await fetch(`${this.baseURL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        agent_type: agentType
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getSession(sessionId) {
    const response = await fetch(`${this.baseURL}/sessions/${sessionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getAvailableAgents() {
    const response = await fetch(`${this.baseURL}/agents/available`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Robot control endpoints
  async getRobotStatus() {
    const response = await fetch(`${this.baseURL}/robot/status`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async sendRobotCommand(command, parameters = null) {
    const response = await fetch(`${this.baseURL}/robot/control`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        command,
        parameters
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Simulation endpoints
  async startSimulation(simulatorType = 'gazebo', sceneName = 'default') {
    const response = await fetch(`${this.baseURL}/simulation/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        simulator_type: simulatorType,
        scene_name: sceneName
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async stopSimulation() {
    const response = await fetch(`${this.baseURL}/simulation/stop`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async resetSimulation() {
    const response = await fetch(`${this.baseURL}/simulation/reset`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async loadScene(sceneName) {
    const response = await fetch(`${this.baseURL}/simulation/load_scene`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        scene_name: sceneName
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async spawnRobot(robotModel, position = [0, 0, 0], name = null) {
    const response = await fetch(`${this.baseURL}/simulation/spawn_robot`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        robot_model: robotModel,
        position,
        name
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Health check
  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

// Create a singleton instance
const roboticsApiClient = new RoboticsApiClient();

// Export for use in components
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { RoboticsApiClient, roboticsApiClient };
} else if (typeof window !== 'undefined') {
  window.RoboticsApiClient = RoboticsApiClient;
  window.roboticsApiClient = roboticsApiClient;
}