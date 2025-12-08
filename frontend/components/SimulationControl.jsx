import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SimulationControl = () => {
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [availableScenes, setAvailableScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState('default');
  const [simulationStatus, setSimulationStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  // Define available scenes
  useEffect(() => {
    setAvailableScenes([
      { id: 'default', name: 'Default Environment' },
      { id: 'classroom', name: 'Classroom Setting' },
      { id: 'lab', name: 'Robotics Lab' },
      { id: 'home', name: 'Home Environment' },
      { id: 'outdoor', name: 'Outdoor Space' }
    ]);
  }, []);

  const startSimulation = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/simulation/start', {
        simulator_type: 'gazebo',
        scene_name: selectedScene
      });
      
      if (response.data.success) {
        setIsSimulationRunning(true);
        // Update simulation status
        fetchSimulationStatus();
      }
    } catch (error) {
      console.error('Error starting simulation:', error);
    } finally {
      setLoading(false);
    }
  };

  const stopSimulation = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/simulation/stop');
      
      if (response.data.success) {
        setIsSimulationRunning(false);
      }
    } catch (error) {
      console.error('Error stopping simulation:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetSimulation = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/simulation/reset');
      
      if (response.data.success) {
        // Update simulation status
        fetchSimulationStatus();
      }
    } catch (error) {
      console.error('Error resetting simulation:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSimulationStatus = async () => {
    try {
      // In a real implementation, we'd have an endpoint for this
      // For now, we'll just update the local status
      setSimulationStatus({
        running: isSimulationRunning,
        scene: selectedScene,
        time: new Date().toLocaleTimeString(),
        robots: 1
      });
    } catch (error) {
      console.error('Error fetching simulation status:', error);
    }
  };

  const loadScene = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/simulation/load_scene', {
        scene_name: selectedScene
      });
      
      if (response.data.success) {
        fetchSimulationStatus();
      }
    } catch (error) {
      console.error('Error loading scene:', error);
    } finally {
      setLoading(false);
    }
  };

  const spawnRobot = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/simulation/spawn_robot', {
        robot_model: 'humanoid_a',
        position: [0, 0, 0],
        name: 'student_robot'
      });
      
      if (response.data.success) {
        fetchSimulationStatus();
      }
    } catch (error) {
      console.error('Error spawning robot:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="simulation-control">
      <h2>Simulation Environment Control</h2>
      
      <div className="simulation-actions">
        <div className="scene-selector">
          <label htmlFor="scene-select">Select Scene:</label>
          <select
            id="scene-select"
            value={selectedScene}
            onChange={(e) => setSelectedScene(e.target.value)}
            disabled={isSimulationRunning}
          >
            {availableScenes.map(scene => (
              <option key={scene.id} value={scene.id}>
                {scene.name}
              </option>
            ))}
          </select>
          <button onClick={loadScene} disabled={loading || !isSimulationRunning}>
            Load Scene
          </button>
        </div>
        
        <div className="simulation-controls">
          <button 
            onClick={startSimulation} 
            disabled={loading || isSimulationRunning}
            className={isSimulationRunning ? 'active' : ''}
          >
            {isSimulationRunning ? 'Running...' : 'Start Simulation'}
          </button>
          
          <button 
            onClick={stopSimulation} 
            disabled={loading || !isSimulationRunning}
          >
            Stop Simulation
          </button>
          
          <button 
            onClick={resetSimulation} 
            disabled={loading || !isSimulationRunning}
          >
            Reset Simulation
          </button>
        </div>
        
        <button 
          onClick={spawnRobot} 
          disabled={loading || !isSimulationRunning}
        >
          Spawn Robot
        </button>
      </div>
      
      <div className="simulation-status">
        <h3>Simulation Status</h3>
        {simulationStatus ? (
          <div className="status-details">
            <p><strong>Running:</strong> {simulationStatus.running ? 'Yes' : 'No'}</p>
            <p><strong>Scene:</strong> {simulationStatus.scene}</p>
            <p><strong>Time:</strong> {simulationStatus.time}</p>
            <p><strong>Active Robots:</strong> {simulationStatus.robots}</p>
          </div>
        ) : (
          <p>Simulation not started</p>
        )}
      </div>
      
      <div className="simulation-info">
        <h3>About Simulation Environments</h3>
        <p>
          Simulation environments allow safe testing of humanoid robot behaviors before deploying 
          on actual hardware. This educational platform supports multiple simulation environments 
          including Gazebo, Isaac Sim, and Unity Robotics.
        </p>
        
        <h4>Key Features:</h4>
        <ul>
          <li>Physics simulation with realistic dynamics</li>
          <li>Sensor simulation for cameras, IMUs, and other sensors</li>
          <li>Environment customization</li>
          <li>Scenario setup and testing</li>
          <li>Integration with ROS 2 for seamless transition to hardware</li>
        </ul>
      </div>
    </div>
  );
};

export default SimulationControl;