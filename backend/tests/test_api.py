import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api.main import app
from api.config import settings

client = TestClient(app)

@pytest.fixture
def mock_subagent_coordinator():
    with patch('api.main.subagent_coordinator') as mock:
        yield mock

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_chat_endpoint_basic(mock_subagent_coordinator):
    # Test the chat endpoint with basic parameters
    response = client.post("/chat", json={
        "message": "Hello, can you explain ROS 2?",
        "agent_type": "ros"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data
    assert "confidence" in data
    assert data["confidence"] > 0.0
    assert "sources" in data

def test_get_robot_status(mock_subagent_coordinator):
    """Test the robot status endpoint"""
    response = client.get("/robot/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "is_connected" in data
    assert "joint_angles" in data
    assert "battery_level" in data
    assert "active_agents" in data

def test_send_robot_control(mock_subagent_coordinator):
    """Test sending robot control commands"""
    response = client.post("/robot/control", json={
        "command": "wave_hand",
        "parameters": {}
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"

def test_get_available_agents(mock_subagent_coordinator):
    """Test getting available agents"""
    response = client.get("/agents/available")
    assert response.status_code == 200
    
    data = response.json()
    assert "agents" in data
    assert isinstance(data["agents"], list)
    
    agent_types = [agent["type"] for agent in data["agents"]]
    assert "ros" in agent_types
    assert "vlm" in agent_types
    assert "simulation" in agent_types
    assert "control" in agent_types
    assert "knowledge" in agent_types

def test_simulation_endpoints(mock_subagent_coordinator):
    """Test simulation-related endpoints"""
    # Test start simulation
    response = client.post("/simulation/start", json={
        "simulator_type": "gazebo",
        "scene_name": "default"
    })
    assert response.status_code == 200
    
    # Test stop simulation
    response = client.post("/simulation/stop")
    assert response.status_code == 200
    
    # Test load scene
    response = client.post("/simulation/load_scene", json={
        "scene_name": "classroom"
    })
    assert response.status_code == 200
    
    # Test spawn robot
    response = client.post("/simulation/spawn_robot", json={
        "robot_model": "humanoid_a",
        "position": [0.0, 0.0, 0.0],
        "name": "student_robot"
    })
    assert response.status_code == 200

def test_session_management(mock_subagent_coordinator):
    """Test session management functionality"""
    # Create a session by sending a chat message
    response = client.post("/chat", json={
        "message": "Hello",
        "agent_type": "general"
    })
    assert response.status_code == 200
    
    session_id = response.json()["session_id"]
    assert session_id is not None
    
    # Retrieve the session
    response = client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["session_id"] == session_id
    assert "messages" in data