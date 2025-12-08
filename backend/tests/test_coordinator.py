import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from agents.coordinator import SubagentCoordinator
from api.config import settings


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.qdrant_url = "http://localhost:6333"
    config.openai_api_key = "test-key"
    config.embedding_model = "text-embedding-3-small"
    config.embedding_dimensions = 1536
    config.ros2_domain_id = 0
    config.ros2_node_name = "test_node"
    return config


@pytest.fixture
def coordinator(mock_config):
    with patch('agents.coordinator.RoboticsKnowledgeRAG'), \
         patch('agents.coordinator.ROS2Subagent'), \
         patch('agents.coordinator.VLMAgent'), \
         patch('agents.coordinator.SimulationAgent'), \
         patch('agents.coordinator.WriterAgent'):
        coord = SubagentCoordinator(mock_config)
        yield coord


@pytest.mark.asyncio
async def test_coordinator_initialization(coordinator):
    """Test that the coordinator initializes without errors"""
    with patch.object(coordinator, '_initialize_agents_mocks'):
        result = await coordinator.initialize_agents()
        assert result is True


@pytest.mark.asyncio
async def test_route_request_unknown_agent(coordinator):
    """Test routing to unknown agent type"""
    from fastapi import HTTPException
    
    with pytest.raises(HTTPException) as exc_info:
        await coordinator.route_request("unknown_agent", "test_action", {})
    
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_route_request_ros2_agent(coordinator):
    """Test routing to ROS2 agent"""
    from agents.ros2_subagent import RobotStatus
    
    # Mock the ROS2 agent and its methods
    mock_ros2_agent = AsyncMock()
    mock_ros2_agent.get_robot_status = AsyncMock(return_value=RobotStatus())
    coordinator.ros2_agent = mock_ros2_agent
    
    # Test getting robot status
    result = await coordinator.route_request("ros2", "get_status", {})
    assert "status" in result


@pytest.mark.asyncio
async def test_route_request_vlm_agent(coordinator):
    """Test routing to VLM agent"""
    from agents.vlm_agent import VLMResult
    
    # Mock the VLM agent and its methods
    mock_vlm_agent = AsyncMock()
    mock_vlm_agent.process_image = AsyncMock(return_value=VLMResult(labels=[], scores=[]))
    coordinator.vlm_agent = mock_vlm_agent
    
    # Test image processing
    result = await coordinator.route_request("vlm", "process_image", {"image_path": "test.jpg"})
    assert "result" in result


@pytest.mark.asyncio
async def test_route_request_simulation_agent(coordinator):
    """Test routing to simulation agent"""
    # Mock the simulation agent and its methods
    mock_sim_agent = AsyncMock()
    mock_sim_agent.start_simulation = AsyncMock(return_value=True)
    coordinator.simulation_agent = mock_sim_agent
    
    # Test starting simulation
    result = await coordinator.route_request("simulation", "start_simulation", {
        "simulator_type": "gazebo",
        "scene_name": "default"
    })
    assert "success" in result


@pytest.mark.asyncio
async def test_route_request_writer_agent(coordinator):
    """Test routing to writer agent"""
    from agents.writer_agent import EducationalContent
    
    # Mock the writer agent and its methods
    mock_writer_agent = AsyncMock()
    mock_writer_agent.generate_explanation = AsyncMock(return_value="Test explanation")
    coordinator.writer_agent = mock_writer_agent
    
    # Test explanation generation
    result = await coordinator.route_request("writer", "generate_explanation", {
        "topic": "ROS 2",
        "difficulty": "beginner",
        "length": "medium"
    })
    assert "explanation" in result
    assert result["explanation"] == "Test explanation"


@pytest.mark.asyncio
async def test_route_request_knowledge_agent(coordinator):
    """Test routing to knowledge agent"""
    # Mock the knowledge RAG system and its methods
    mock_knowledge_rag = AsyncMock()
    mock_knowledge_rag.retrieve_relevant_context = AsyncMock(return_value=[])
    coordinator.knowledge_rag = mock_knowledge_rag
    
    # Test context retrieval
    result = await coordinator.route_request("knowledge", "retrieve_context", {
        "query": "What is ROS 2?",
        "top_k": 5
    })
    assert "context" in result


def test_get_available_agents(coordinator):
    """Test getting available agents information"""
    result = asyncio.run(coordinator.get_available_agents())
    assert "agents" in result
    assert isinstance(result["agents"], dict)
    
    # Check that all expected agent types are present
    expected_agents = ["ros2", "vlm", "simulation", "writer", "knowledge"]
    for agent_type in expected_agents:
        assert agent_type in result["agents"]