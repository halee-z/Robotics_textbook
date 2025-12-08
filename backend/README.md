# Educational AI & Humanoid Robotics Backend

This directory contains the backend infrastructure for the Educational AI & Humanoid Robotics platform. The backend provides:

- FastAPI web server with REST API endpoints
- Integration with multiple AI subagents (ROS 2, VLM, Simulation, Writer)
- RAG (Retrieval-Augmented Generation) system for educational content
- Interfaces to robotics simulation environments
- Robot control capabilities

## Components

### API Server (`api/main.py`)
- Main FastAPI application
- Endpoints for chat, robot control, and simulation
- Integration with subagent coordinator

### Subagents
- **ROS 2 Agent**: Interfaces with ROS 2 systems for robot control
- **VLM Agent**: Vision-Language Model processing for perception
- **Simulation Agent**: Controls simulation environments (Gazebo, Isaac Sim, Unity)
- **Writer Agent**: Generates educational content and explanations
- **RAG System**: Knowledge base with retrieval capabilities

### Configuration (`api/config.py`)
- Application settings and configuration
- Database URLs and credentials
- API keys and model settings

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables (copy `.env.example` to `.env` and update values)

3. Start the server:
   ```bash
   python start_server.py
   ```

## API Endpoints

### Chat & Educational AI
- `POST /chat` - Send messages to educational AI agents
- `GET /agents/available` - List available agents
- `GET /sessions/{session_id}` - Retrieve chat session history

### Robot Control
- `GET /robot/status` - Get current robot status
- `POST /robot/control` - Send control commands to robot

### Simulation Control
- `POST /simulation/start` - Start simulation environment
- `POST /simulation/stop` - Stop simulation
- `POST /simulation/reset` - Reset simulation
- `POST /simulation/load_scene` - Load a scene
- `POST /simulation/spawn_robot` - Spawn a robot in simulation

### Health Check
- `GET /health` - Check server health
- `GET /` - Root endpoint

## Architecture

The system uses a subagent coordinator pattern where specialized agents handle different aspects of humanoid robotics education:

1. The coordinator manages all subagents and routes requests appropriately
2. Each subagent focuses on a specific domain (ROS 2, VLM, Simulation, etc.)
3. The RAG system provides knowledge retrieval for educational responses
4. The API layer provides a unified interface for frontend applications

## Running in Development

For development with auto-reload:

```bash
python start_server.py --reload
```

To specify a different host/port:

```bash
python start_server.py --host 0.0.0.0 --port 8001
```

## Dependencies

- Python 3.8+
- FastAPI
- Pydantic
- Qdrant (for vector database)
- LangChain
- OpenAI library
- ROS 2 Python libraries (rclpy) - for real robot integration

Note: This backend simulates ROS 2 functionality. For real robot integration, ROS 2 with Python libraries must be installed in the environment.