---
sidebar_position: 3
---

# API Documentation

The Educational AI & Humanoid Robotics Platform provides a comprehensive API for interacting with educational content, robotics systems, and AI agents. This documentation covers all available endpoints and their usage.

## Base URL

All API endpoints are available under:
```
https://your-deployment-url/api/
```

## Authentication

Most endpoints do not require authentication for educational purposes, but robot control endpoints and user-specific features will require appropriate authentication headers.

## Common Response Format

All API responses follow this format:
```json
{
  "status": "success|error",
  "data": {},
  "message": "Optional description"
}
```

## Endpoints

### Chat & Educational AI

#### `POST /chat`
Send a message to the educational AI system.

**Request Body:**
```json
{
  "message": "Your question or message",
  "session_id": "Optional session identifier",
  "agent_type": "general|ros|vlm|simulation|control"
}
```

**Response:**
```json
{
  "response": "AI response text",
  "session_id": "Session identifier",
  "confidence": 0.85,
  "sources": ["source1", "source2"]
}
```

#### `GET /sessions/{session_id}`
Retrieve a session's message history.

**Response:**
```json
{
  "session_id": "Session identifier",
  "messages": [
    {
      "message_id": "Message id",
      "timestamp": "ISO 8601 timestamp",
      "sender": "user|agent",
      "content": "Message content",
      "agent_type": "Agent type"
    }
  ]
}
```

#### `GET /agents/available`
Get information about available AI agents.

**Response:**
```json
{
  "agents": [
    {
      "name": "Agent name",
      "type": "Agent type",
      "description": "Agent description",
      "status": "available|uninitialized",
      "capabilities": ["capability1", "capability2"]
    }
  ]
}
```

### Robot Control

#### `GET /robot/status`
Get the current status of the humanoid robot.

**Response:**
```json
{
  "is_connected": true,
  "joint_angles": {
    "joint_name": 0.5
  },
  "battery_level": 0.85,
  "active_agents": ["agent1", "agent2"]
}
```

#### `POST /robot/control`
Send a control command to the robot.

**Request Body:**
```json
{
  "command": "walk_forward|wave_hand|balance|speak|move_joints",
  "parameters": {
    "param_name": "param_value"
  }
}
```

**Response:**
```json
{
  "status": "success|unknown_command",
  "message": "Command execution result"
}
```

### Simulation Control

#### `POST /simulation/start`
Start a simulation environment.

**Request Body:**
```json
{
  "simulator_type": "gazebo|isaac-sim|unity",
  "scene_name": "Scene to load"
}
```

**Response:**
```json
{
  "success": true,
  "simulation_id": "Simulation identifier",
  "message": "Status message"
}
```

#### `POST /simulation/spawn_robot`
Spawn a robot in the simulation.

**Request Body:**
```json
{
  "robot_model": "Model to spawn",
  "position": [0.0, 0.0, 0.0],
  "name": "Optional robot name"
}
```

**Response:**
```json
{
  "success": true,
  "robot_id": "Spawned robot identifier",
  "message": "Status message"
}
```

### Vision-Language Models (VLM)

#### `POST /vlm/process_image`
Process an image with the Vision-Language Model.

**Request Body:**
```json
{
  "image_path": "Path to image",
  "top_k": 5
}
```

**Response:**
```json
{
  "result": {
    "labels": ["label1", "label2"],
    "scores": [0.9, 0.8],
    "bounding_boxes": [
      {
        "object": "object_type",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "confidence": 0.9
      }
    ],
    "embeddings": [0.1, 0.2, 0.3, ...]
  }
}
```

#### `POST /vlm/upload_and_process`
Upload and process an image with the VLM.

**Request Body:**
Form data with file attachment.

**Response:**
```json
{
  "result": {
    "labels": ["label1", "label2"],
    "scores": [0.9, 0.8],
    "bounding_boxes": [...]
  }
}
```

### Knowledge Management

#### `POST /knowledge/query`
Query the knowledge base.

**Request Body:**
```json
{
  "query": "Your question",
  "agent_type": "ros|vlm|simulation|control|general",
  "top_k": 5
}
```

**Response:**
```json
{
  "response": {
    "response": "Generated response",
    "sources": ["source1", "source2"],
    "confidence": 0.85
  }
}
```

### Educational Content Writer

#### `POST /writer/explanation`
Generate an explanation for a topic.

**Request Body:**
```json
{
  "topic": "Topic to explain",
  "difficulty": "beginner|intermediate|advanced",
  "length": "short|medium|long"
}
```

**Response:**
```json
{
  "explanation": "Generated explanation text"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error

Error responses follow the format:
```json
{
  "detail": "Error message"
}
```

## Rate Limiting

All endpoints are subject to rate limiting to prevent abuse. Current limits are 100 requests per minute per IP address, with burst capacity of 20 requests.