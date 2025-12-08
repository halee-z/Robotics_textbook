from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the subagent coordinator
from agents.coordinator import SubagentCoordinator
from api.config import settings

# Import extended route modules
from api.extended_routes import vlm_app, knowledge_app, writer_app

# Global variable to hold subagent coordinator (will be initialized on startup)
subagent_coordinator = None

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    agent_type: Optional[str] = Field("general", description="Type of agent to handle the request")


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response: str = Field(..., description="The agent's response")
    session_id: str = Field(..., description="Session identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the response")
    sources: Optional[List[str]] = Field(None, description="Sources referenced in the response")


class RobotControlRequest(BaseModel):
    """Request model for robot control commands"""
    command: str = Field(..., description="The command to execute on the robot")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the command")


class RobotStatus(BaseModel):
    """Model for reporting robot status"""
    is_connected: bool = Field(..., description="Whether the robot is connected")
    joint_angles: Optional[Dict[str, float]] = Field(None, description="Current joint angles")
    battery_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Battery level as percentage")
    active_agents: List[str] = Field(..., description="Currently active subagents")


class SimulationStartRequest(BaseModel):
    """Request model for starting simulation"""
    simulator_type: str = Field(..., description="Type of simulator (gazebo, isaac-sim, unity)")
    scene_name: str = Field(..., description="Name of the scene to load")


class SimulationLoadSceneRequest(BaseModel):
    """Request model for loading a scene in simulation"""
    scene_name: str = Field(..., description="Name of the scene to load")


class SimulationSpawnRobotRequest(BaseModel):
    """Request model for spawning a robot in simulation"""
    robot_model: str = Field(..., description="Model of the robot to spawn")
    position: List[float] = Field(..., description="Position [x, y, z] to spawn the robot", max_items=3, min_items=3)
    name: Optional[str] = Field(None, description="Name for the spawned robot")


# Store for chat sessions
chat_sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Educational AI & Humanoid Robotics backend...")

    # Initialize subagent coordinator
    global subagent_coordinator
    subagent_coordinator = SubagentCoordinator(settings)
    success = await subagent_coordinator.initialize_agents()

    if success:
        logger.info("All subagents initialized successfully")
    else:
        logger.error("Error initializing subagents")

    yield

    # Shutdown
    logger.info("Shutting down backend...")
    if subagent_coordinator:
        await subagent_coordinator.shutdown_agents()


# Create FastAPI app
app = FastAPI(
    title="Educational AI & Humanoid Robotics API",
    description="Backend API for the Educational AI & Humanoid Robotics textbook system",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the extended API modules
app.mount("/api/vlm", vlm_app)
app.mount("/api/knowledge", knowledge_app)
app.mount("/api/writer", writer_app)


@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "Educational AI & Humanoid Robotics Backend is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Educational AI & Humanoid Robotics Backend",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint for chat interactions with educational AI agents"""
    logger.info(f"Received chat request: {request.message}")
    
    # Create or retrieve session
    if request.session_id:
        session_id = request.session_id
    else:
        session_id = f"session_{len(chat_sessions) + 1}"
    
    # Initialize session if not exists
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # In a real implementation, we would connect to OpenAI or another LLM service
    # For now, we'll simulate a response based on the agent type
    if request.agent_type == "ros":
        response = f"I am the ROS 2 specialist. Regarding '{request.message}', I can help explain ROS 2 concepts, nodes, topics, and services for humanoid robotics."
        confidence = 0.9
    elif request.agent_type == "vlm":
        response = f"I am the Vision-Language Model specialist. About '{request.message}', I can explain how VLMs are integrated with humanoid robot perception."
        confidence = 0.85
    elif request.agent_type == "simulation":
        response = f"I am the Simulation specialist. Concerning '{request.message}', I can guide you on using Gazebo, Isaac Sim, or Unity for humanoid robot simulation."
        confidence = 0.9
    elif request.agent_type == "control":
        response = f"I am the Control Systems specialist. Regarding '{request.message}', I can explain humanoid robot kinematics, dynamics, and control algorithms."
        confidence = 0.88
    else:
        response = f"I'm an educational AI assistant. You asked: '{request.message}'. I can help with ROS 2, VLMs, simulation, or control systems for humanoid robots."
        confidence = 0.8
    
    # Store in session
    chat_sessions[session_id].append({
        "user": request.message,
        "assistant": response,
        "timestamp": asyncio.get_event_loop().time()
    })
    
    # Return response
    return ChatResponse(
        response=response,
        session_id=session_id,
        confidence=confidence,
        sources=["robotics_textbook_chapter_1", "ros_documentation"]
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve chat session history"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "messages": chat_sessions[session_id]}


@app.get("/robot/status")
async def get_robot_status():
    """Get current status of the humanoid robot"""
    # In a real implementation, this would query the actual robot
    # through ROS 2 topics/services
    status = RobotStatus(
        is_connected=True,  # Simulated connection
        joint_angles={
            "left_hip": 0.1,
            "right_hip": 0.1,
            "left_knee": 0.5,
            "right_knee": 0.5,
            "left_shoulder": 0.2,
            "right_shoulder": 0.2
        },
        battery_level=0.85,
        active_agents=["ros2_agent", "vlm_agent", "control_agent"]
    )
    
    return status


@app.post("/robot/control")
async def send_robot_control(request: RobotControlRequest):
    """Send control commands to the humanoid robot"""
    logger.info(f"Received robot control request: {request.command}")
    
    # In a real implementation, this would publish to ROS 2 topics
    # or call ROS 2 services to control the actual robot
    # For now, we'll simulate success
    
    if request.command == "walk_forward":
        # In real implementation: publish to navigation topic
        response = {"status": "success", "message": "Robot moving forward"}
    elif request.command == "wave_hand":
        # In real implementation: publish to manipulator trajectory topic
        response = {"status": "success", "message": "Robot waving hand"}
    elif request.command == "balance":
        # In real implementation: call balance service
        response = {"status": "success", "message": "Robot balancing"}
    elif request.command == "speak":
        # In real implementation: call speech service
        response = {"status": "success", "message": "Robot speaking"}
    else:
        response = {"status": "unknown_command", "message": f"Unknown command: {request.command}"}
    
    return response


@app.get("/agents/available")
async def get_available_agents():
    """List available educational AI agents"""
    if subagent_coordinator:
        return await subagent_coordinator.get_available_agents()
    else:
        # Fallback response if coordinator not initialized
        agents = [
            {
                "name": "ROS 2 Specialist",
                "type": "ros",
                "description": "Expert in Robot Operating System 2 concepts and implementation",
                "status": "uninitialized"
            },
            {
                "name": "Vision-Language Model Specialist",
                "type": "vlm",
                "description": "Expert in VLM integration with robotic perception and planning",
                "status": "uninitialized"
            },
            {
                "name": "Simulation Specialist",
                "type": "simulation",
                "description": "Expert in Gazebo, Isaac Sim, and Unity simulation environments",
                "status": "uninitialized"
            },
            {
                "name": "Control Systems Specialist",
                "type": "control",
                "description": "Expert in humanoid robot kinematics, dynamics, and control",
                "status": "uninitialized"
            },
            {
                "name": "Knowledge Base RAG",
                "type": "knowledge",
                "description": "Retrieval-Augmented Generation for educational content",
                "status": "uninitialized"
            }
        ]

        return {"agents": agents}


@app.post("/simulation/start")
async def start_simulation(request: SimulationStartRequest):
    """Start a simulation environment"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")

    try:
        result = await subagent_coordinator.route_request(
            "simulation",
            "start_simulation",
            {
                "simulator_type": request.simulator_type,
                "scene_name": request.scene_name
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop the active simulation"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")

    try:
        result = await subagent_coordinator.route_request(
            "simulation",
            "stop_simulation",
            {}
        )
        return result
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping simulation: {str(e)}")


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset the simulation to its initial state"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")

    try:
        result = await subagent_coordinator.route_request(
            "simulation",
            "reset_simulation",
            {}
        )
        return result
    except Exception as e:
        logger.error(f"Error resetting simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting simulation: {str(e)}")


@app.post("/simulation/load_scene")
async def load_simulation_scene(request: SimulationLoadSceneRequest):
    """Load a scene in the simulation environment"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")

    try:
        result = await subagent_coordinator.route_request(
            "simulation",
            "load_scene",
            {"scene_name": request.scene_name}
        )
        return result
    except Exception as e:
        logger.error(f"Error loading scene: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading scene: {str(e)}")


@app.post("/simulation/spawn_robot")
async def spawn_robot_in_simulation(request: SimulationSpawnRobotRequest):
    """Spawn a robot in the simulation environment"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")

    try:
        result = await subagent_coordinator.route_request(
            "simulation",
            "spawn_robot",
            {
                "robot_model": request.robot_model,
                "position": request.position,
                "name": request.name
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error spawning robot: {e}")
        raise HTTPException(status_code=500, detail=f"Error spawning robot: {str(e)}")


@app.post("/simulation/robot_state")
async def get_robot_simulation_state(robot_name: str):
    """Get the state of a robot in simulation"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")

    try:
        result = await subagent_coordinator.route_request(
            "simulation",
            "get_robot_state",
            {"robot_name": robot_name}
        )
        return result
    except Exception as e:
        logger.error(f"Error getting robot state: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting robot state: {str(e)}")


# Example of how a ROS 2 node would be integrated (commented for now since we don't have rclpy)
"""
class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('educational_backend_controller')
        
        # Publishers and subscribers would be created here
        self.joint_state_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.status_sub = self.create_subscription(RobotStatus, '/robot_status', self.status_callback, 10)
        
        # Services for robot control
        self.control_service = self.create_service(RobotCommand, '/execute_command', self.execute_command_callback)
        
        self.get_logger().info('Educational Backend Controller node initialized')
    
    def status_callback(self, msg):
        # Handle robot status updates
        pass
    
    def execute_command_callback(self, request, response):
        # Execute robot command
        self.get_logger().info(f'Received command: {request.command}')
        response.success = True
        response.message = f'Command {request.command} executed'
        return response
    
    def start_execution(self):
        # Start spinning the node
        self.get_logger().info('Starting execution...')
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)