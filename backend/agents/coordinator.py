import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException

from .ros2_subagent import ROS2Subagent
from .vlm_agent import VLMAgent
from .simulation_agent import SimulationAgent
from .writer_agent import WriterAgent
from ..rag.knowledge_rag import RoboticsKnowledgeRAG

logger = logging.getLogger(__name__)


class SubagentCoordinator:
    """
    Coordinator for all educational AI subagents in the humanoid robotics system.
    Manages ROS 2, VLM, Simulation, and Writer subagents, providing a unified interface.
    """
    
    def __init__(self, config):
        self.config = config
        self.agents = {}
        
        # Initialize all subagents
        self.ros2_agent = None
        self.vlm_agent = None
        self.simulation_agent = None
        self.writer_agent = None
        self.knowledge_rag = None
        
        logger.info("Initialized Subagent Coordinator")
    
    async def initialize_agents(self):
        """Initialize all subagents"""
        logger.info("Initializing all subagents...")
        
        try:
            # Initialize the knowledge base
            self.knowledge_rag = RoboticsKnowledgeRAG(self.config)
            
            # Initialize ROS 2 agent
            self.ros2_agent = ROS2Subagent(self.config)
            ros2_success = await self.ros2_agent.start()
            if ros2_success:
                logger.info("ROS2 subagent initialized successfully")
            else:
                logger.warning("ROS2 subagent initialization failed or simulated")
            
            # Initialize VLM agent
            self.vlm_agent = VLMAgent(self.config)
            logger.info("VLM agent initialized successfully")
            
            # Initialize Simulation agent
            self.simulation_agent = SimulationAgent(self.config)
            logger.info("Simulation agent initialized successfully")
            
            # Initialize Writer agent
            self.writer_agent = WriterAgent(self.config)
            logger.info("Writer agent initialized successfully")
            
            logger.info("All subagents initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing subagents: {e}")
            return False
    
    async def shutdown_agents(self):
        """Shutdown all subagents"""
        logger.info("Shutting down all subagents...")

        try:
            if self.ros2_agent:
                await self.ros2_agent.stop()

            logger.info("All subagents shut down successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down subagents: {e}")
            return False
    
    async def route_request(self, agent_type: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route requests to appropriate subagents"""
        logger.info(f"Routing request to {agent_type} agent for action: {action}")
        
        try:
            if agent_type == "ros2":
                return await self._handle_ros2_request(action, params)
            elif agent_type == "vlm":
                return await self._handle_vlm_request(action, params)
            elif agent_type == "simulation":
                return await self._handle_simulation_request(action, params)
            elif agent_type == "writer":
                return await self._handle_writer_request(action, params)
            elif agent_type == "knowledge":
                return await self._handle_knowledge_request(action, params)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_type}")
                
        except Exception as e:
            logger.error(f"Error routing request to {agent_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Error in {agent_type} agent: {str(e)}")
    
    async def _handle_ros2_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for ROS 2 subagent"""
        if not self.ros2_agent:
            raise HTTPException(status_code=500, detail="ROS2 agent not initialized")
        
        if action == "get_status":
            status = await self.ros2_agent.get_robot_status()
            return {"status": status.__dict__}
        elif action == "send_command":
            command = params.get("command")
            cmd_params = params.get("params", {})
            success = await self.ros2_agent.send_command(command, cmd_params)
            return {"success": success}
        elif action == "execute_trajectory":
            trajectory = params.get("trajectory")
            success = await self.ros2_agent.execute_trajectory(trajectory)
            return {"success": success}
        elif action == "get_joint_state":
            joint_name = params.get("joint_name")
            state = await self.ros2_agent.get_joint_state(joint_name)
            return {"joint_state": state}
        elif action == "set_control_mode":
            controller = params.get("controller")
            mode = params.get("mode")
            success = await self.ros2_agent.set_control_mode(controller, mode)
            return {"success": success}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown ROS2 action: {action}")
    
    async def _handle_vlm_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for VLM subagent"""
        if not self.vlm_agent:
            raise HTTPException(status_code=500, detail="VLM agent not initialized")
        
        if action == "process_image":
            image_path = params.get("image_path")
            result = await self.vlm_agent.process_image(image_path)
            return {"result": result.__dict__}
        elif action == "image_captioning":
            image_path = params.get("image_path")
            result = await self.vlm_agent.image_captioning(image_path)
            return {"result": result.__dict__}
        elif action == "visual_grounding":
            image_path = params.get("image_path")
            text_query = params.get("text_query")
            result = await self.vlm_agent.visual_grounding(image_path, text_query)
            return {"result": result.__dict__}
        elif action == "similarity_search":
            image_path = params.get("image_path")
            reference_texts = params.get("reference_texts", [])
            result = await self.vlm_agent.similarity_search(image_path, reference_texts)
            return {"result": result}
        elif action == "command_interpretation":
            image_path = params.get("image_path")
            command = params.get("command")
            result = await self.vlm_agent.command_interpretation(image_path, command)
            return {"result": result}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown VLM action: {action}")
    
    async def _handle_simulation_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for Simulation subagent"""
        if not self.simulation_agent:
            raise HTTPException(status_code=500, detail="Simulation agent not initialized")
        
        if action == "start_simulation":
            simulator_type = params.get("simulator_type", "gazebo")
            scene_name = params.get("scene_name", "default")
            success = await self.simulation_agent.start_simulation(simulator_type, scene_name)
            return {"success": success}
        elif action == "stop_simulation":
            success = await self.simulation_agent.stop_simulation()
            return {"success": success}
        elif action == "reset_simulation":
            success = await self.simulation_agent.reset_simulation()
            return {"success": success}
        elif action == "load_scene":
            scene_name = params.get("scene_name")
            success = await self.simulation_agent.load_scene(scene_name)
            return {"success": success}
        elif action == "spawn_robot":
            robot_model = params.get("robot_model")
            position = params.get("position", [0.0, 0.0, 0.0])
            name = params.get("name")
            success = await self.simulation_agent.spawn_robot(robot_model, position, name)
            return {"success": success}
        elif action == "get_robot_state":
            robot_name = params.get("robot_name")
            state = await self.simulation_agent.get_robot_state(robot_name)
            return {"state": state}
        elif action == "set_robot_state":
            robot_name = params.get("robot_name")
            state = params.get("state", {})
            success = await self.simulation_agent.set_robot_state(robot_name, state)
            return {"success": success}
        elif action == "execute_step":
            num_steps = params.get("num_steps", 1)
            success = await self.simulation_agent.execute_simulation_step(num_steps)
            return {"success": success}
        elif action == "run_scenario":
            scenario_config = params.get("scenario_config", {})
            results = await self.simulation_agent.run_simulation_scenario(scenario_config)
            return {"results": results}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown Simulation action: {action}")
    
    async def _handle_writer_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for Writer subagent"""
        if not self.writer_agent:
            raise HTTPException(status_code=500, detail="Writer agent not initialized")
        
        if action == "generate_explanation":
            topic = params.get("topic")
            difficulty = params.get("difficulty", "intermediate")
            length = params.get("length", "medium")
            explanation = await self.writer_agent.generate_explanation(topic, difficulty, length)
            return {"explanation": explanation}
        elif action == "generate_exercise":
            topic = params.get("topic")
            difficulty = params.get("difficulty", "intermediate")
            exercise = await self.writer_agent.generate_exercise(topic, difficulty)
            return {"exercise": exercise.__dict__}
        elif action == "generate_project":
            topic = params.get("topic")
            duration = params.get("duration_weeks", 4)
            project = await self.writer_agent.generate_project(topic, duration)
            return {"project": project.__dict__}
        elif action == "generate_summary":
            topics = params.get("topics", [])
            difficulty = params.get("difficulty", "intermediate")
            summary = await self.writer_agent.generate_summary(topics, difficulty)
            return {"summary": summary}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown Writer action: {action}")
    
    async def _handle_knowledge_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for Knowledge RAG system"""
        if not self.knowledge_rag:
            raise HTTPException(status_code=500, detail="Knowledge RAG not initialized")
        
        if action == "retrieve_context":
            query = params.get("query")
            top_k = params.get("top_k", 5)
            context = await self.knowledge_rag.retrieve_relevant_context(query, top_k)
            return {"context": context}
        elif action == "generate_response":
            query = params.get("query")
            agent_type = params.get("agent_type", "general")
            response = await self.knowledge_rag.generate_response(query, agent_type)
            return {"response": response}
        elif action == "add_documents":
            # This would require a different approach in a real implementation
            # since we can't receive documents directly in this format
            return {"status": "not_implemented", "message": "Add documents requires direct access to document objects"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown Knowledge action: {action}")
    
    async def get_available_agents(self) -> Dict[str, Any]:
        """Get information about available agents"""
        agents_info = {
            "ros2": {
                "name": "ROS 2 Specialist",
                "type": "ros2",
                "capabilities": [
                    "Robot status monitoring",
                    "Command execution",
                    "Trajectory execution",
                    "Joint state management"
                ],
                "status": "available" if self.ros2_agent else "uninitialized"
            },
            "vlm": {
                "name": "Vision-Language Model Specialist",
                "type": "vlm",
                "capabilities": [
                    "Image classification",
                    "Visual grounding",
                    "Image captioning",
                    "Similarity search",
                    "Command interpretation"
                ],
                "status": "available" if self.vlm_agent else "uninitialized"
            },
            "simulation": {
                "name": "Simulation Specialist",
                "type": "simulation",
                "capabilities": [
                    "Gazebo integration",
                    "Isaac Sim integration",
                    "Unity integration",
                    "Scenario execution",
                    "Robot spawning"
                ],
                "status": "available" if self.simulation_agent else "uninitialized"
            },
            "writer": {
                "name": "Educational Content Writer",
                "type": "writer",
                "capabilities": [
                    "Explanation generation",
                    "Exercise creation",
                    "Project outlines",
                    "Content summarization"
                ],
                "status": "available" if self.writer_agent else "uninitialized"
            },
            "knowledge": {
                "name": "Knowledge Base RAG",
                "type": "knowledge",
                "capabilities": [
                    "Information retrieval",
                    "Context-aware responses",
                    "Educational content access"
                ],
                "status": "available" if self.knowledge_rag else "uninitialized"
            }
        }

        return {"agents": agents_info}