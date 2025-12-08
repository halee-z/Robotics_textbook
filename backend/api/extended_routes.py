from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import os
from api.config import settings
from api.main import subagent_coordinator, logger

vlm_app = FastAPI(title="VLM Subsystem API")

class VLMProcessImageRequest(BaseModel):
    """Request model for processing an image"""
    image_path: str = Field(..., description="Path to the image to process")
    top_k: int = Field(5, description="Number of top results to return")


class VLMVisualGroundingRequest(BaseModel):
    """Request model for visual grounding"""
    image_path: str = Field(..., description="Path to the image for visual grounding")
    text_query: str = Field(..., description="Text query to ground in the image")


class VLMSimilaritySearchRequest(BaseModel):
    """Request model for similarity search"""
    image_path: str = Field(..., description="Path to the image for similarity search")
    reference_texts: List[str] = Field(..., description="Reference texts to compare against")


class VLMCommandInterpretationRequest(BaseModel):
    """Request model for command interpretation"""
    image_path: str = Field(..., description="Path to the image for context")
    command: str = Field(..., description="Command to interpret in the context of the image")


@vlm_app.post("/process_image")
async def process_image(request: VLMProcessImageRequest):
    """Process an image using the VLM agent"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "vlm", 
            "process_image", 
            {
                "image_path": request.image_path,
                "top_k": request.top_k
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@vlm_app.post("/image_captioning")
async def image_captioning(request: VLMProcessImageRequest):
    """Generate a caption for an image using the VLM agent"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "vlm", 
            "image_captioning", 
            {
                "image_path": request.image_path
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error generating image caption: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image caption: {str(e)}")


@vlm_app.post("/visual_grounding")
async def visual_grounding(request: VLMVisualGroundingRequest):
    """Perform visual grounding to find objects in an image based on text query"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "vlm", 
            "visual_grounding", 
            {
                "image_path": request.image_path,
                "text_query": request.text_query
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error performing visual grounding: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing visual grounding: {str(e)}")


@vlm_app.post("/similarity_search")
async def similarity_search(request: VLMSimilaritySearchRequest):
    """Perform similarity search between an image and reference texts"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "vlm", 
            "similarity_search", 
            {
                "image_path": request.image_path,
                "reference_texts": request.reference_texts
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {str(e)}")


@vlm_app.post("/command_interpretation")
async def command_interpretation(request: VLMCommandInterpretationRequest):
    """Interpret a natural language command in the context of an image"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "vlm", 
            "command_interpretation", 
            {
                "image_path": request.image_path,
                "command": request.command
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error interpreting command: {e}")
        raise HTTPException(status_code=500, detail=f"Error interpreting command: {str(e)}")


@vlm_app.post("/upload_and_process")
async def upload_and_process_image(file: UploadFile = File(...)):
    """Upload an image and process it with the VLM agent"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(settings.temp_upload_dir, unique_filename)
        
        # Ensure the temp directory exists
        os.makedirs(settings.temp_upload_dir, exist_ok=True)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the image
        result = await subagent_coordinator.route_request(
            "vlm", 
            "process_image", 
            {
                "image_path": file_path
            }
        )
        
        # Clean up the file after processing (optional, depending on requirements)
        # os.remove(file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error uploading and processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading and processing image: {str(e)}")


# Additional routes for knowledge management
knowledge_app = FastAPI(title="Knowledge Management API")

class KnowledgeQueryRequest(BaseModel):
    query: str = Field(..., description="The query to search for in the knowledge base")
    agent_type: str = Field("general", description="The type of agent to use for response generation")
    top_k: int = Field(5, description="Number of top results to return")


class KnowledgeAddDocumentRequest(BaseModel):
    content: str = Field(..., description="The content to add to the knowledge base")
    source: str = Field(..., description="The source of the content")
    tags: List[str] = Field([], description="Tags to associate with the content")


@knowledge_app.post("/query")
async def query_knowledge_base(request: KnowledgeQueryRequest):
    """Query the knowledge base and get a response"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "knowledge",
            "generate_response",
            {
                "query": request.query,
                "agent_type": request.agent_type
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")


@knowledge_app.post("/add_document")
async def add_document_to_knowledge(request: KnowledgeAddDocumentRequest):
    """Add a document to the knowledge base"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        # In a real implementation, this would add the document to the RAG system
        # For now, we'll return a not-implemented response
        result = await subagent_coordinator.route_request(
            "knowledge",
            "add_documents",
            {
                "content": request.content,
                "source": request.source,
                "tags": request.tags
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error adding document to knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding document to knowledge base: {str(e)}")


# Additional routes for writer agent
writer_app = FastAPI(title="Educational Content Writer API")

class GenerateExplanationRequest(BaseModel):
    topic: str = Field(..., description="The topic to generate an explanation for")
    difficulty: str = Field("intermediate", description="The difficulty level (beginner, intermediate, advanced)")
    length: str = Field("medium", description="The length (short, medium, long)")


class GenerateExerciseRequest(BaseModel):
    topic: str = Field(..., description="The topic for the exercise")
    difficulty: str = Field("intermediate", description="The difficulty level")
    learning_objectives: List[str] = Field([], description="Specific learning objectives")


class GenerateProjectRequest(BaseModel):
    topic: str = Field(..., description="The topic for the project")
    duration_weeks: int = Field(4, description="Estimated duration in weeks")
    prerequisites: List[str] = Field([], description="Required prerequisites")


@writer_app.post("/explanation")
async def generate_explanation(request: GenerateExplanationRequest):
    """Generate an explanation for a specific topic"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "writer",
            "generate_explanation",
            {
                "topic": request.topic,
                "difficulty": request.difficulty,
                "length": request.length
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@writer_app.post("/exercise")
async def generate_exercise(request: GenerateExerciseRequest):
    """Generate an exercise for a specific topic"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "writer",
            "generate_exercise",
            {
                "topic": request.topic,
                "difficulty": request.difficulty
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error generating exercise: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating exercise: {str(e)}")


@writer_app.post("/project")
async def generate_project(request: GenerateProjectRequest):
    """Generate a project outline for a specific topic"""
    if not subagent_coordinator:
        raise HTTPException(status_code=500, detail="Subagent coordinator not initialized")
    
    try:
        result = await subagent_coordinator.route_request(
            "writer",
            "generate_project",
            {
                "topic": request.topic,
                "duration_weeks": request.duration_weeks
            }
        )
        return result
    except Exception as e:
        logger.error(f"Error generating project: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating project: {str(e)}")