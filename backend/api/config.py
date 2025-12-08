from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application settings
    app_name: str = "Educational AI & Humanoid Robotics Backend"
    app_version: str = "1.0.0"
    debug: bool = False

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database settings (for session storage, embeddings, etc.)
    database_url: str = "postgresql://user:password@localhost/robotics_db"

    # Vector database settings (for RAG)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "robotics_knowledge"

    # OpenAI settings (for educational AI agents)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo"

    # ROS 2 settings
    ros2_domain_id: int = 0
    ros2_node_name: str = "educational_backend"

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # File upload settings
    temp_upload_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "temp_uploads")

    class Config:
        env_file = ".env"


# Create settings instance
settings = Settings()