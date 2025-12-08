#!/usr/bin/env python3
"""
Startup script for the Educational AI & Humanoid Robotics platform.

This script initializes and runs the backend API server with all subagents.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the backend directory to the path so we can import modules
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from api.main import app
from api.config import settings
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Educational AI & Humanoid Robotics Platform")
    parser.add_argument(
        "--host", 
        type=str, 
        default=settings.api_host,
        help="Host to bind to (default: from config)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=settings.api_port,
        help="Port to bind to (default: from config)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload on code changes (development only)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Educational AI & Humanoid Robotics Backend...")
    print(f"Configuration: {settings.app_name} v{settings.app_version}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Debug mode: {settings.debug}")
    print("-" * 50)
    
    # Ensure temp upload directory exists
    os.makedirs(settings.temp_upload_dir, exist_ok=True)
    print(f"Temp upload directory: {settings.temp_upload_dir}")
    
    # Run the application
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not settings.debug else "debug"
    )


if __name__ == "__main__":
    main()