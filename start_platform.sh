#!/bin/bash
# Startup script for Educational AI & Humanoid Robotics Platform

echo "Starting Educational AI & Humanoid Robotics Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r backend/requirements.txt

# Start the backend server
echo "Starting backend server..."
cd backend
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Wait a bit for backend to start
sleep 5

# Start the frontend (if available)
if [ -d "../frontend" ]; then
    echo "Starting frontend development server..."
    cd ../frontend
    npm install
    nohup npm start > frontend.log 2>&1 &
fi

echo "Platform started successfully!"
echo "Backend API available at: http://localhost:8000"
echo "Documentation available at: http://localhost:3000 (if Docusaurus is running)"

# Keep the script running
wait