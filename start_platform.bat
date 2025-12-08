@echo off
REM Startup script for Educational AI & Humanoid Robotics Platform (Windows)

echo Starting Educational AI & Humanoid Robotics Platform...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r backend\requirements.txt

REM Start the backend server in a separate window
echo Starting backend server...
start "Backend Server" cmd /c "cd backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

REM Wait a bit for backend to start
timeout /t 5 /nobreak >nul

REM Start the frontend if available
if exist "frontend" (
    echo Starting frontend development server...
    cd frontend
    npm install
    start "Frontend Server" cmd /c "npm start"
    cd ..
)

echo Platform started successfully!
echo Backend API available at: http://localhost:8000
echo Documentation available at: http://localhost:3000 (if Docusaurus is running)

pause