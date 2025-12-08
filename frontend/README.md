# Educational AI & Humanoid Robotics Frontend

This directory contains the frontend components for the Educational AI & Humanoid Robotics platform. The frontend provides an interactive interface for students and educators to engage with humanoid robotics concepts through:

- Educational AI chat interface
- Robot status and control dashboard
- Simulation environment controls
- Interactive learning materials

## Components

### Chat Interface (`chat/ChatInterface.jsx`)
- Interactive chat with educational AI agents
- Topic-specific experts (ROS 2, VLMs, Simulation, Control)
- Context-aware responses with confidence scoring

### Robot Status (`components/RobotStatus.jsx`)
- Real-time robot status monitoring
- Joint angle visualization
- Battery level indicator
- Quick control buttons

### Simulation Control (`components/SimulationControl.jsx`)
- Scene selection and loading
- Simulation start/stop/reset controls
- Robot spawning in simulation

### Main Page (`pages/MainPage.jsx`)
- Unified interface combining all components
- Tab-based navigation
- Responsive design

## API Integration

The frontend communicates with the backend through the `/api` endpoints. The `api/client.js` file provides a convenient client for all API interactions.

## Styling

CSS styles are located in `components/styles.css` and provide a responsive, accessible interface for educational use.

## Getting Started

1. Ensure the backend server is running
2. Serve the HTML file through a web server (due to CORS restrictions with fetch API)
3. Access the application through your browser

## Dependencies

- React (for UI components)
- Axios (for API calls)
- CSS (for styling)

## Notes

This frontend is designed for educational purposes and demonstrates how to create interactive interfaces for humanoid robotics education. In a production environment, additional security measures and optimizations would be implemented.