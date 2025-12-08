# Educational AI & Humanoid Robotics Platform

An AI-Native Textbook Platform for Advanced Robotics Education

## Overview

This repository contains the Educational AI & Humanoid Robotics Platform - a comprehensive learning system that combines an interactive textbook with specialized AI agents for teaching humanoid robotics concepts. The platform covers ROS 2, Vision-Language Models, simulation environments, and humanoid robotics control.

## Key Features

- **Interactive Docusaurus-based textbook** with comprehensive content on humanoid robotics
- **Specialized AI agents** for different robotics domains (ROS 2, VLMs, Simulation, Control)
- **Simulation-integrated learning** with Gazebo, Isaac Sim, and Unity support
- **Educational exercises and projects** for hands-on learning
- **Real-time robot monitoring and control interface**
- **Retrieval-Augmented Generation (RAG) system** for contextual learning

## Architecture

The platform uses a microservices architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Frontend      │◄──►│   Backend API    │◄──►│   Subagents      │
│   (React/Docusaurus)│ │   (FastAPI)      │    │                  │
└─────────────────┘    └──────────────────┘    ├──────────────────┤
                                              │• ROS 2 Agent    │
                                              │• VLM Agent      │
                                              │• Simulation Agent│
                                              │• Writer Agent   │
                                              │• Knowledge RAG  │
                                              └──────────────────┘
```

## Educational Content

The textbook covers:
1. **ROS 2 Fundamentals** for humanoid robotics
2. **Vision-Language Models** in robotic applications  
3. **Simulation Environments** (Gazebo, Isaac Sim, Unity)
4. **Humanoid Robotics** (kinematics, control, walking algorithms)
5. **Human-Robot Interaction** principles
6. **Practical Exercises** and Projects

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/halee-z/educational-ai-humanoid-robotics.git
cd educational-ai-humanoid-robotics
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the documentation/learning platform:
```bash
cd ..
npm install
```

4. Start the backend server:
```bash
cd backend
python start_server.py
```

5. In a new terminal, start the documentation platform:
```bash  
cd ..
npm start
```

## Project Structure

```
├── docs/                   # Docusaurus textbook content
│   ├── intro/             # Introduction materials
│   ├── ros/              # ROS 2 fundamentals
│   ├── vlm/              # Vision-Language Models
│   ├── simulation/       # Simulation environments  
│   ├── humanoid-robotics/ # Humanoid robotics concepts
│   ├── exercises/        # Practice exercises
│   └── projects/         # Project materials
├── backend/              # FastAPI backend services
│   ├── agents/          # Specialized AI agents
│   ├── api/             # API endpoints
│   ├── rag/             # Knowledge RAG system
│   └── embeddings/      # Embedding processing
├── frontend/             # React components (if applicable)
├── docusaurus.config.js # Documentation configuration
├── sidebars.js          # Navigation structure
└── README.md            # This file
```

## Educational Philosophy

This platform embodies the principles of AI-native education:
- Interactive learning through specialized AI agents
- Simulation-based safety-first approach
- Project-based learning methodology
- Cross-domain integration (ROS 2 + AI + Control + Simulation)

## Contributing

We welcome contributions to improve the platform! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [Docusaurus](https://docusaurus.io/) for documentation
- [FastAPI](https://fastapi.tiangolo.com/) for backend API
- [ROS 2](https://docs.ros.org/en/humble/) for robotics framework
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) for simulation
- [OpenAI](https://openai.com/) for language model integration

---

**Educational AI & Humanoid Robotics Platform** - Bridging the gap between theoretical knowledge and practical implementation in humanoid robotics education.