// Improved content search function
export const searchBookContent = (query) => {
  // This is a simplified approach - in a real implementation, you would:
  // 1. Use a more sophisticated text processing approach
  // 2. Implement semantic search using embeddings
  // 3. Store and retrieve content from an actual index

  // For now, we'll enhance the existing search with more sophisticated matching
  const lowerQuery = query.toLowerCase();
  const terms = lowerQuery.split(/\s+/).filter(term => term.length > 0);

  // In a real implementation, we would search through the actual book content
  // Here we'll simulate searching by checking for relevant terms
  const mockBookContent = [
    {
      title: "ROS 2 Fundamentals for Humanoid Robotics",
      content: `Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. In the context of humanoid robotics, ROS 2 provides the communication infrastructure needed for the complex sensorimotor loops and distributed processing requirements.

      ROS 2 vs ROS 1: Key Differences
      ROS 2 addresses several limitations of the original ROS, making it more suitable for humanoid robotics applications:

      - **Real-Time Support**: Critical for humanoid robot control systems that require deterministic timing
      - **Improved Security**: Essential when humanoid robots interact with humans in educational settings
      - **Better Multi-Robot Support**: Necessary for scenarios involving multiple humanoid robots
      - **Quality of Service (QoS) Settings**: Allows specification of delivery guarantees for different types of data
      - **DDS-Based Communication**: Provides more robust and configurable communication patterns

      Core Concepts:

      Nodes: In ROS 2, a node is a process that performs computation. Each node in a ROS graph can be written in different programming languages (C++, Python, etc.) and can run on different machines. For humanoid robots, common nodes include:

      - Sensor processing nodes (IMU, cameras, LIDAR)
      - Control nodes (walking, manipulation, balance)
      - Perception nodes (object recognition, localization)
      - Planning nodes (motion planning, path planning)

      Topics and Messages: Topics are named buses over which nodes exchange messages. In humanoid robotics, common topics include:

      - /joint_states - Current joint positions, velocities, and efforts
      - /cmd_vel - Velocity commands for base movement
      - /sensor_msgs/Image - Camera image data
      - /tf - Transform data for coordinate frames`
    },
    {
      title: "Vision-Language Models in Robotics",
      content: `Vision-Language Models (VLMs) represent a significant advancement in artificial intelligence, combining visual perception with linguistic understanding. In humanoid robotics, VLMs enable robots to interpret complex visual scenes and respond appropriately using natural language. This integration is especially valuable in educational settings where humanoid robots need to understand and communicate with students about their environment.

      Foundations of Vision-Language Models:

      Transformer Architecture: Most modern VLMs are based on transformer architectures that can process multimodal inputs:

      - **Visual Encoder**: Processes images using convolutional neural networks or vision transformers
      - **Language Encoder**: Processes text using transformer-based language models
      - **Multimodal Fusion**: Combines visual and linguistic representations

      Contrastive Learning: VLMs often use contrastive learning to align visual and textual representations:

      - Training on large datasets of image-text pairs
      - Learning to associate similar concepts across modalities
      - Creating shared embedding spaces for visual and textual information

      VLM Architectures for Robotics:

      CLIP (Contrastive Language-Image Pre-training): CLIP creates a joint embedding space for images and text.

      BLIP (Bootstrapping Language-Image Pre-training): BLIP excels at both understanding and generation tasks:
      - Image captioning
      - Visual question answering
      - Text-guided image retrieval

      Open-Vocabulary Detection Models: Models like Grounding DINO allow detection of objects based on text descriptions:
      - Zero-shot object detection
      - Flexible querying of scene elements
      - Integration with robotic manipulation planning`
    },
    {
      title: "Humanoid Robot Control Systems",
      content: `Humanoid robotics involves creating robots with human-like characteristics and capabilities. Key areas include kinematics (the study of motion), control systems for locomotion and manipulation, and walking algorithms for bipedal movement. Balance and stability control are critical for humanoid robots.

      Kinematics and Dynamics: Kinematics deals with the geometry of motion, while dynamics considers the forces causing motion in humanoid robots.

      Walking and Locomotion Algorithms: Different approaches to achieve stable bipedal walking:
      - Zero Moment Point (ZMP) control
      - Linear Inverted Pendulum Model (LIPM)
      - Capture Point approaches
      - Dynamic walking algorithms

      Control Systems: Modern control systems for humanoid robots incorporate:
      - Feedback control for stability
      - Model Predictive Control (MPC)
      - Adaptive control for changing conditions
      - Learning-based control methods`
    },
    {
      title: "Simulation Environments",
      content: `Simulation environments like Gazebo, Isaac Sim, and Unity Robotics Toolkit are crucial for developing and testing humanoid robots. They provide physics modeling, sensor simulation, and realistic environments to test robot behaviors before deploying on real hardware.

      Gazebo simulation for humanoid robots provides:
      - Physics modeling with realistic dynamics
      - Sensor simulation for cameras, IMUs, and other robot sensors
      - Plugin architecture for extending functionality
      - Integration with ROS 2 for seamless simulation-to-reality transfer

      Isaac Sim and NVIDIA tools offer:
      - Photorealistic rendering capabilities
      - GPU-accelerated physics simulation
      - AI training environments
      - Advanced sensor simulation including ray tracing

      Unity robotics toolkit provides:
      - Game engine-based simulation
      - Cross-platform deployment
      - Advanced graphics capabilities
      - Integration with machine learning frameworks`
    }
  ];

  // Simple search algorithm that finds documents containing the query terms
  const results = [];

  mockBookContent.forEach((doc) => {
    // Check if the document contains any of the query terms
    const docContent = (doc.title + ' ' + doc.content).toLowerCase();
    const matches = terms.filter(term => docContent.includes(term));

    if (matches.length > 0) {
      // Calculate a simple relevance score based on number of matches
      const relevanceScore = matches.length / terms.length;

      // Extract relevant sentences that contain the query terms
      const sentences = doc.content.split(/(?<=[.!?])\s+/);
      const relevantSentences = sentences.filter(sentence =>
        terms.some(term => sentence.toLowerCase().includes(term))
      );

      results.push({
        title: doc.title,
        content: relevantSentences.slice(0, 3).join(' '), // Take up to 3 relevant sentences
        relevance: relevanceScore
      });
    }
  });

  // Sort results by relevance
  results.sort((a, b) => b.relevance - a.relevance);

  // Return the content of the most relevant results
  return results.slice(0, 2).map(result => result.content);
};