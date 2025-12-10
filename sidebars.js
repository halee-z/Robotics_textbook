// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      items: [
        'ros/fundamentals',
        'ros/nodes-topics-services',
        'ros/launch-files',
        'ros/packages-workspaces'
      ],
    },
    {
      type: 'category',
      label: 'Vision-Language Models in Robotics',
      items: [
        'vlm/introduction',
        'vlm/vla-architectures',
        'vlm/embedding-techniques',
        'vlm/planning-with-vlm'
      ],
    },
    {
      type: 'category',
      label: 'Simulation Environments',
      items: [
        'simulation/gazebo',
        'simulation/isaac-sim',
        'simulation/unity-robotics'
      ],
    },
    {
      type: 'category',
      label: 'Humanoid Robotics',
      items: [
        'humanoid-robotics/introduction',
        'humanoid-robotics/kinematics',
        'humanoid-robotics/control-systems',
        'humanoid-robotics/walking-algorithms',
        'humanoid-robotics/human-robot-interaction'
      ],
    },
    {
      type: 'category',
      label: 'Exercises',
      items: [
        'exercises/chapter1',
        'exercises/chapter2',
        'exercises/chapter3'
      ],
    },
    {
      type: 'category',
      label: 'Projects',
      items: [
        'projects/project1',
        'projects/project2',
        'projects/project3'
      ],
    },
    {
      type: 'category',
      label: 'Interactive Learning Tools',
      items: [
        'interactive-tools/quiz-agent',
        'interactive-tools/quiz-agent-demo',
        'interactive-tools/personalization-agent',
        'interactive-tools/personalization-agent-demo'
      ],
    }
  ],
};

module.exports = sidebars;