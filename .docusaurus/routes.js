import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '1da'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'a56'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '7b5'),
            routes: [
              {
                path: '/docs/api/overview',
                component: ComponentCreator('/docs/api/overview', '5d7'),
                exact: true
              },
              {
                path: '/docs/exercises/chapter1',
                component: ComponentCreator('/docs/exercises/chapter1', '711'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/exercises/chapter2',
                component: ComponentCreator('/docs/exercises/chapter2', 'c8a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/exercises/chapter3',
                component: ComponentCreator('/docs/exercises/chapter3', '1df'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/humanoid-robotics/control-systems',
                component: ComponentCreator('/docs/humanoid-robotics/control-systems', 'dd3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/humanoid-robotics/human-robot-interaction',
                component: ComponentCreator('/docs/humanoid-robotics/human-robot-interaction', '30b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/humanoid-robotics/introduction',
                component: ComponentCreator('/docs/humanoid-robotics/introduction', '564'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/humanoid-robotics/kinematics',
                component: ComponentCreator('/docs/humanoid-robotics/kinematics', '3fe'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/humanoid-robotics/walking-algorithms',
                component: ComponentCreator('/docs/humanoid-robotics/walking-algorithms', '044'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro/getting-started',
                component: ComponentCreator('/docs/intro/getting-started', 'a76'),
                exact: true
              },
              {
                path: '/docs/projects/project1',
                component: ComponentCreator('/docs/projects/project1', '4e9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/projects/project2',
                component: ComponentCreator('/docs/projects/project2', '410'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/projects/project3',
                component: ComponentCreator('/docs/projects/project3', '040'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros/fundamentals',
                component: ComponentCreator('/docs/ros/fundamentals', '6e0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros/launch-files',
                component: ComponentCreator('/docs/ros/launch-files', 'd4c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros/nodes-topics-services',
                component: ComponentCreator('/docs/ros/nodes-topics-services', '1ed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/ros/packages-workspaces',
                component: ComponentCreator('/docs/ros/packages-workspaces', '2e0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/simulation/gazebo',
                component: ComponentCreator('/docs/simulation/gazebo', 'f39'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/simulation/isaac-sim',
                component: ComponentCreator('/docs/simulation/isaac-sim', 'bd7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/simulation/unity-robotics',
                component: ComponentCreator('/docs/simulation/unity-robotics', 'de1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vlm/embedding-techniques',
                component: ComponentCreator('/docs/vlm/embedding-techniques', '664'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vlm/introduction',
                component: ComponentCreator('/docs/vlm/introduction', 'eb6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vlm/planning-with-vlm',
                component: ComponentCreator('/docs/vlm/planning-with-vlm', '78d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/vlm/vla-architectures',
                component: ComponentCreator('/docs/vlm/vla-architectures', '8c3'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', '2bc'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
