import React from 'react';
import clsx from 'clsx';

const FeatureList = [
  {
    title: 'Easy to Learn',
    description: (
      <>
        Our textbook provides a comprehensive introduction to Educational AI and Humanoid Robotics
        with clear explanations and practical examples.
      </>
    ),
  },
  {
    title: 'Focus on Modern Techniques',
    description: (
      <>
        We cover the latest developments in Vision-Language Models, simulation environments,
        and AI techniques for robotics.
      </>
    ),
  },
  {
    title: 'Powered by Docusaurus',
    description: (
      <>
        Built using Docusaurus, this textbook offers a modern, responsive interface
        with an integrated chatbot for learning assistance.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className="features">
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}