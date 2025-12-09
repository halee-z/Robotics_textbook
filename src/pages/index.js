import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Textbook - 5 min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="AI-Native Textbook for Advanced Robotics Education">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col">
                <h2>Key Features of this Textbook</h2>
                <p>
                  This AI-Native textbook combines theoretical foundations with practical implementations,
                  focusing on AI techniques applied to humanoid robotics systems.
                </p>
              </div>
            </div>
            
            <div className="row">
              <div className="col col--4">
                <h3>ROS 2 Integration</h3>
                <p>
                  Comprehensive coverage of ROS 2 fundamentals specifically for humanoid robotics applications.
                </p>
              </div>
              
              <div className="col col--4">
                <h3>Vision-Language Models</h3>
                <p>
                  Modern approaches to combining visual perception with linguistic understanding in robotics.
                </p>
              </div>
              
              <div className="col col--4">
                <h3>Human-Robot Interaction</h3>
                <p>
                  Principles of effective communication between humans and humanoid robots in educational settings.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}