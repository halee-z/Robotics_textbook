import React from 'react';
import Layout from '@theme/Layout';
import QuizAgent from '@site/src/components/quiz-agent/QuizAgent';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

function QuizPage() {
  const { siteConfig } = useDocusaurusContext();
  
  return (
    <Layout
      title={`Interactive Quiz Agent`}
      description="An AI-powered quiz system for learning about AI and humanoid robotics">
      <main>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--8 col--offset-2">
              <header className="hero hero--primary">
                <div className="container">
                  <h1 className="hero__title">Interactive Quiz Agent</h1>
                  <p className="hero__subtitle">Test your knowledge with our AI-powered quizzes</p>
                </div>
              </header>
              
              <section className="margin-vert--lg">
                <QuizAgent />
              </section>
              
              <section className="margin-vert--lg">
                <div className="text--center padding-horiz--md">
                  <h2>How It Works</h2>
                  <p>
                    Our Quiz Agent creates personalized quizzes tailored to your knowledge level.
                    It adapts to your performance, offering the perfect challenge to accelerate your learning.
                  </p>
                </div>
                
                <div className="row">
                  <div className="col col--4">
                    <h3>Select Topic</h3>
                    <p>Choose from ROS 2, Vision-Language Models, Humanoid Robotics, and more.</p>
                  </div>
                  
                  <div className="col col--4">
                    <h3>Answer Questions</h3>
                    <p>Respond to MCQs, True/False, and short-answer questions with instant feedback.</p>
                  </div>
                  
                  <div className="col col--4">
                    <h3>Review Results</h3>
                    <p>Get a detailed breakdown of your answers with explanations and improvement tips.</p>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default QuizPage;