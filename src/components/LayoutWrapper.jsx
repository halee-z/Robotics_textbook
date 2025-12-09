import React from 'react';
import Layout from '@theme/Layout';
import ChatbotToggle from '../components/ChatbotToggle';

// Layout wrapper that includes the chatbot toggle
const LayoutWrapper = (props) => {
  return (
    <>
      <Layout {...props}>
        {props.children}
        <ChatbotToggle />
      </Layout>
    </>
  );
};

export default LayoutWrapper;