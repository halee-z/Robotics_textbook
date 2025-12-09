import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotToggle from '../components/ChatbotToggle';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>{props.children}</OriginalLayout>
      <ChatbotToggle />
    </>
  );
}