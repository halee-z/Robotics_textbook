import React, { useState } from 'react';
import BookChatbot from './BookChatbot';
import styles from './ChatbotToggle.module.css';

const ChatbotToggle = () => {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);

  const toggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <>
      {isChatbotOpen && (
        <BookChatbot 
          isOpen={isChatbotOpen} 
          onClose={() => setIsChatbotOpen(false)} 
        />
      )}
      <div className={styles.chatbotToggleButton} onClick={toggleChatbot}>
        <div className={styles.icon}>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M4.913 2.658c2.075-1.932 5.419-1.932 7.494 0 2.075 1.933 2.075 5.072 0 7.005-2.075 1.933-5.419 1.933-7.494 0-2.075-1.932-2.075-5.072 0-7.005zm8.41 1.271a3.02 3.02 0 0 1 4.218 0 3.02 3.02 0 0 1 0 4.274 3.02 3.02 0 0 1-4.218 0 3.02 3.02 0 0 1 0-4.274zM16.771 8.572l3.236-3.153 1.428 1.46-3.236 3.153-1.428-1.46z"></path>
            <path d="M4.97 12.292c-.39-.389-.39-.986 0-1.374.39-.389 1.02-.389 1.41 0l5.282 5.216c.39.389.39.986 0 1.374-.39.389-1.02.389-1.41 0l-5.282-5.216zm0 4.274c-.39.389-1.02.389-1.41 0-.39-.389-.39-.986 0-1.374l5.282-5.216c.39-.389 1.02-.389 1.41 0 .39.389.39.986 0 1.374l-5.282 5.216z"></path>
          </svg>
        </div>
      </div>
    </>
  );
};

export default ChatbotToggle;