# Book Chatbot Component

## Overview
The Book Chatbot is an AI-powered assistant integrated into the Educational AI & Humanoid Robotics textbook website. It allows readers to ask questions about the book content and receive answers based on the textbook material.

## Features
- Context-aware responses based on book content
- Persistent chat history using localStorage
- Clean, modern UI with responsive design
- Toggle button for easy access from any page
- Clear chat history functionality

## Components

### BookChatbot.jsx
The main chatbot interface with:
- Message display area
- User input field
- Typing indicators
- Header with action buttons

### ChatbotToggle.jsx
A floating button that appears on all pages allowing users to:
- Open/close the chatbot
- Has a pulsing animation to be more noticeable
- Positioned on the bottom right of the screen

### contentService.js
Provides content search and retrieval functionality:
- Processes book content for relevant information
- Implements keyword matching algorithm
- Returns contextually relevant responses

## How It Works
1. The chatbot is integrated into the Docusaurus layout via the Layout override
2. A floating toggle button is displayed on all pages
3. When clicked, the chatbot slides in from the right side
4. User queries are processed against the book content
5. Responses are generated based on relevant content sections
6. Chat history is preserved between sessions using localStorage

## Implementation Details
- Built with React and integrated into Docusaurus
- Uses localStorage for persisting chat history
- Implements a simulated content search (in a real implementation, this would connect to a vector database or similar system)
- Responsive design works on desktop and mobile
- Styled with CSS modules for scoped styling

## Future Enhancements
For a production implementation, consider:

1. **Enhanced Content Search**: 
   - Implement semantic search using embeddings
   - Connect to a vector database
   - Add support for real-time content updates

2. **AI Integration**:
   - Connect to an LLM API for more sophisticated responses
   - Implement citations to specific book sections
   - Add support for multi-turn conversations

3. **Enhanced UI/UX**:
   - Add message timestamps
   - Support for code snippets and images
   - Export chat history functionality

## Files
- `BookChatbot.jsx` - Main chatbot component
- `BookChatbot.module.css` - Styles for the chatbot
- `ChatbotToggle.jsx` - Floating toggle button
- `ChatbotToggle.module.css` - Styles for the toggle button
- `contentService.js` - Content search and retrieval logic
- `Layout.jsx` - Docusaurus layout override to include the chatbot