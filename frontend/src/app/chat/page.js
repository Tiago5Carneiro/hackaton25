"use client";
import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import '../styles/chat.css';

// Sample initial message from the F1 Analyst
const initialMessage = {
  id: 1,
  user: 'F1 Analyst',
  message: "Hello! I'm your F1 Expert Assistant. I can provide analysis on races, drivers, teams, and technical aspects of Formula 1. What would you like to know about?",
  timestamp: 'Just now',
  avatar: 'F',
  isBot: true
};

const suggestionQuestions = [
  "Who do you think will win the next race?",
  "Which team has the best strategy this season?",
  "Can you analyze Max Verstappen's driving style?",
  "What makes the Red Bull car so fast?",
  "How does DRS work in Formula 1?"
];

export default function Chat() {
  const [messages, setMessages] = useState([initialMessage]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  useEffect(() => {
    setIsLoaded(true);
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!newMessage.trim()) return;
    
    const message = {
      id: Date.now(),
      user: 'You',
      message: newMessage,
      timestamp: 'Just now',
      avatar: 'Y',
      isUser: true
    };
    
    setMessages([...messages, message]);
    setNewMessage('');
    
    // Simulate analyst response
    simulateAnalystResponse(newMessage);
  };
  
  const simulateAnalystResponse = (userMessage) => {
    // Show typing indicator
    setIsTyping(true);
    
    // Random delay between 1-3 seconds to simulate thinking
    const delay = Math.floor(Math.random() * 2000) + 1000;
    
    setTimeout(() => {
      setIsTyping(false);
      
      // Generate relevant response based on user message
      let response;
      
      if (userMessage.toLowerCase().includes('win') || userMessage.toLowerCase().includes('champion')) {
        response = "Based on current form, Max Verstappen remains the favorite for race wins, but Charles Leclerc and Lando Norris have shown impressive pace recently. The championship battle is far from over, with Red Bull facing stronger competition from Ferrari and McLaren than in previous seasons.";
      } else if (userMessage.toLowerCase().includes('strategy')) {
        response = "Ferrari has significantly improved their race strategy execution this season. They've addressed previous weaknesses in tire management and timing of pit stops. However, Red Bull still maintains the edge in adapting to changing conditions mid-race, particularly with their outstanding pit crew performance averaging 2.1 second stops.";
      } else if (userMessage.toLowerCase().includes('verstappen') || userMessage.toLowerCase().includes('max')) {
        response = "Max Verstappen's driving style combines exceptional car control with relentless aggression. His key strengths include incredible braking precision, mid-corner adjustments that few other drivers can execute, and exceptional tire management despite his aggressive style. His ability to find performance beyond the car's expected potential has been particularly evident at circuits like Spa-Francorchamps.";
      } else if (userMessage.toLowerCase().includes('red bull')) {
        response = "Red Bull's current dominance stems from their exceptional aerodynamic efficiency. Adrian Newey's design philosophy has created a car with remarkable balance through high and low-speed corners while maintaining excellent straight-line speed. Their underfloor design creates consistent downforce without the porpoising issues that affected other teams early in the ground effect era.";
      } else if (userMessage.toLowerCase().includes('drs')) {
        response = "DRS (Drag Reduction System) works by allowing the driver to open a gap in the rear wing, reducing drag on straights. The system can only be used in designated DRS zones when a driver is within 1 second of the car ahead. When activated, it increases top speed by approximately 10-12 km/h, providing a crucial overtaking advantage while maintaining safety through automatic deactivation when braking.";
      } else {
        response = "That's an interesting question about Formula 1. The sport is constantly evolving, with teams pushing technical boundaries while navigating complex regulations. Analyzing driver performance requires considering both raw speed and consistency over race distances. Do you have a specific aspect you'd like me to elaborate on further?";
      }
      
      const analytMessage = {
        id: Date.now(),
        user: "F1 Analyst",
        message: response,
        timestamp: 'Just now',
        avatar: 'F',
        isBot: true
      };
      
      setMessages(prev => [...prev, analytMessage]);
    }, delay);
  };
  
  const handleSuggestionClick = (suggestion) => {
    setNewMessage(suggestion);
    inputRef.current.focus();
  };
  
  // Auto-grow textarea
  const handleInput = (e) => {
    setNewMessage(e.target.value);
    
    // Reset height to auto to prevent the scrollHeight from compounding
    e.target.style.height = 'auto';
    
    // Set the height to scrollHeight + border (if any)
    const newHeight = e.target.scrollHeight;
    if (newHeight <= 150) { // Max height
      e.target.style.height = newHeight + 'px';
    }
  };

  return (
    <div className="chat-container">
      <motion.div 
        className="chat-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="chat-title">F1 Expert Analysis</h1>
        <p className="chat-subtitle">
          Get professional insights on races, drivers, teams, and technical aspects of Formula 1
        </p>
      </motion.div>
      
      <motion.div 
        className="chat-window"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <div className="analyst-profile">
          <div className="analyst-avatar">
            F
          </div>
          <div className="analyst-info">
            <div className="analyst-name">F1 Expert Analyst</div>
            <div className="analyst-status">Online now</div>
          </div>
        </div>
        
        <div className="messages-container">
          <AnimatePresence>
            {messages.map((msg, index) => (
              <motion.div 
                key={msg.id}
                className={`message ${msg.isUser ? 'user-message' : ''}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ delay: isLoaded ? 0 : index * 0.1, duration: 0.3 }}
              >
                {!msg.isUser && (
                  <div className="avatar avatar-f1">
                    {msg.avatar}
                  </div>
                )}
                <div className="message-content">
                  <div className="message-header">
                    <span className="message-username">{msg.user}</span>
                    <span className="message-timestamp">{msg.timestamp}</span>
                  </div>
                  <p className="message-text">{msg.message}</p>
                </div>
                {msg.isUser && (
                  <div className="avatar avatar-user">
                    {msg.avatar}
                  </div>
                )}
              </motion.div>
            ))}
            
            {isTyping && (
              <motion.div 
                className="message"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="avatar avatar-f1">F</div>
                <div className="message-content">
                  <div className="message-header">
                    <span className="message-username">F1 Analyst</span>
                  </div>
                  <div className="typing-indicator">
                    typing
                    <span className="typing-dots">
                      <span className="dot"></span>
                      <span className="dot"></span>
                      <span className="dot"></span>
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
            // ...existing code...
            {messages.length === 1 && (
              <motion.div 
                className="suggestion-container"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7, duration: 0.5 }}
              >
                <h3 className="suggestion-title">Suggested Questions</h3>
                <div className="suggestion-list">
                  {suggestionQuestions.map((question, index) => (
                    <motion.button
                      key={index}
                      className="suggestion-button"
                      onClick={() => handleSuggestionClick(question)}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.8 + (index * 0.1), duration: 0.3 }}
                      whileHover={{ scale: 1.03, backgroundColor: 'rgba(225, 6, 0, 0.1)' }}
                      whileTap={{ scale: 0.98 }}
                    >
                      {question}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <div ref={messagesEndRef} />
        </div>
        
        <div className="message-input-container">
          <form onSubmit={handleSendMessage} className="message-form">
            <textarea
              ref={inputRef}
              value={newMessage}
              onChange={handleInput}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
              placeholder="Ask about F1 drivers, teams, strategies, or rules..."
              className="message-input"
              rows="1"
            />
            <button
              type="submit"
              className="send-button"
              disabled={!newMessage.trim()}
              aria-label="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </form>
        </div>
      </motion.div>
      
      <motion.div 
        className="chat-footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.5 }}
      >
        <p>This F1 Analyst uses advanced AI to provide professional insights on Formula 1.</p>
      </motion.div>
    </div>
  );
}