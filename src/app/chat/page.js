"use client";
import { useState, useEffect, useRef } from 'react';

const sampleMessages = [
  { id: 1, user: 'Max Verstappen Fan', message: 'Anyone watching the Belgian GP this weekend?', timestamp: '2 hours ago' },
  { id: 2, user: 'Lewis Hamilton', message: 'Looking forward to it! Weather forecast says rain...', timestamp: '1 hour ago' },
  { id: 3, user: 'F1Enthusiast', message: 'Spa in the rain is always exciting!', timestamp: '45 minutes ago' },
];

export default function Chat() {
  const [messages, setMessages] = useState(sampleMessages);
  const [newMessage, setNewMessage] = useState('');
  const [username, setUsername] = useState('Guest');
  const messagesEndRef = useRef(null);

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
      user: username,
      message: newMessage,
      timestamp: 'Just now',
    };
    
    setMessages([...messages, message]);
    setNewMessage('');
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">F1 Community Chat</h1>
      
      <div className="mb-4">
        <label htmlFor="username" className="block text-sm font-medium text-gray-700">
          Your Name
        </label>
        <input
          type="text"
          id="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-red-500 focus:border-red-500"
        />
      </div>
      
      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="h-96 p-4 overflow-y-auto">
          {messages.map((msg) => (
            <div key={msg.id} className="mb-4">
              <div className="flex space-x-3">
                <div className="flex-shrink-0">
                  <div className="h-10 w-10 rounded-full bg-red-600 flex items-center justify-center text-white font-bold">
                    {msg.user.charAt(0).toUpperCase()}
                  </div>
                </div>
                <div>
                  <div className="text-sm">
                    <span className="font-semibold">{msg.user}</span>
                    <span className="text-gray-500 ml-2">{msg.timestamp}</span>
                  </div>
                  <div className="mt-1 text-sm text-gray-800">
                    <p>{msg.message}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="border-t border-gray-200 p-4">
          <form onSubmit={handleSendMessage} className="flex space-x-2">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 min-w-0 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-red-500 focus:border-red-500"
            />
            <button
              type="submit"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}