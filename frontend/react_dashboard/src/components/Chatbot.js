import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './Chatbot.css';

const API_BASE_URL = 'http://localhost:8000/api';

function Chatbot({ dataset, onClose }) {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking'); // checking, available, unavailable
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (dataset) {
      createChatSession();
    }
  }, [dataset]);

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      // const response = await axios.get(`${API_BASE_URL}/chatbot/session/create/`, {
     const response = await axios.post(`${API_BASE_URL}/chatbot/session/create/`, {
        timeout: 5000
      });
      setApiStatus('available');
    } catch (error) {
      console.error('API status check failed:', error);
      setApiStatus('unavailable');
      setError('Backend API is not available. Please check if the Django server is running.');
    }
  };

  const createChatSession = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await axios.post(`${API_BASE_URL}/chatbot/session/create/`, {
        dataset_id: dataset.id
      });
      
      setSessionId(response.data.session_id);
      setMessages([{
        id: 1,
        type: 'assistant',
        content: response.data.welcome_message,
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('Error creating chat session:', error);
      
      let errorMessage = 'Failed to create chat session';
      if (error.response) {
        if (error.response.status === 404) {
          errorMessage = 'Dataset not found. Please upload a dataset first.';
        } else if (error.response.status === 503) {
          errorMessage = 'Chatbot service is not available. Please check your API key configuration.';
        } else if (error.response.data && error.response.data.error) {
          errorMessage = error.response.data.error;
        }
      } else if (error.code === 'ECONNREFUSED') {
        errorMessage = 'Cannot connect to backend server. Please ensure the Django server is running.';
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !sessionId) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}/chatbot/message/send/`, {
        session_id: sessionId,
        message: inputMessage
      });

      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response.data.assistant_response,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      let errorMessage = 'Failed to send message';
      if (error.response) {
        if (error.response.status === 404) {
          errorMessage = 'Chat session not found. Please refresh the page.';
        } else if (error.response.status === 503) {
          errorMessage = 'Chatbot service is not available. Please check your API key configuration.';
        } else if (error.response.data && error.response.data.error) {
          errorMessage = error.response.data.error;
        }
      } else if (error.code === 'ECONNREFUSED') {
        errorMessage = 'Cannot connect to backend server. Please ensure the Django server is running.';
      } else if (error.code === 'NETWORK_ERROR') {
        errorMessage = 'Network error. Please check your internet connection.';
      }
      
      setError(errorMessage);
      
      // Add error message to chat
      const errorChatMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: `I'm sorry, I encountered an error: ${errorMessage}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorChatMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const quickActions = [
    "Analyze this dataset",
    "What preprocessing steps do I need?",
    "Show me data quality insights",
    "Create a visualization",
    "What are the key patterns in this data?",
    "Suggest machine learning models"
  ];

  const handleQuickAction = (action) => {
    setInputMessage(action);
  };

  const retryConnection = () => {
    setError('');
    setApiStatus('checking');
    checkApiStatus();
  };

  if (apiStatus === 'unavailable') {
    return (
      <div className="chatbot-modal">
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h3>🤖 AI Data Assistant</h3>
            <button className="close-btn" onClick={onClose}>×</button>
          </div>
          <div className="error-container">
            <div className="error-message">
              <h4>⚠️ Connection Error</h4>
              <p>{error}</p>
              <button onClick={retryConnection} className="retry-btn">
                🔄 Retry Connection
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chatbot-modal">
      <div className="chatbot-container">
        <div className="chatbot-header">
          <h3>🤖 AI Data Assistant</h3>
          {dataset && (
            <span className="dataset-info">
              Analyzing: {dataset.name}
            </span>
          )}
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="chat-messages">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
                {message.content}
              </div>
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <div className="error-message">
            <span>{error}</span>
            <button onClick={() => setError('')} className="error-close-btn">×</button>
          </div>
        )}

        <div className="quick-actions">
          <h4>Quick Actions:</h4>
          <div className="action-buttons">
            {quickActions.map((action, index) => (
              <button
                key={index}
                className="quick-action-btn"
                onClick={() => handleQuickAction(action)}
                disabled={loading}
              >
                {action}
              </button>
            ))}
          </div>
        </div>

        <div className="chat-input">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about your data..."
            disabled={loading || !sessionId}
          />
          <button 
            onClick={sendMessage}
            disabled={loading || !inputMessage.trim() || !sessionId}
            className="send-btn"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chatbot; 