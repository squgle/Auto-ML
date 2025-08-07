import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

function Home() {
  return (
    <div className="home">
      <div className="hero-section">
        <div className="hero-content">
          <h1>ML AutoML Platform</h1>
          <p>Upload your data, train machine learning models, and generate comprehensive reports with ease.</p>
          <div className="hero-features">
            <div className="feature">
              <h3>📊 Data Preprocessing</h3>
              <p>Automated data cleaning, encoding, and scaling</p>
            </div>
            <div className="feature">
              <h3>🤖 Model Selection</h3>
              <p>Intelligent model suggestions based on your data</p>
            </div>
            <div className="feature">
              <h3>📈 Accuracy Reporting</h3>
              <p>Comprehensive evaluation metrics and visualizations</p>
            </div>
          </div>
          <Link to="/dashboard" className="cta-button">
            Get Started
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home; 