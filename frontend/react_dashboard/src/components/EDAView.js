import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './EDAView.css';

const API_BASE_URL = 'http://localhost:8000/api';

function EDAView({ dataset, onClose, onTrainModel, onCompareModels }) {
  const [datasetDetails, setDatasetDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState('');
  const [targetColumn, setTargetColumn] = useState(dataset.target_column || '');
  const [showTrainForm, setShowTrainForm] = useState(false);
  const [showCompareForm, setShowCompareForm] = useState(false);
  const [intelligentSuggestions, setIntelligentSuggestions] = useState(null);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [problemType, setProblemType] = useState('classification'); // Default to classification

  useEffect(() => {
    fetchDatasetDetails();
  }, [dataset.id]);

  useEffect(() => {
    if (targetColumn) {
      fetchIntelligentSuggestions();
    }
  }, [targetColumn]);

  // Set the AI-recommended model as default when suggestions are loaded
  useEffect(() => {
    if (intelligentSuggestions?.final_recommendation?.best_model) {
      setSelectedModel(intelligentSuggestions.final_recommendation.best_model);
    }
  }, [intelligentSuggestions]);

  const fetchDatasetDetails = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/ingestion/datasets/${dataset.id}/`);
      setDatasetDetails(response.data);
      
      // Set problem type from dataset if available
      if (response.data.dataset.problem_type) {
        setProblemType(response.data.dataset.problem_type.toLowerCase());
      }
    } catch (error) {
      console.error('Error fetching dataset details:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchIntelligentSuggestions = async () => {
    if (!targetColumn) return;
    
    setLoadingSuggestions(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/modeling/suggestions/${dataset.id}/`);
      setIntelligentSuggestions(response.data.intelligent_suggestions);
      
      // Update problem type from suggestions if available
      if (response.data.problem_type) {
        setProblemType(response.data.problem_type.toLowerCase());
      }
    } catch (error) {
      console.error('Error fetching intelligent suggestions:', error);
    } finally {
      setLoadingSuggestions(false);
    }
  };

  const handleTrainModel = () => {
    if (!targetColumn) {
      alert('Please specify a target column');
      return;
    }
    onTrainModel(dataset.id, selectedModel, targetColumn);
    onClose();
  };

  const handleCompareModels = () => {
    if (!targetColumn) {
      alert('Please specify a target column');
      return;
    }
    
    // Get relevant models for the problem type
    const relevantModels = getModelsForProblemType(problemType);
    const selectedModels = relevantModels.slice(0, 3).map(model => model.value); // Take first 3 models
    onCompareModels(dataset.id, selectedModels, targetColumn);
    onClose();
  };

  const getModelsForProblemType = (type) => {
    const classificationModels = [
      { value: 'random_forest', label: 'Random Forest' },
      // { value: 'xgboost', label: 'XGBoost' },
      // { value: 'lightgbm', label: 'LightGBM' },
      { value: 'catboost', label: 'CatBoost' },
      // { value: 'logistic_regression', label: 'Logistic Regression' },
      { value: 'svm', label: 'Support Vector Machine' },
      { value: 'knn', label: 'K-Nearest Neighbors' },
      { value: 'decision_tree', label: 'Decision Tree' },
      { value: 'gradient_boosting', label: 'Gradient Boosting' },
      { value: 'ada_boost', label: 'AdaBoost' }
    ];

    const regressionModels = [
      { value: 'random_forest', label: 'Random Forest' },
      // { value: 'xgboost', label: 'XGBoost' },
      // { value: 'lightgbm', label: 'LightGBM' },
      { value: 'catboost', label: 'CatBoost' },
      { value: 'linear_regression', label: 'Linear Regression' },
      { value: 'ridge', label: 'Ridge Regression' },
      { value: 'lasso', label: 'Lasso Regression' },
      { value: 'elastic_net', label: 'Elastic Net' },
      { value: 'svm', label: 'Support Vector Regression' },
      { value: 'knn', label: 'K-Nearest Neighbors' },
      { value: 'decision_tree', label: 'Decision Tree' },
      { value: 'gradient_boosting', label: 'Gradient Boosting' }
    ];

    return type === 'regression' ? regressionModels : classificationModels;
  };

  // Determine if dataset has identified problem type
  const hasIdentifiedProblemType = dataset.problem_type || (datasetDetails && datasetDetails.dataset.problem_type);

  if (loading) {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <div className="loading-spinner"></div>
          <p>Loading dataset details...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content eda-modal">
        <button className="modal-close" onClick={onClose}>×</button>
        
        <h2>Dataset Analysis: {dataset.name}</h2>
        
        {datasetDetails && (
          <div className="eda-content">
            <div className="dataset-overview">
              <h3>Overview</h3>
              <div className="overview-stats">
                <div className="stat">
                  <span className="stat-label">Rows:</span>
                  <span className="stat-value">{datasetDetails.dataset.rows}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Columns:</span>
                  <span className="stat-value">{datasetDetails.dataset.columns}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Target:</span>
                  <span className="stat-value">{datasetDetails.dataset.target_column || 'Not specified'}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Type:</span>
                  <span className="stat-value">
                    {datasetDetails.dataset.problem_type || 'Unknown'}
                    {hasIdentifiedProblemType && (
                      <span className={`problem-type-badge ${problemType}`}>
                        {problemType}
                      </span>
                    )}
                  </span>
                </div>
              </div>
            </div>

            <div className="column-analysis">
              <h3>Column Analysis</h3>
              <div className="columns-grid">
                {datasetDetails.columns_info.map((col, index) => (
                  <div key={index} className="column-card">
                    <h4>{col.name}</h4>
                    <p><strong>Type:</strong> {col.data_type}</p>
                    <p><strong>Missing:</strong> {col.missing_count}</p>
                    <p><strong>Unique:</strong> {col.unique_count}</p>
                    {col.min_value !== null && (
                      <p><strong>Range:</strong> {col.min_value} - {col.max_value}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="model-actions">
              <h3>Model Training</h3>
              
              <div className="form-group">
                <label>
                  Target Column:
                  {hasIdentifiedProblemType && (
                    <span className={`problem-type-indicator ${problemType}`}>
                      {problemType}
                    </span>
                  )}
                </label>
                <input
                  type="text"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="Enter target column name"
                />
              </div>

              {targetColumn && intelligentSuggestions && (
                <div className="intelligent-suggestions">
                  <h4>🤖 AI-Powered Model Recommendations</h4>
                  {loadingSuggestions ? (
                    <p>Analyzing your data with AI...</p>
                  ) : (
                    <div className="suggestions-content">
                      {intelligentSuggestions.final_recommendation && (
                        <div className="best-recommendation">
                          <h5>🎯 Best Model: {intelligentSuggestions.final_recommendation.best_model}</h5>
                          <p><strong>Confidence:</strong> {(intelligentSuggestions.final_recommendation.confidence * 100).toFixed(1)}%</p>
                          <p><strong>Reasoning:</strong> {intelligentSuggestions.final_recommendation.reasoning}</p>
                        </div>
                      )}
                      
                      {intelligentSuggestions.recommended_models && (
                        <div className="recommended-models">
                          <h5>📋 All Recommended Models:</h5>
                          <ul>
                            {intelligentSuggestions.recommended_models.map((model, index) => (
                              <li key={index}>
                                <strong>{model.name || model}</strong>
                                {model.reasoning && <p>{model.reasoning}</p>}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="action-buttons">
                <button 
                  className="btn-primary"
                  onClick={() => setShowTrainForm(true)}
                >
                  Train Single Model
                </button>
                {/* <button 
                  className="btn-secondary"
                  onClick={() => setShowCompareForm(true)}
                >
                  Compare Models
                </button> */}
              </div>
            </div>
          </div>
        )}

        {showTrainForm && (
          <div className="modal-overlay">
            <div className="modal-content">
              <button className="modal-close" onClick={() => setShowTrainForm(false)}>×</button>
              <h3>Train Model</h3>
              
              {hasIdentifiedProblemType && (
                <div className="problem-type-notice">
                  <p>
                    <strong>Dataset Type:</strong> {problemType.charAt(0).toUpperCase() + problemType.slice(1)}
                    <br />
                    <small>Showing only {problemType} algorithms based on your dataset analysis</small>
                  </p>
                </div>
              )}
              
              <div className="form-group">
                <label>Model Type:</label>
                <select 
                  value={selectedModel} 
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <option value="">Select a model</option>
                  
                  {getModelsForProblemType(problemType).map((model) => (
                    <option key={model.value} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </select>
                <small>
                  {hasIdentifiedProblemType 
                    ? `Showing ${problemType} models based on your dataset type`
                    : 'Showing all available models (dataset type not identified)'
                  }
                  {intelligentSuggestions?.final_recommendation?.best_model === selectedModel && (
                    <span style={{color: '#27ae60', fontWeight: 'bold'}}>
                      {' '}• AI Recommended
                    </span>
                  )}
                </small>
              </div>

              <div className="form-actions">
                <button 
                  className="btn-primary"
                  onClick={handleTrainModel}
                  disabled={!selectedModel || !targetColumn}
                >
                  Train Model
                </button>
                <button 
                  className="btn-secondary"
                  onClick={() => setShowTrainForm(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {showCompareForm && (
          <div className="modal-overlay">
            <div className="modal-content">
              <button className="modal-close" onClick={() => setShowCompareForm(false)}>×</button>
              <h3>Compare Models</h3>
              
              {hasIdentifiedProblemType && (
                <div className="problem-type-notice">
                  <p>
                    <strong>Dataset Type:</strong> {problemType.charAt(0).toUpperCase() + problemType.slice(1)}
                    <br />
                    <small>Will compare top {problemType} algorithms</small>
                  </p>
                </div>
              )}
              
              <p>This will train and compare multiple models on your dataset.</p>
              
              <div className="form-actions">
                <button 
                  className="btn-primary"
                  onClick={handleCompareModels}
                  disabled={!targetColumn}
                >
                  Start Comparison
                </button>
                <button 
                  className="btn-secondary"
                  onClick={() => setShowCompareForm(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default EDAView; 