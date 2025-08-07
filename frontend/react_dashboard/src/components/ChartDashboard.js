import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ChartDashboard.css';

const API_BASE_URL = 'http://localhost:8000/api';

function ChartDashboard({ model, onClose }) {
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [error, setError] = useState('');

  useEffect(() => {
    fetchModelMetrics();
  }, [model.id]);

  const fetchModelMetrics = async () => {
    try {
      setLoading(true);
      setError('');
      
      // Fetch actual model metrics from the backend
      const response = await axios.get(`${API_BASE_URL}/modeling/models/${model.id}/metrics/`);
      
      if (response.data && response.data.metrics) {
        console.log('Received model metrics:', response.data);
        // Store the entire response data, not just metrics
        setModelMetrics(response.data);
      } else {
        // Fallback to mock data if API doesn't return expected format
        console.warn('API returned unexpected format, using mock data');
        const mockMetrics = generateMockMetrics(model);
        setModelMetrics(mockMetrics);
      }
    } catch (error) {
      console.error('Error fetching model metrics:', error);
      setError('Failed to load model metrics. Using sample data for demonstration.');
      
      // Use mock data as fallback
      const mockMetrics = generateMockMetrics(model);
      setModelMetrics(mockMetrics);
    } finally {
      setLoading(false);
    }
  };

  // Improved model type detection
  const isClassificationModel = (modelType) => {
    const classificationKeywords = [
      'classification', 'logistic', 'svm', 'knn', 'decision_tree', 
      'random_forest', 'gradient_boosting', 'ada_boost', 'catboost',
      'xgboost', 'lightgbm', 'naive_bayes', 'neural_network'
    ];
    
    const regressionKeywords = [
      'regression', 'linear', 'ridge', 'lasso', 'elastic_net',
      'polynomial', 'support_vector_regression', 'svr'
    ];
    
    const modelTypeLower = modelType.toLowerCase();
    
    // Check for regression keywords first (more specific)
    for (const keyword of regressionKeywords) {
      if (modelTypeLower.includes(keyword)) {
        return false; // It's a regression model
      }
    }
    
    // Check for classification keywords
    for (const keyword of classificationKeywords) {
      if (modelTypeLower.includes(keyword)) {
        return true; // It's a classification model
      }
    }
    
    // Default to classification if no specific keywords found
    // This is a safe default since most ML models are classification
    return true;
  };

  const generateMockMetrics = (model) => {
    const isClassification = isClassificationModel(model.model_type);

    if (isClassification) {
      return {
        accuracy: 0.87,
        precision: 0.89,
        recall: 0.85,
        f1_score: 0.87,
        auc_roc: 0.92,
        confusion_matrix: {
          true_negatives: 245,
          false_positives: 15,
          false_negatives: 25,
          true_positives: 215
        },
        roc_curve: {
          fpr: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
          tpr: [0, 0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.91, 0.94, 0.97, 1.0]
        },
        feature_importance: [
          { feature: 'feature_1', importance: 0.25 },
          { feature: 'feature_2', importance: 0.18 },
          { feature: 'feature_3', importance: 0.15 },
          { feature: 'feature_4', importance: 0.12 },
          { feature: 'feature_5', importance: 0.10 },
          { feature: 'feature_6', importance: 0.08 },
          { feature: 'feature_7', importance: 0.07 },
          { feature: 'feature_8', importance: 0.05 }
        ],
        classification_report: {
          precision: { '0': 0.85, '1': 0.89 },
          recall: { '0': 0.88, '1': 0.85 },
          f1_score: { '0': 0.86, '1': 0.87 },
          support: { '0': 260, '1': 240 }
        }
      };
    } else {
      // Regression metrics
      return {
        mse: 0.045,
        rmse: 0.212,
        mae: 0.156,
        r2_score: 0.89,
        explained_variance: 0.89,
        feature_importance: [
          { feature: 'feature_1', importance: 0.28 },
          { feature: 'feature_2', importance: 0.20 },
          { feature: 'feature_3', importance: 0.16 },
          { feature: 'feature_4', importance: 0.14 },
          { feature: 'feature_5', importance: 0.12 },
          { feature: 'feature_6', importance: 0.10 }
        ],
        residuals: Array.from({ length: 100 }, (_, i) => ({
          predicted: Math.random() * 10 + 5,
          actual: Math.random() * 10 + 5,
          residual: (Math.random() - 0.5) * 2
        }))
      };
    }
  };

  const renderConfusionMatrix = (matrix) => {
    if (!matrix) {
      return <p>Confusion matrix data not available</p>;
    }

    // Handle different data formats
    let cm = matrix;
    
    if (typeof matrix === 'object' && matrix.true_negatives !== undefined) {
      // If it's already in the expected format
      cm = matrix;
    } else if (Array.isArray(matrix) && matrix.length >= 2) {
      // If it's a 2D array from sklearn confusion_matrix
      if (matrix[0].length >= 2) {
        cm = {
          true_negatives: matrix[0][0] || 0,
          false_positives: matrix[0][1] || 0,
          false_negatives: matrix[1][0] || 0,
          true_positives: matrix[1][1] || 0
        };
      }
    } else {
      return <p>Confusion matrix data format not recognized</p>;
    }

    const total = cm.true_negatives + cm.false_positives + cm.false_negatives + cm.true_positives;
    
    if (total === 0) {
      return <p>Confusion matrix data is empty</p>;
    }
    
    return (
      <div className="confusion-matrix">
        <div className="matrix-grid">
          <div className="matrix-cell tn">
            <div className="cell-value">{cm.true_negatives}</div>
            <div className="cell-label">True Negatives</div>
            <div className="cell-percentage">{((cm.true_negatives / total) * 100).toFixed(1)}%</div>
          </div>
          <div className="matrix-cell fp">
            <div className="cell-value">{cm.false_positives}</div>
            <div className="cell-label">False Positives</div>
            <div className="cell-percentage">{((cm.false_positives / total) * 100).toFixed(1)}%</div>
          </div>
          <div className="matrix-cell fn">
            <div className="cell-value">{cm.false_negatives}</div>
            <div className="cell-label">False Negatives</div>
            <div className="cell-percentage">{((cm.false_negatives / total) * 100).toFixed(1)}%</div>
          </div>
          <div className="matrix-cell tp">
            <div className="cell-value">{cm.true_positives}</div>
            <div className="cell-label">True Positives</div>
            <div className="cell-percentage">{((cm.true_positives / total) * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    );
  };

  const renderFeatureImportance = (features) => {
    if (!features) {
      return <p>Feature importance data not available</p>;
    }

    // Handle different data formats from API
    let featureArray = [];
    
    if (Array.isArray(features)) {
      // If it's already an array of objects with feature and importance
      featureArray = features;
    } else if (typeof features === 'object' && features !== null) {
      // If it's an object with feature names as keys and importance as values
      featureArray = Object.entries(features).map(([feature, importance]) => ({
        feature: feature,
        importance: typeof importance === 'number' ? importance : parseFloat(importance) || 0
      }));
    } else {
      return <p>Feature importance data format not recognized</p>;
    }

    // Sort by importance (descending)
    featureArray.sort((a, b) => b.importance - a.importance);

    if (featureArray.length === 0) {
      return <p>No feature importance data available</p>;
    }

    return (
      <div className="feature-importance">
        {featureArray.map((feature, index) => (
          <div key={index} className="feature-bar">
            <div className="feature-name">{feature.feature}</div>
            <div className="feature-bar-container">
              <div 
                className="feature-bar-fill" 
                style={{ width: `${feature.importance * 100}%` }}
              ></div>
            </div>
            <div className="feature-value">{(feature.importance * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    );
  };

  const renderROCCurve = (rocData) => {
    if (!rocData) {
      return <p>ROC curve data not available</p>;
    }

    // Handle different data formats
    let fpr, tpr;
    
    if (rocData.fpr && rocData.tpr) {
      fpr = Array.isArray(rocData.fpr) ? rocData.fpr : [];
      tpr = Array.isArray(rocData.tpr) ? rocData.tpr : [];
    } else if (Array.isArray(rocData)) {
      // If rocData is an array of points
      const points = rocData;
      fpr = points.map(p => p.fpr || p.x || 0);
      tpr = points.map(p => p.tpr || p.y || 0);
    } else {
      return <p>ROC curve data format not recognized</p>;
    }

    if (fpr.length === 0 || tpr.length === 0) {
      return <p>ROC curve data is empty</p>;
    }

    // Create a simple ROC curve visualization using CSS
    const points = fpr.map((fpr_val, i) => ({
      x: fpr_val * 100,
      y: tpr[i] * 100
    }));

    return (
      <div className="roc-curve">
        <svg width="300" height="300" viewBox="0 0 300 300">
          <defs>
            <linearGradient id="rocGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3498db" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#3498db" stopOpacity="0.1"/>
            </linearGradient>
          </defs>
          
          {/* Grid lines */}
          <line x1="0" y1="300" x2="300" y2="300" stroke="#ecf0f1" strokeWidth="1"/>
          <line x1="0" y1="0" x2="0" y2="300" stroke="#ecf0f1" strokeWidth="1"/>
          
          {/* ROC curve */}
          <polyline
            points={points.map(p => `${p.x * 3},${300 - p.y * 3}`).join(' ')}
            fill="none"
            stroke="#3498db"
            strokeWidth="2"
          />
          
          {/* Area under curve */}
          <polygon
            points={`0,300 ${points.map(p => `${p.x * 3},${300 - p.y * 3}`).join(' ')} 300,300`}
            fill="url(#rocGradient)"
          />
          
          {/* Random classifier line */}
          <line x1="0" y1="300" x2="300" y2="0" stroke="#e74c3c" strokeWidth="1" strokeDasharray="5,5"/>
          
          {/* Labels */}
          <text x="150" y="320" textAnchor="middle" fontSize="12" fill="#7f8c8d">False Positive Rate</text>
          <text x="-150" y="150" textAnchor="middle" fontSize="12" fill="#7f8c8d" transform="rotate(-90)">True Positive Rate</text>
        </svg>
        <div className="auc-score">AUC: {modelMetrics.auc_roc ? modelMetrics.auc_roc.toFixed(3) : 'N/A'}</div>
      </div>
    );
  };

  const renderRegressionMetrics = (metrics) => {
    if (!metrics) return <p>Regression metrics not available</p>;

    return (
      <div className="regression-metrics">
        <div className="metric-grid">
          <div className="metric-card">
            <h4>Mean Squared Error (MSE)</h4>
            <div className="metric-value">{metrics.mse ? metrics.mse.toFixed(4) : 'N/A'}</div>
            <div className="metric-description">Lower is better</div>
          </div>
          <div className="metric-card">
            <h4>Root Mean Squared Error (RMSE)</h4>
            <div className="metric-value">{metrics.rmse ? metrics.rmse.toFixed(4) : 'N/A'}</div>
            <div className="metric-description">Lower is better</div>
          </div>
          <div className="metric-card">
            <h4>Mean Absolute Error (MAE)</h4>
            <div className="metric-value">{metrics.mae ? metrics.mae.toFixed(4) : 'N/A'}</div>
            <div className="metric-description">Lower is better</div>
          </div>
          <div className="metric-card">
            <h4>R² Score</h4>
            <div className="metric-value">{metrics.r2_score ? metrics.r2_score.toFixed(3) : 'N/A'}</div>
            <div className="metric-description">Higher is better (max 1.0)</div>
          </div>
          <div className="metric-card">
            <h4>Explained Variance</h4>
            <div className="metric-value">{metrics.explained_variance ? metrics.explained_variance.toFixed(3) : 'N/A'}</div>
            <div className="metric-description">Higher is better (max 1.0)</div>
          </div>
        </div>
      </div>
    );
  };

  const renderClassificationMetrics = (metrics) => {
    if (!metrics) return <p>Classification metrics not available</p>;

    return (
      <div className="classification-metrics">
        <div className="metric-grid">
          <div className="metric-card">
            <h4>Accuracy</h4>
            <div className="metric-value">{metrics.accuracy ? (metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
            <div className="metric-description">Overall correctness</div>
          </div>
          <div className="metric-card">
            <h4>Precision</h4>
            <div className="metric-value">{metrics.precision ? (metrics.precision * 100).toFixed(1) + '%' : 'N/A'}</div>
            <div className="metric-description">True positives / (True + False positives)</div>
          </div>
          <div className="metric-card">
            <h4>Recall</h4>
            <div className="metric-value">{metrics.recall ? (metrics.recall * 100).toFixed(1) + '%' : 'N/A'}</div>
            <div className="metric-description">True positives / (True positives + False negatives)</div>
          </div>
          <div className="metric-card">
            <h4>F1 Score</h4>
            <div className="metric-value">{metrics.f1_score ? (metrics.f1_score * 100).toFixed(1) + '%' : 'N/A'}</div>
            <div className="metric-description">Harmonic mean of precision and recall</div>
          </div>
          <div className="metric-card">
            <h4>AUC-ROC</h4>
            <div className="metric-value">{metrics.auc_roc ? metrics.auc_roc.toFixed(3) : 'N/A'}</div>
            <div className="metric-description">Area under ROC curve</div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <div className="loading-spinner"></div>
          <p>Loading model performance metrics...</p>
        </div>
      </div>
    );
  }

  // Use problem_type from backend metrics if available, otherwise fallback to model type detection
  const isClassification = modelMetrics && modelMetrics.problem_type 
    ? modelMetrics.problem_type === 'classification'
    : isClassificationModel(model.model_type);

  return (
    <div className="modal-overlay">
      <div className="modal-content chart-modal">
        <button className="modal-close" onClick={onClose}>×</button>
        
        <h2>Model Performance: {model.name}</h2>
        
        {error && (
          <div className="error-notice">
            <p>{error}</p>
          </div>
        )}
        
        <div className="model-tabs">
          <button 
            className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button 
            className={`tab-button ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            Performance Metrics
          </button>
          <button 
            className={`tab-button ${activeTab === 'charts' ? 'active' : ''}`}
            onClick={() => setActiveTab('charts')}
          >
            Charts & Visualizations
          </button>
          <button 
            className={`tab-button ${activeTab === 'features' ? 'active' : ''}`}
            onClick={() => setActiveTab('features')}
          >
            Feature Analysis
          </button>
        </div>

        {activeTab === 'overview' && (
          <div className="overview-section">
            <div className="model-details">
              <div className="detail-section">
                <h3>Model Information</h3>
                <div className="details-grid">
                  <div className="detail-item">
                    <span className="detail-label">Model Type:</span>
                    <span className="detail-value">{model.model_type}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Dataset:</span>
                    <span className="detail-value">{model.dataset_name}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Target Column:</span>
                    <span className="detail-value">{model.target_column}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Status:</span>
                    <span className="detail-value">{model.status}</span>
                  </div>
                  {model.training_duration && (
                    <div className="detail-item">
                      <span className="detail-label">Training Time:</span>
                      <span className="detail-value">{model.training_duration.toFixed(2)}s</span>
                    </div>
                  )}
                  <div className="detail-item">
                    <span className="detail-label">Problem Type:</span>
                    <span className="detail-value">
                      {isClassification ? 'Classification' : 'Regression'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="quick-metrics">
                <h3>Quick Performance Summary</h3>
                <div className="metrics-summary">
                  {isClassification ? (
                    <>
                      <div className="summary-metric">
                        <span className="metric-label">Accuracy</span>
                        <span className="metric-value">
                          {modelMetrics?.metrics?.accuracy ? (modelMetrics.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}
                        </span>
                      </div>
                      <div className="summary-metric">
                        <span className="metric-label">F1 Score</span>
                        <span className="metric-value">
                          {modelMetrics?.metrics?.f1_score ? (modelMetrics.metrics.f1_score * 100).toFixed(1) + '%' : 'N/A'}
                        </span>
                      </div>
                      <div className="summary-metric">
                        <span className="metric-label">AUC-ROC</span>
                        <span className="metric-value">
                          {modelMetrics?.metrics?.auc_roc ? modelMetrics.metrics.auc_roc.toFixed(3) : 'N/A'}
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="summary-metric">
                        <span className="metric-label">R² Score</span>
                        <span className="metric-value">
                          {modelMetrics?.metrics?.r2_score ? modelMetrics.metrics.r2_score.toFixed(3) : 'N/A'}
                        </span>
                      </div>
                      <div className="summary-metric">
                        <span className="metric-label">RMSE</span>
                        <span className="metric-value">
                          {modelMetrics?.metrics?.rmse ? modelMetrics.metrics.rmse.toFixed(4) : 'N/A'}
                        </span>
                      </div>
                      <div className="summary-metric">
                        <span className="metric-label">MAE</span>
                        <span className="metric-value">
                          {modelMetrics?.metrics?.mae ? modelMetrics.metrics.mae.toFixed(4) : 'N/A'}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="metrics-section">
            <h3>Detailed Performance Metrics</h3>
            {isClassification ? renderClassificationMetrics(modelMetrics?.metrics) : renderRegressionMetrics(modelMetrics?.metrics)}
          </div>
        )}

        {activeTab === 'charts' && (
          <div className="charts-section">
            <h3>Performance Charts</h3>
            <div className="charts-grid">
              {isClassification && (
                <>
                  <div className="chart-card">
                    <h4>Confusion Matrix</h4>
                    <div className="chart-content">
                      {renderConfusionMatrix(modelMetrics?.metrics?.confusion_matrix)}
                    </div>
                  </div>
                  <div className="chart-card">
                    <h4>ROC Curve</h4>
                    <div className="chart-content">
                      {renderROCCurve(modelMetrics?.metrics?.roc_curve)}
                    </div>
                  </div>
                </>
              )}
              <div className="chart-card">
                <h4>Feature Importance</h4>
                <div className="chart-content">
                  {renderFeatureImportance(modelMetrics?.metrics?.feature_importance)}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'features' && (
          <div className="features-section">
            <h3>Feature Analysis</h3>
            <div className="feature-analysis">
              <div className="feature-importance-chart">
                <h4>Feature Importance Ranking</h4>
                {renderFeatureImportance(modelMetrics?.feature_importance)}
              </div>
              <div className="feature-insights">
                <h4>Key Insights</h4>
                {(() => {
                  // Process feature importance data to get insights
                  const features = modelMetrics?.metrics?.feature_importance;
                  if (!features) {
                    return <p>Feature importance data not available</p>;
                  }

                  let featureArray = [];
                  if (Array.isArray(features)) {
                    featureArray = features;
                  } else if (typeof features === 'object' && features !== null) {
                    featureArray = Object.entries(features).map(([feature, importance]) => ({
                      feature: feature,
                      importance: typeof importance === 'number' ? importance : parseFloat(importance) || 0
                    }));
                  }

                  if (featureArray.length === 0) {
                    return <p>No feature importance data available</p>;
                  }

                  // Sort by importance (descending)
                  featureArray.sort((a, b) => b.importance - a.importance);

                  const topFeature = featureArray[0];
                  const top3Importance = featureArray.slice(0, 3).reduce((sum, f) => sum + f.importance, 0);

                  return (
                    <ul>
                      <li><strong>Top Feature:</strong> {topFeature.feature} contributes {(topFeature.importance * 100).toFixed(1)}% to predictions</li>
                      <li><strong>Top 3 Features:</strong> Account for {(top3Importance * 100).toFixed(1)}% of total importance</li>
                      <li><strong>Feature Count:</strong> {featureArray.length} features analyzed</li>
                    </ul>
                  );
                })()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChartDashboard; 