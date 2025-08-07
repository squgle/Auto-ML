import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FileUploader from '../components/FileUploader';
import EDAView from '../components/EDAView';
import ChartDashboard from '../components/ChartDashboard';
import ReportViewer from '../components/ReportViewer';
import Chatbot from '../components/Chatbot';
import './Dashboard.css';

const API_BASE_URL = 'http://localhost:8000/api';

function Dashboard() {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [reports, setReports] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showChatbot, setShowChatbot] = useState(false);
  const [chatbotDataset, setChatbotDataset] = useState(null);

  useEffect(() => {
    fetchDatasets();
    fetchModels();
    fetchReports();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/ingestion/datasets/`);
      setDatasets(response.data);
    } catch (error) {
      setError('Failed to fetch datasets');
      console.error('Error fetching datasets:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/modeling/models/`);
      setModels(response.data);
    } catch (error) {
      setError('Failed to fetch models');
      console.error('Error fetching models:', error);
    }
  };

  const fetchReports = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/reporting/reports/`);
      setReports(response.data);
    } catch (error) {
      setError('Failed to fetch reports');
      console.error('Error fetching reports:', error);
    }
  };

  const handleDatasetUpload = async (file, name, description, targetColumn) => {
    setLoading(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', name);
      formData.append('description', description);
      formData.append('target_column', targetColumn);

      const response = await axios.post(`${API_BASE_URL}/ingestion/upload/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      await fetchDatasets();
      setActiveTab('datasets');
      alert('Dataset uploaded successfully!');
    } catch (error) {
      setError('Failed to upload dataset');
      console.error('Error uploading dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleModelTraining = async (datasetId, modelType, targetColumn, hyperparameters = {}) => {
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/modeling/train/`, {
        dataset_id: datasetId,
        model_type: modelType,
        target_column: targetColumn,
        hyperparameters: hyperparameters,
      });

      await fetchModels();
      setActiveTab('models');
      alert('Model trained successfully!');
    } catch (error) {
      setError('Failed to train model');
      console.error('Error training model:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleModelComparison = async (datasetId, modelTypes, targetColumn) => {
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/modeling/compare/`, {
        dataset_id: datasetId,
        model_types: modelTypes,
        target_column: targetColumn,
      });

      await fetchModels();
      setActiveTab('models');
      alert('Model comparison completed!');
    } catch (error) {
      setError('Failed to compare models');
      console.error('Error comparing models:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async (reportType, datasetId = null, modelId = null) => {
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/reporting/generate/`, {
        report_type: reportType,
        dataset_id: datasetId,
        model_id: modelId,
      });

      await fetchReports();
      setActiveTab('reports');
      alert('Report generated successfully!');
    } catch (error) {
      setError('Failed to generate report');
      console.error('Error generating report:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDataset = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset and all related models and reports?')) return;
    setLoading(true);
    setError('');
    try {
      await axios.delete(`${API_BASE_URL}/ingestion/datasets/${datasetId}/delete/`);
      await fetchDatasets();
      await fetchModels();
      await fetchReports();
      alert('Dataset and all related data deleted successfully!');
    } catch (error) {
      setError('Failed to delete dataset');
      console.error('Error deleting dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>ML AutoML Dashboard</h1>
        <p>Upload data, train models, and generate reports</p>
      </header>

      {error && (
        <div className="error-message">
          {error}
          <button onClick={() => setError('')}>×</button>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Processing...</p>
        </div>
      )}

      <nav className="dashboard-nav">
        <button
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          Upload Dataset
        </button>
        <button
          className={activeTab === 'datasets' ? 'active' : ''}
          onClick={() => setActiveTab('datasets')}
        >
          Datasets ({datasets.length})
        </button>
        <button
          className={activeTab === 'models' ? 'active' : ''}
          onClick={() => setActiveTab('models')}
        >
          Models ({models.length})
        </button>
        <button
          className={activeTab === 'reports' ? 'active' : ''}
          onClick={() => setActiveTab('reports')}
        >
          Reports ({reports.length})
        </button>
      </nav>

      <main className="dashboard-content">
        {activeTab === 'upload' && (
          <div className="upload-section">
            <div className="section-header">
              <h2>Upload Dataset</h2>
              <p>Upload your dataset and configure target column</p>
            </div>
            
            <div className="upload-container">
              <FileUploader onUpload={handleDatasetUpload} />
            </div>
          </div>
        )}

        {activeTab === 'datasets' && (
          <div className="datasets-section">
            <div className="section-header">
              <h2>Data Management</h2>
            </div>
            
            <div className="datasets-grid">
              {datasets.map((dataset) => (
                <div key={dataset.id} className="dataset-card">
                  <button
                    className="delete-dataset-btn"
                    title="Delete Dataset"
                    onClick={() => handleDeleteDataset(dataset.id)}
                    style={{ position: 'absolute', top: 8, right: 8, zIndex: 2 }}
                  >
                    ❌
                  </button>
                  <h3>{dataset.name}</h3>
                  <p>{dataset.description}</p>
                  <div className="dataset-stats">
                    <span>Rows: {dataset.rows}</span>
                    <span>Columns: {dataset.columns}</span>
                  </div>
                  {dataset.target_column && (
                    <p><strong>Target:</strong> {dataset.target_column}</p>
                  )}
                  {dataset.problem_type && (
                    <p><strong>Type:</strong> {dataset.problem_type}</p>
                  )}
                  <div className="dataset-actions">
                    <button onClick={() => setSelectedDataset(dataset)}>
                      View Details
                    </button>
                    {/* <button onClick={() => handleGenerateReport('data_analysis', dataset.id)}>
                      Generate Report
                    </button> */}
                    <button 
                      className="chatbot-btn"
                      onClick={() => {
                        setChatbotDataset(dataset);
                        setShowChatbot(true);
                      }}
                    >
                      🤖 Ask AI
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {selectedDataset && (
              <EDAView 
                dataset={selectedDataset} 
                onClose={() => setSelectedDataset(null)}
                onTrainModel={handleModelTraining}
                onCompareModels={handleModelComparison}
              />
            )}
          </div>
        )}

        {activeTab === 'models' && (
          <div className="models-section">
            <h2>Model Management</h2>
            
            <div className="models-grid">
              {models.map((model) => (
                <div key={model.id} className="model-card">
                  <h3>{model.name}</h3>
                  <p><strong>Type:</strong> {model.model_type}</p>
                  <p><strong>Dataset:</strong> {model.dataset_name}</p>
                  <p><strong>Target:</strong> {model.target_column}</p>
                  <p><strong>Status:</strong> {model.status}</p>
                  {model.training_duration && (
                    <p><strong>Training Time:</strong> {model.training_duration.toFixed(2)}s</p>
                  )}
                  <div className="model-actions">
                    <button onClick={() => setSelectedModel(model)}>
                      View Details
                    </button>
                    <button onClick={() => handleGenerateReport('model_evaluation', null, model.id)}>
                      Generate Report
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {selectedModel && (
              <ChartDashboard 
                model={selectedModel} 
                onClose={() => setSelectedModel(null)}
              />
            )}
          </div>
        )}

        {activeTab === 'reports' && (
          <div className="reports-section">
            <h2>Reports</h2>
            
            <div className="reports-grid">
              {reports.map((report) => (
                <div key={report.id} className="report-card">
                  <h3>{report.title}</h3>
                  <p><strong>Type:</strong> {report.report_type}</p>
                  {report.dataset_name && (
                    <p><strong>Dataset:</strong> {report.dataset_name}</p>
                  )}
                  {report.model_name && (
                    <p><strong>Model:</strong> {report.model_name}</p>
                  )}
                  <p>{report.summary}</p>
                  <div className="report-actions">
                    {/* <button onClick={() => setSelectedReport(report)}> */}
                      {/* View Report */}
                    {/* </button> */}
                  </div>
                </div>
              ))}
            </div>

            {selectedReport && (
              <ReportViewer 
                report={selectedReport} 
                onClose={() => setSelectedReport(null)}
              />
            )}
          </div>
        )}

        {showChatbot && chatbotDataset && (
          <Chatbot 
            dataset={chatbotDataset}
            onClose={() => {
              setShowChatbot(false);
              setChatbotDataset(null);
            }}
          />
        )}
      </main>
    </div>
  );
}

export default Dashboard; 