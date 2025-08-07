import React, { useState } from 'react';
import './FileUploader.css';

function FileUploader({ onUpload }) {
  const [showForm, setShowForm] = useState(false);
  const [file, setFile] = useState(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [availableColumns, setAvailableColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileLoading(true);
      
      // Auto-set name from filename
      if (!name) {
        setName(selectedFile.name.replace(/\.[^/.]+$/, '')); // Remove extension
      }
      
      // Read file and extract column names
      try {
        const columns = await readFileColumns(selectedFile);
        setAvailableColumns(columns);
        setTargetColumn(''); // Reset target column when file changes
      } catch (error) {
        console.error('Error reading file columns:', error);
        setAvailableColumns([]);
      } finally {
        setFileLoading(false);
      }
    }
  };

  const readFileColumns = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          let columns = [];
          
          if (file.name.toLowerCase().endsWith('.csv')) {
            // Parse CSV
            const lines = content.split('\n');
            if (lines.length > 0) {
              const headerLine = lines[0];
              columns = headerLine.split(',').map(col => col.trim().replace(/"/g, ''));
            }
          } else if (file.name.toLowerCase().endsWith('.xlsx') || file.name.toLowerCase().endsWith('.xls')) {
            // For Excel files, we'll need to use a library like SheetJS
            // For now, we'll show a message that Excel column detection is not available
            alert('Excel file column detection is not available in this demo. Please manually enter the target column name.');
            resolve([]);
            return;
          }
          
          resolve(columns);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      
      if (file.name.toLowerCase().endsWith('.csv')) {
        reader.readAsText(file);
      } else {
        // For Excel files, we can't read them directly in the browser
        // In a real implementation, you'd use a library like SheetJS
        resolve([]);
      }
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!file) {
      alert('Please select a file');
      return;
    }

    if (!name.trim()) {
      alert('Please enter a dataset name');
      return;
    }

    setLoading(true);
    
    try {
      await onUpload(file, name.trim(), description.trim(), targetColumn.trim());
      // Reset form
      setFile(null);
      setName('');
      setDescription('');
      setTargetColumn('');
      setAvailableColumns([]);
      setShowForm(false);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setFile(null);
    setName('');
    setDescription('');
    setTargetColumn('');
    setAvailableColumns([]);
    setShowForm(false);
  };

  return (
    <div className="file-uploader">
      {!showForm ? (
        <button 
          className="upload-button"
          onClick={() => setShowForm(true)}
        >
          + Upload Dataset
        </button>
      ) : (
        <div className="upload-form">
          <h3>Upload Dataset</h3>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="file">Dataset File *</label>
              <input
                type="file"
                id="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileChange}
                required
              />
              <small>Supported formats: CSV, Excel (.xlsx, .xls)</small>
              {fileLoading && <small className="loading-text">Reading file columns...</small>}
            </div>

            <div className="form-group">
              <label htmlFor="name">Dataset Name *</label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter dataset name"
                required
              />
            </div>

            {/* <div className="form-group">
              <label htmlFor="description">Description</label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter dataset description"
                rows="3"
              />
            </div> */}

            <div className="form-group">
              <label htmlFor="targetColumn">Target Column</label>
              {availableColumns.length > 0 ? (
                <select
                  id="targetColumn"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                >
                  <option value="">Select target column</option>
                  {availableColumns.map((column, index) => (
                    <option key={index} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  id="targetColumn"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="Enter target column name"
                />
              )}
              <small>
                {availableColumns.length > 0 
                  ? "Select the column you want to predict"
                  : "Enter the column name you want to predict (or leave empty to specify later)"
                }
              </small>
            </div>

            {availableColumns.length > 0 && (
              <div className="columns-preview">
                <h4>Available Columns ({availableColumns.length})</h4>
                <div className="columns-list">
                  {availableColumns.map((column, index) => (
                    <span key={index} className="column-tag">
                      {column}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="form-actions">
              <button 
                type="submit" 
                className="btn-primary"
                disabled={loading || !file}
              >
                {loading ? 'Uploading...' : 'Upload Dataset'}
              </button>
              <button 
                type="button" 
                className="btn-secondary"
                onClick={handleCancel}
                disabled={loading}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

export default FileUploader; 