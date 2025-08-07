# AutoML System

A comprehensive AutoML system with AI-powered data analysis, preprocessing, modeling, and chatbot assistance.

## 🚀 Features

- **Data Ingestion**: Upload and manage datasets (CSV, Excel)
- **AI Chatbot**: Interactive data analysis assistant powered by Google Gemini
- **Data Preprocessing**: Automated data cleaning and feature engineering
- **Model Training**: AutoML with multiple algorithms (Random Forest, XGBoost, etc.)
- **Visualization**: Interactive charts and data insights
- **Reporting**: Comprehensive analysis reports

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Google Gemini API key

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Automl
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .backend_venv

# Activate virtual environment
# On Windows:
.backend_venv\Scripts\activate
# On macOS/Linux:
source .backend_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp env_example.txt .env
# Edit .env and add your Google Gemini API key
```

### 3. Frontend Setup

```bash
cd frontend/react_dashboard

# Install dependencies
npm install
```

### 4. Database Setup

```bash
cd backend

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

## 🔑 API Configuration

### Google Gemini API Setup

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Edit `backend/.env`:
```env
GOOGLE_API_KEY=your-actual-gemini-api-key-here
```

### Environment Variables

Create `backend/.env` with:
```env
# Django Settings
SECRET_KEY=your-django-secret-key
DEBUG=True

# Google Gemini API
GOOGLE_API_KEY=your-google-gemini-api-key-here

# Optional: Other LLM APIs
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### Option 1: Start

#### Backend
```bash
cd backend
source .backend_venv/bin/activate  # or .backend_venv\Scripts\activate on Windows
python manage.py runserver
```

#### Frontend
```bash
cd frontend/react_dashboard
npm start
```

## 🔧 Troubleshooting

### Chatbot Issues

#### "Failed to send message" Error

1. **Check API Key**: Ensure your Google Gemini API key is correctly set in `backend/.env`
2. **Check Backend**: Make sure Django server is running on port 8000
3. **Check Network**: Verify no firewall is blocking the connection

#### Dataset Upload Issues

1. **File Format**: Only CSV and Excel files are supported
2. **File Size**: Large files may take time to process
3. **Permissions**: Ensure the `media/` directory is writable

### Common Issues

#### Virtual Environment Not Activated
```bash
# Windows
.backend_venv\Scripts\activate

# macOS/Linux
source .backend_venv/bin/activate
```

#### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python manage.py runserver 8001
```

#### Node Modules Missing
```bash
cd frontend/react_dashboard
rm -rf node_modules package-lock.json
npm install
```

#### Database Issues
```bash
cd backend
python manage.py makemigrations
python manage.py migrate
```

## 📁 Project Structure

```
Unofficial-Automl/
├── backend/                 # Django backend
│   ├── chatbot/            # AI chatbot app
│   ├── ingestion/          # Data upload and management
│   ├── modeling/          # ML model training
│   ├── reporting/         # Analysis reports
│   ├── users/             # User management
│   └── django_api/        # Main Django project
├── frontend/              # React frontend
│   └── react_dashboard/   # Main dashboard
```

## 🤖 Chatbot Features

The AI chatbot provides:

- **Dataset Analysis**: Automatic insights and data quality assessment
- **Preprocessing Suggestions**: AI-powered data cleaning recommendations
- **Visualization Help**: Chart suggestions and creation
- **Model Recommendations**: ML algorithm suggestions
- **Interactive Q&A**: Natural language data queries

### Dataset Sampling

The chatbot intelligently samples datasets to work within API limits:
- Sends only column names and first 3 rows to Gemini
- Provides comprehensive analysis based on sample data
- Handles large datasets efficiently

## 📊 API Endpoints

### Chatbot
- `POST /api/chatbot/session/create/` - Create chat session
- `POST /api/chatbot/message/send/` - Send message
- `GET /api/chatbot/session/{id}/history/` - Get chat history
- `POST /api/chatbot/analyze/` - Analyze dataset
- `POST /api/chatbot/visualize/` - Generate visualization

### Data Management
- `POST /api/ingestion/upload/` - Upload dataset
- `GET /api/ingestion/datasets/` - List datasets
- `DELETE /api/ingestion/datasets/{id}/` - Delete dataset

## 🛡️ Security Notes

- Never commit your `.env` file to version control
- Use strong Django secret keys in production
- Configure CORS properly for production
- Set `DEBUG=False` in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the health check utility
3. Check the Django logs for backend errors
4. Check browser console for frontend errors
5. Ensure all dependencies are installed correctly

For chatbot-specific issues:
- Verify your Google Gemini API key is valid
- Check that the backend is running on port 8000
- Ensure datasets are properly uploaded before using the chatbot
