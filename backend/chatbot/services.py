import os
import json
import pandas as pd
import numpy as np
from decouple import config
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Configure Gemini API
GOOGLE_API_KEY = config('GOOGLE_API_KEY', default='your-google-api-key')
genai.configure(api_key=GOOGLE_API_KEY)

class ChatbotService:
    """AI Chatbot service using Gemini API for data analysis and insights"""
    
    def __init__(self):
        # Check if API key is properly configured
        if GOOGLE_API_KEY == 'your-google-api-key':
            raise ValueError("GOOGLE_API_KEY not properly configured. Please set it in your .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY
        )
    
    def _sample_dataset_for_llm(self, df, max_rows=3):
        """Sample dataset to send to LLM - only column names and first few rows"""
        try:
            # Get basic dataset info
            dataset_info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(max_rows).to_dict('records'),
                'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns)
            }
            return dataset_info
        except Exception as e:
            return {'error': f"Error sampling dataset: {str(e)}"}
    
    def analyze_dataset(self, df, dataset_name):
        """Analyze dataset and provide comprehensive insights"""
        try:
            # Sample dataset for LLM
            dataset_info = self._sample_dataset_for_llm(df)
            if 'error' in dataset_info:
                return dataset_info
            
            # Basic dataset statistics
            stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'data_types': df.dtypes.to_dict(),
                'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
            
            # Generate insights using Gemini with sampled data
            prompt = f"""
            Analyze this dataset: {dataset_name}
            
            Dataset Statistics:
            - Rows: {stats['rows']}
            - Columns: {stats['columns']}
            - Missing values: {stats['missing_values']}
            - Numerical columns: {stats['numerical_columns']}
            - Categorical columns: {stats['categorical_columns']}
            - Memory usage: {stats['memory_usage']:.2f} MB
            
            Column names: {dataset_info['columns']}
            Data types: {dataset_info['data_types']}
            Missing values per column: {dataset_info['missing_values']}
            
            Sample data (first 3 rows):
            {json.dumps(dataset_info['sample_data'], indent=2)}
            
            Please provide:
            1. Key insights about the data quality
            2. Potential data preprocessing steps needed
            3. Suggestions for analysis
            4. Any data quality issues to address
            
            Format your response in a clear, helpful way for a data scientist.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {
                'statistics': stats,
                'insights': response.content,
                'dataset_name': dataset_name
            }
            
        except Exception as e:
            return {
                'error': f"Error analyzing dataset: {str(e)}",
                'statistics': stats if 'stats' in locals() else {}
            }
    
    def answer_question(self, question, df, dataset_name, context=""):
        """Answer user questions about the dataset using Gemini"""
        try:
            # Sample dataset for LLM
            dataset_info = self._sample_dataset_for_llm(df)
            if 'error' in dataset_info:
                return {
                    'error': dataset_info['error'],
                    'answer': "I'm sorry, I encountered an error while processing the dataset. Please try again."
                }
            
            # Prepare dataset summary for context
            dataset_summary = f"""
            Dataset: {dataset_name}
            Shape: {dataset_info['shape']}
            Columns: {dataset_info['columns']}
            Data types: {dataset_info['data_types']}
            Missing values: {dataset_info['missing_values']}
            Numerical columns: {dataset_info['numerical_columns']}
            Categorical columns: {dataset_info['categorical_columns']}
            
            Sample data (first 3 rows):
            {json.dumps(dataset_info['sample_data'], indent=2)}
            """
            
            prompt = f"""
            Context: {context}
            
            Dataset Information:
            {dataset_summary}
            
            User Question: {question}
            
            Please provide a helpful, accurate answer about this dataset. 
            If the question is about data analysis, provide specific insights based on the sample data.
            If the question is about preprocessing, suggest appropriate steps.
            If the question is about visualization, suggest what charts would be useful.
            
            Be conversational but professional. Include specific details from the data when relevant.
            If you need more data to answer accurately, mention that you're working with a sample.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {
                'answer': response.content,
                'question': question,
                'dataset_name': dataset_name
            }
            
        except Exception as e:
            return {
                'error': f"Error answering question: {str(e)}",
                'answer': "I'm sorry, I encountered an error while processing your question. Please try again."
            }
    
    def generate_visualization_suggestion(self, df, dataset_name, user_request):
        """Suggest and generate visualizations based on user request"""
        try:
            # Sample dataset for LLM
            dataset_info = self._sample_dataset_for_llm(df)
            if 'error' in dataset_info:
                return {'error': dataset_info['error']}
            
            numerical_cols = dataset_info['numerical_columns']
            categorical_cols = dataset_info['categorical_columns']
            
            prompt = f"""
            Dataset: {dataset_name}
            Numerical columns: {numerical_cols}
            Categorical columns: {categorical_cols}
            
            Sample data:
            {json.dumps(dataset_info['sample_data'], indent=2)}
            
            User request: {user_request}
            
            Based on the dataset structure and user request, suggest the most appropriate visualization.
            Consider:
            1. Data types (numerical vs categorical)
            2. Number of unique values
            3. Distribution patterns
            4. User's specific question
            
            Return a JSON response with:
            - suggested_chart_type: (histogram, scatter, bar, box, heatmap, etc.)
            - x_axis: column name for x-axis
            - y_axis: column name for y-axis (if applicable)
            - title: suggested chart title
            - description: why this visualization is appropriate
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Try to parse JSON response
            try:
                suggestion = json.loads(response.content)
            except:
                # Fallback to simple suggestion
                suggestion = {
                    'suggested_chart_type': 'histogram',
                    'x_axis': numerical_cols[0] if numerical_cols else df.columns[0],
                    'title': f'Distribution of {dataset_name}',
                    'description': 'Default visualization suggestion'
                }
            
            return suggestion
            
        except Exception as e:
            return {
                'error': f"Error generating visualization suggestion: {str(e)}"
            }
    
    def create_chart(self, df, chart_type, x_axis, y_axis=None, title=None):
        """Create a chart using Plotly based on the specified parameters"""
        try:
            if chart_type == 'histogram':
                fig = px.histogram(df, x=x_axis, title=title or f'Distribution of {x_axis}')
            elif chart_type == 'scatter':
                if y_axis:
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=title or f'{x_axis} vs {y_axis}')
                else:
                    return {'error': 'Scatter plot requires both x and y axes'}
            elif chart_type == 'bar':
                fig = px.bar(df, x=x_axis, title=title or f'Bar chart of {x_axis}')
            elif chart_type == 'box':
                fig = px.box(df, x=x_axis, title=title or f'Box plot of {x_axis}')
            elif chart_type == 'heatmap':
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr_matrix, title=title or 'Correlation Heatmap')
            else:
                # Default to histogram
                fig = px.histogram(df, x=x_axis, title=title or f'Distribution of {x_axis}')
            
            # Convert to base64 for embedding in response
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode()
            
            return {
                'chart_type': chart_type,
                'image_base64': img_base64,
                'title': fig.layout.title.text if fig.layout.title else title
            }
            
        except Exception as e:
            return {
                'error': f"Error creating chart: {str(e)}"
            }
    
    def get_data_preprocessing_suggestions(self, df, dataset_name):
        """Get AI-powered suggestions for data preprocessing"""
        try:
            # Sample dataset for LLM
            dataset_info = self._sample_dataset_for_llm(df)
            if 'error' in dataset_info:
                return {'error': dataset_info['error']}
            
            missing_data = dataset_info['missing_values']
            categorical_cols = dataset_info['categorical_columns']
            numerical_cols = dataset_info['numerical_columns']
            
            prompt = f"""
            Dataset: {dataset_name}
            
            Data Quality Issues:
            - Missing values: {missing_data}
            - Categorical columns: {categorical_cols}
            - Numerical columns: {numerical_cols}
            
            Sample data:
            {json.dumps(dataset_info['sample_data'], indent=2)}
            
            Please provide specific preprocessing suggestions including:
            1. How to handle missing values for each column type
            2. Encoding strategies for categorical variables
            3. Scaling/normalization needs for numerical variables
            4. Feature engineering opportunities
            5. Outlier detection and handling
            
            Format as a clear, actionable list.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {
                'suggestions': response.content,
                'dataset_name': dataset_name
            }
            
        except Exception as e:
            return {
                'error': f"Error getting preprocessing suggestions: {str(e)}"
            } 