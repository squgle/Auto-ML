from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import pandas as pd
import json
import uuid
from datetime import datetime
from .models import ChatSession, ChatMessage
from ingestion.models import Dataset
from .services import ChatbotService
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Initialize chatbot service with error handling
try:
    chatbot_service = ChatbotService()
except ValueError as e:
    logger.error(f"ChatbotService initialization failed: {e}")
    chatbot_service = None

@api_view(['POST'])
def create_chat_session(request):
    """Create a new chat session"""
    try:
        data = request.data
        dataset_id = data.get('dataset_id')
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Get dataset if provided
        dataset = None
        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id)
            except Dataset.DoesNotExist:
                return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Create chat session
        session = ChatSession.objects.create(
            session_id=session_id,
            dataset=dataset
        )
        
        # Add welcome message
        welcome_message = "Hello! I'm your AI data assistant. I can help you analyze your data, answer questions, and create visualizations. What would you like to know about your dataset?"
        ChatMessage.objects.create(
            session=session,
            message_type='assistant',
            content=welcome_message
        )
        
        return Response({
            'session_id': session_id,
            'dataset_name': dataset.name if dataset else None,
            'welcome_message': welcome_message,
            'message': 'Chat session created successfully'
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        return Response({'error': 'Failed to create chat session'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def send_message(request):
    """Send a message to the chatbot and get response"""
    try:
        # Check if chatbot service is available
        if chatbot_service is None:
            return Response({
                'error': 'Chatbot service is not available. Please check your API key configuration.'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        data = request.data
        session_id = data.get('session_id')
        message = data.get('message')
        
        if not session_id or not message:
            return Response({'error': 'Session ID and message are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get chat session
        try:
            session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            return Response({'error': 'Chat session not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Save user message
        ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=message
        )
        
        # Get dataset if available
        df = None
        dataset_name = None
        if session.dataset:
            try:
                dataset = session.dataset
                dataset_name = dataset.name
                
                # Read dataset
                if dataset.file.name.endswith('.csv'):
                    df = pd.read_csv(dataset.file.path)
                elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
                    df = pd.read_excel(dataset.file.path)
                else:
                    return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                logger.error(f"Error reading dataset: {e}")
                return Response({'error': 'Failed to read dataset file'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get previous messages for context
        previous_messages = list(session.messages.filter(message_type__in=['user', 'assistant']).order_by('timestamp'))
        context = "\n".join([f"{msg.message_type}: {msg.content}" for msg in previous_messages[-5:]])  # Last 5 messages
        
        # Generate response using chatbot service
        try:
            if df is not None:
                response_data = chatbot_service.answer_question(message, df, dataset_name, context)
            else:
                response_data = {
                    'answer': "I don't have access to a dataset yet. Please upload a dataset first to get data-specific insights.",
                    'question': message,
                    'dataset_name': None
                }
            
            # Check for errors in response
            if 'error' in response_data:
                logger.error(f"Chatbot service error: {response_data['error']}")
                response_data['answer'] = "I'm sorry, I encountered an error while processing your request. Please try again."
            
        except Exception as e:
            logger.error(f"Error generating chatbot response: {e}")
            response_data = {
                'answer': "I'm sorry, I encountered an error while processing your request. Please try again.",
                'question': message,
                'dataset_name': dataset_name
            }
        
        # Save assistant response
        ChatMessage.objects.create(
            session=session,
            message_type='assistant',
            content=response_data['answer']
        )
        
        return Response({
            'session_id': session_id,
            'user_message': message,
            'assistant_response': response_data['answer'],
            'dataset_name': dataset_name
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in send_message: {e}")
        return Response({'error': 'Failed to send message. Global Error msg'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_chat_history(request, session_id):
    """Get chat history for a session"""
    try:
        session = ChatSession.objects.get(session_id=session_id)
        messages = session.messages.all()
        
        history = []
        for message in messages:
            history.append({
                'id': message.id,
                'type': message.message_type,
                'content': message.content,
                'timestamp': message.timestamp.isoformat()
            })
        
        return Response({
            'session_id': session_id,
            'dataset_name': session.dataset.name if session.dataset else None,
            'messages': history
        }, status=status.HTTP_200_OK)
        
    except ChatSession.DoesNotExist:
        return Response({'error': 'Chat session not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return Response({'error': 'Failed to get chat history'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def analyze_dataset(request):
    """Analyze a dataset and provide AI insights"""
    try:
        # Check if chatbot service is available
        if chatbot_service is None:
            return Response({
                'error': 'Chatbot service is not available. Please check your API key configuration.'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        data = request.data
        dataset_id = data.get('dataset_id')
        
        if not dataset_id:
            return Response({'error': 'Dataset ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Read dataset
        try:
            if dataset.file.name.endswith('.csv'):
                df = pd.read_csv(dataset.file.path)
            elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
                df = pd.read_excel(dataset.file.path)
            else:
                return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            return Response({'error': 'Failed to read dataset file'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Analyze dataset using chatbot service
        try:
            analysis = chatbot_service.analyze_dataset(df, dataset.name)
            if 'error' in analysis:
                logger.error(f"Dataset analysis error: {analysis['error']}")
                return Response({'error': 'Failed to analyze dataset'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            return Response({'error': 'Failed to analyze dataset'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response({
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'analysis': analysis
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_dataset: {e}")
        return Response({'error': 'Failed to analyze dataset'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def generate_visualization(request):
    """Generate a visualization based on user request"""
    try:
        # Check if chatbot service is available
        if chatbot_service is None:
            return Response({
                'error': 'Chatbot service is not available. Please check your API key configuration.'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        data = request.data
        dataset_id = data.get('dataset_id')
        user_request = data.get('request')
        
        if not dataset_id or not user_request:
            return Response({'error': 'Dataset ID and request are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Read dataset
        try:
            if dataset.file.name.endswith('.csv'):
                df = pd.read_csv(dataset.file.path)
            elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
                df = pd.read_excel(dataset.file.path)
            else:
                return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            return Response({'error': 'Failed to read dataset file'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get visualization suggestion
        try:
            suggestion = chatbot_service.generate_visualization_suggestion(df, dataset.name, user_request)
            
            if 'error' in suggestion:
                logger.error(f"Visualization suggestion error: {suggestion['error']}")
                return Response({'error': 'Failed to generate visualization suggestion'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Error generating visualization suggestion: {e}")
            return Response({'error': 'Failed to generate visualization suggestion'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Create the chart
        try:
            chart_data = chatbot_service.create_chart(
                df, 
                suggestion['suggested_chart_type'],
                suggestion['x_axis'],
                suggestion.get('y_axis'),
                suggestion.get('title')
            )
            
            if 'error' in chart_data:
                logger.error(f"Chart creation error: {chart_data['error']}")
                return Response({'error': 'Failed to create chart'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return Response({'error': 'Failed to create chart'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response({
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'suggestion': suggestion,
            'chart': chart_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_visualization: {e}")
        return Response({'error': 'Failed to generate visualization'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def get_preprocessing_suggestions(request):
    """Get AI-powered preprocessing suggestions for a dataset"""
    try:
        # Check if chatbot service is available
        if chatbot_service is None:
            return Response({
                'error': 'Chatbot service is not available. Please check your API key configuration.'
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        data = request.data
        dataset_id = data.get('dataset_id')
        
        if not dataset_id:
            return Response({'error': 'Dataset ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Read dataset
        try:
            if dataset.file.name.endswith('.csv'):
                df = pd.read_csv(dataset.file.path)
            elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
                df = pd.read_excel(dataset.file.path)
            else:
                return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            return Response({'error': 'Failed to read dataset file'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get preprocessing suggestions
        try:
            suggestions = chatbot_service.get_data_preprocessing_suggestions(df, dataset.name)
            
            if 'error' in suggestions:
                logger.error(f"Preprocessing suggestions error: {suggestions['error']}")
                return Response({'error': 'Failed to get preprocessing suggestions'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Error getting preprocessing suggestions: {e}")
            return Response({'error': 'Failed to get preprocessing suggestions'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response({
            'dataset_id': dataset_id,
            'dataset_name': dataset.name,
            'suggestions': suggestions
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Unexpected error in get_preprocessing_suggestions: {e}")
        return Response({'error': 'Failed to get preprocessing suggestions'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 