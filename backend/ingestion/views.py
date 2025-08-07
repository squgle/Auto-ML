from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
import pandas as pd
import numpy as np
import json
import math
from .models import Dataset, DataInfo
from modeling.services import MLService

ml_service = MLService()

def safe_float(val):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return val

def sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)  # Convert any other types to string

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_dataset(request):
    """Upload and analyze a dataset"""
    try:
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        file = request.FILES['file']
        name = request.data.get('name', file.name)
        description = request.data.get('description', '')
        target_column = request.data.get('target_column', '')
        
        # Read the dataset
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create dataset record
        dataset = Dataset.objects.create(
            name=name,
            description=description,
            file=file,
            rows=len(df),
            columns=len(df.columns),
            target_column=target_column
        )
        
        # Detect problem type
        if target_column and target_column in df.columns:
            problem_type = ml_service.detect_problem_type(target_column, df)
            dataset.problem_type = problem_type
            dataset.save()
        
        # Analyze data and create DataInfo records
        for column in df.columns:
            col_data = df[column]
            data_info = DataInfo.objects.create(
                dataset=dataset,
                column_name=column,
                data_type=str(col_data.dtype),
                missing_count=int(col_data.isnull().sum()),
                unique_count=int(col_data.nunique())
            )
            
            # Add statistical info for numerical columns
            if col_data.dtype in ['int64', 'float64']:
                data_info.min_value = float(col_data.min())
                data_info.max_value = float(col_data.max())
                data_info.mean_value = float(col_data.mean())
                data_info.std_value = float(col_data.std())
                data_info.save()
        
        # Get preprocessing suggestions
        preprocessing_suggestions = ml_service.suggest_preprocessing_steps(df, target_column)
        
        # Get model suggestions
        if target_column and target_column in df.columns:
            problem_type = ml_service.detect_problem_type(target_column, df)
            model_suggestions = ml_service.suggest_models(
                problem_type, len(df), len(df.columns) - 1
            )
        else:
            model_suggestions = []
        
        return Response({
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'rows': dataset.rows,
            'columns': dataset.columns,
            'target_column': dataset.target_column,
            'problem_type': dataset.problem_type,
            'preprocessing_suggestions': preprocessing_suggestions,
            'model_suggestions': model_suggestions,
            'message': 'Dataset uploaded successfully'
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_datasets(request):
    """Get all datasets"""
    try:
        datasets = Dataset.objects.all().order_by('-uploaded_at')
        data = []
        
        for dataset in datasets:
            data.append({
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'rows': dataset.rows,
                'columns': dataset.columns,
                'target_column': dataset.target_column,
                'problem_type': dataset.problem_type,
                'uploaded_at': dataset.uploaded_at.isoformat(),
                'file_size': dataset.file_size
            })
        
        return Response(data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_dataset_details(request, dataset_id):
    """Get detailed information about a dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        data_infos = DataInfo.objects.filter(dataset=dataset)
        
        # Read the dataset file
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
            df = pd.read_excel(dataset.file.path)
        else:
            return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Sanitize DataFrame for JSON serialization
        df = df.replace([np.inf, -np.inf], np.nan)
        sample_data = df.head(10).where(pd.notnull(df.head(10)), None).to_dict('records')
        
        # Get column statistics
        columns_info = []
        for data_info in data_infos:
            col_info = {
                'name': data_info.column_name,
                'data_type': data_info.data_type,
                'missing_count': data_info.missing_count,
                'unique_count': data_info.unique_count,
                'min_value': safe_float(data_info.min_value),
                'max_value': safe_float(data_info.max_value),
                'mean_value': safe_float(data_info.mean_value),
                'std_value': safe_float(data_info.std_value),
            }
            columns_info.append(col_info)
        
        # Generate basic charts
        charts = ml_service.generate_charts(df, dataset.target_column)
        # Sanitize charts for JSON serialization
        charts = sanitize_for_json(charts)
        
        return Response(sanitize_for_json({
            'dataset': {
                'id': dataset.id,
                'name': dataset.name,
                'description': dataset.description,
                'rows': dataset.rows,
                'columns': dataset.columns,
                'target_column': dataset.target_column,
                'problem_type': dataset.problem_type,
                'uploaded_at': dataset.uploaded_at.isoformat(),
            },
            'columns_info': columns_info,
            'sample_data': sample_data,
            'charts': charts
        }), status=status.HTTP_200_OK)
        
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
def delete_dataset(request, dataset_id):
    """Delete a dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        dataset.delete()
        return Response({'message': 'Dataset deleted successfully'}, status=status.HTTP_200_OK)
        
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
