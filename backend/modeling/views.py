from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from .models import MLModel, ModelEvaluation, ModelComparison
from ingestion.models import Dataset
# Removed preprocessing import - functionality moved to MLService
from .services import MLService
from django.utils import timezone

ml_service = MLService()

@api_view(['POST'])
def train_model(request):
    """Train a machine learning model"""
    try:
        data = request.data
        dataset_id = data.get('dataset_id')
        model_type = data.get('model_type')
        target_column = data.get('target_column')
        hyperparameters = data.get('hyperparameters', {})
        
        # Get dataset
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Read dataset
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
            df = pd.read_excel(dataset.file.path)
        else:
            return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Preprocess data
        df_processed, encoders, scaler = ml_service.preprocess_data(df, target_column)
        
        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col != target_column]
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use stored problem type from dataset, with fallback to detection
        if dataset.problem_type:
            problem_type = dataset.problem_type
        else:
            problem_type = ml_service.detect_problem_type(target_column, df)
        
        # Create model record
        model = MLModel.objects.create(
            name=f"{model_type}_{dataset.name}",
            model_type=model_type,
            dataset=dataset,
            target_column=target_column,
            hyperparameters=hyperparameters,
            feature_columns=feature_columns,
            status='training'
        )
        
        # Train model
        trained_model, training_time = ml_service.train_model(
            model_type, X_train, y_train, problem_type, hyperparameters
        )
        
        # Evaluate model
        metrics = ml_service.evaluate_model(trained_model, X_test, y_test, problem_type)
        
        # Get predictions for additional metrics
        y_pred = trained_model.predict(X_test)
        y_pred_proba = None
        if hasattr(trained_model, 'predict_proba'):
            y_pred_proba = trained_model.predict_proba(X_test)
        
        # Calculate additional metrics
        additional_metrics = {}
        
        if problem_type == 'classification':
            # Confusion matrix details
            from sklearn.metrics import confusion_matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                if len(cm) == 2:
                    additional_metrics['confusion_matrix'] = {
                        'true_negatives': int(cm[0, 0]),
                        'false_positives': int(cm[0, 1]),
                        'false_negatives': int(cm[1, 0]),
                        'true_positives': int(cm[1, 1])
                    }
            except Exception as e:
                print(f"Warning: Could not calculate confusion matrix: {e}")
                # Continue without confusion matrix
            
            # ROC curve data
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    # Get the positive class index
                    classes = trained_model.classes_
                    if len(classes) == 2:
                        # Use the second class (usually the positive class) for ROC curve
                        positive_class_idx = 1 if len(classes) > 1 else 0
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, positive_class_idx])
                        additional_metrics['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist()
                        }
                except Exception as e:
                    print(f"Warning: Could not calculate ROC curve: {e}")
                    # Continue without ROC curve
        
        # Cross-validation
        try:
            cv_results = ml_service.cross_validate_model(trained_model, X, y)
        except Exception as e:
            print(f"Warning: Cross-validation failed: {e}")
            cv_results = {'scores': [], 'mean': 0.0, 'std': 0.0}
        
        # Get feature importance
        try:
            feature_importance = ml_service.get_feature_importance(trained_model, feature_columns)
        except Exception as e:
            print(f"Warning: Feature importance calculation failed: {e}")
            feature_importance = {}
        
        # Save model
        model_path = f"models/{model.id}_{model_type}.joblib"
        ml_service.save_model(trained_model, model_path)
        
        # Update model record
        model.model_file = model_path
        model.training_completed = timezone.now()
        model.training_duration = training_time
        model.status = 'completed'
        model.save()
        
        # Create evaluation record with comprehensive metrics
        evaluation_data = {
            'model': model,
            **metrics,
            'cv_scores': cv_results['scores'],
            'cv_mean': cv_results['mean'],
            'cv_std': cv_results['std'],
            'feature_importance': feature_importance,
            **additional_metrics
        }
        
        evaluation = ModelEvaluation.objects.create(**evaluation_data)
        
        # Generate charts
        try:
            charts = ml_service.generate_charts(df, target_column, trained_model, y_test, y_pred)
        except Exception as e:
            print(f"Warning: Chart generation failed: {e}")
            charts = {}
        
        return Response({
            'model_id': model.id,
            'model_name': model.name,
            'training_time': training_time,
            'metrics': metrics,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'charts': charts,
            'message': 'Model trained successfully'
        }, status=status.HTTP_201_CREATED)
        
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    except ValueError as e:
        return Response({'error': f'Model training error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    except KeyError as e:
        return Response({'error': f'Unsupported model type: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': f'Internal server error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_model_metrics(request, model_id):
    """Get comprehensive model metrics for the dashboard"""
    try:
        model = MLModel.objects.get(id=model_id)
        evaluation = ModelEvaluation.objects.filter(model=model).first()
        
        if not evaluation:
            return Response({'error': 'No evaluation found for this model'}, status=status.HTTP_404_NOT_FOUND)
        
        # Use the problem type from the dataset, with fallback to model type detection
        if model.dataset.problem_type:
            is_classification = model.dataset.problem_type.lower() == 'classification'
        else:
            # Fallback: determine by model type (excluding 'logistic_regression' misclassification)
            classification_types = [
                'logistic_regression', 'svm', 'knn', 'decision_tree', 
                'random_forest', 'gradient_boosting', 'ada_boost', 'catboost',
                'xgboost', 'lightgbm', 'neural_network'
            ]
            is_classification = model.model_type.lower() in classification_types
        
        # Prepare metrics based on problem type
        if is_classification:
            metrics = {
                'accuracy': evaluation.accuracy,
                'precision': evaluation.precision,
                'recall': evaluation.recall,
                'f1_score': evaluation.f1_score,
                # 'auc_roc': evaluation.roc_auc,
                'confusion_matrix': evaluation.confusion_matrix,
                'roc_curve': getattr(evaluation, 'roc_curve', None),
                'feature_importance': evaluation.feature_importance
            }
        else:
            metrics = {
                'mse': evaluation.mse,
                'rmse': np.sqrt(evaluation.mse) if evaluation.mse else None,
                'mae': evaluation.mae,
                'r2_score': evaluation.r2_score,
                'explained_variance': evaluation.r2_score,  # Using R² as proxy
                'feature_importance': evaluation.feature_importance
            }
        
        return Response({
            'metrics': metrics,
            'model_type': model.model_type,
            'problem_type': 'classification' if is_classification else 'regression',
            'dataset_problem_type': model.dataset.problem_type  # Include the original dataset problem type
        }, status=status.HTTP_200_OK)
        
    except MLModel.DoesNotExist:
        return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_models(request):
    """Get all trained models"""
    try:
        models = MLModel.objects.all().order_by('-training_started')
        data = []
        
        for model in models:
            data.append({
                'id': model.id,
                'name': model.name,
                'model_type': model.model_type,
                'dataset_name': model.dataset.name,
                'target_column': model.target_column,
                'status': model.status,
                'training_started': model.training_started.isoformat(),
                'training_completed': model.training_completed.isoformat() if model.training_completed else None,
                'training_duration': model.training_duration
            })
        
        return Response(data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_model_details(request, model_id):
    """Get detailed information about a model"""
    try:
        model = MLModel.objects.get(id=model_id)
        evaluation = ModelEvaluation.objects.filter(model=model).first()
        
        data = {
            'model': {
                'id': model.id,
                'name': model.name,
                'model_type': model.model_type,
                'dataset_name': model.dataset.name,
                'target_column': model.target_column,
                'hyperparameters': model.hyperparameters,
                'feature_columns': model.feature_columns,
                'status': model.status,
                'training_started': model.training_started.isoformat(),
                'training_completed': model.training_completed.isoformat() if model.training_completed else None,
                'training_duration': model.training_duration
            }
        }
        
        if evaluation:
            data['evaluation'] = {
                'accuracy': evaluation.accuracy,
                'precision': evaluation.precision,
                'recall': evaluation.recall,
                'f1_score': evaluation.f1_score,
                'roc_auc': evaluation.roc_auc,
                'mse': evaluation.mse,
                'mae': evaluation.mae,
                'r2_score': evaluation.r2_score,
                'cv_mean': evaluation.cv_mean,
                'cv_std': evaluation.cv_std,
                'feature_importance': evaluation.feature_importance,
                'confusion_matrix': evaluation.confusion_matrix,
                'evaluated_at': evaluation.evaluated_at.isoformat()
            }
        
        return Response(data, status=status.HTTP_200_OK)
        
    except MLModel.DoesNotExist:
        return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def compare_models(request):
    """Compare multiple models"""
    try:
        data = request.data
        dataset_id = data.get('dataset_id')
        model_types = data.get('model_types', [])
        target_column = data.get('target_column')
        
        if not model_types:
            return Response({'error': 'No models specified'}, status=status.HTTP_400_BAD_REQUEST)
        
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Read dataset
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
            df = pd.read_excel(dataset.file.path)
        else:
            return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Preprocess data
        df_processed, encoders, scaler = ml_service.preprocess_data(df, target_column)
        
        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col != target_column]
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use stored problem type from dataset, with fallback to detection
        if dataset.problem_type:
            problem_type = dataset.problem_type
        else:
            problem_type = ml_service.detect_problem_type(target_column, df)
        
        # Train and evaluate models
        models = []
        evaluations = []
        comparison_metrics = {}
        
        for model_type in model_types:
            # Train model
            trained_model, training_time = ml_service.train_model(
                model_type, X_train, y_train, problem_type
            )
            
            # Evaluate model
            metrics = ml_service.evaluate_model(trained_model, X_test, y_test, problem_type)
            
            # Create model record
            model = MLModel.objects.create(
                name=f"{model_type}_{dataset.name}",
                model_type=model_type,
                dataset=dataset,
                target_column=target_column,
                feature_columns=feature_columns,
                status='completed',
                training_completed=timezone.now(),
                training_duration=training_time
            )
            
            # Save model
            model_path = f"models/{model.id}_{model_type}.joblib"
            ml_service.save_model(trained_model, model_path)
            model.model_file = model_path
            model.save()
            
            # Create evaluation record
            evaluation = ModelEvaluation.objects.create(
                model=model,
                **metrics
            )
            
            models.append(model)
            evaluations.append(evaluation)
            
            # Store metrics for comparison
            comparison_metrics[model_type] = metrics
        
        # Find best model
        if problem_type == 'classification':
            best_metric = 'accuracy'
        else:
            best_metric = 'r2_score'
        
        best_model = None
        best_score = -1
        
        for evaluation in evaluations:
            score = getattr(evaluation, best_metric, 0)
            if score and score > best_score:
                best_score = score
                best_model = evaluation.model
        
        # Create comparison record
        comparison = ModelComparison.objects.create(
            name=f"Comparison_{dataset.name}",
            dataset=dataset,
            best_model=best_model
        )
        comparison.models.set(models)
        comparison.comparison_metrics = comparison_metrics
        comparison.save()
        
        return Response({
            'comparison_id': comparison.id,
            'models': [model.name for model in models],
            'comparison_metrics': comparison_metrics,
            'best_model': best_model.name if best_model else None,
            'best_score': best_score,
            'message': 'Model comparison completed'
        }, status=status.HTTP_201_CREATED)
        
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_model_suggestions(request, dataset_id):
    """Get intelligent model suggestions for a dataset using LangGraph and Gemini"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Read dataset
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
            df = pd.read_excel(dataset.file.path)
        else:
            return Response({'error': 'Unsupported file format'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not dataset.target_column or dataset.target_column not in df.columns:
            return Response({'error': 'Target column not specified or not found'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get intelligent suggestions
        intelligent_suggestions = ml_service.get_intelligent_model_suggestions(df, dataset.target_column)
        
        # Get traditional suggestions as fallback
        if dataset.problem_type:
            problem_type = dataset.problem_type
        else:
            problem_type = ml_service.detect_problem_type(dataset.target_column, df)
        traditional_suggestions = ml_service.suggest_models(
            problem_type, len(df), len(df.columns) - 1
        )
        
        return Response({
            'problem_type': problem_type,
            'intelligent_suggestions': intelligent_suggestions,
            'traditional_suggestions': traditional_suggestions,
            'dataset_size': len(df),
            'feature_count': len(df.columns) - 1
        }, status=status.HTTP_200_OK)
        
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def get_optimized_hyperparameters(request):
    """Get optimized hyperparameters for a specific model using LLM"""
    try:
        data = request.data
        model_name = data.get('model_name')
        problem_type = data.get('problem_type')
        dataset_size = data.get('dataset_size')
        
        if not all([model_name, problem_type, dataset_size]):
            return Response({'error': 'Missing required parameters'}, status=status.HTTP_400_BAD_REQUEST)
        
        hyperparameters = ml_service.get_optimized_hyperparameters(model_name, problem_type, dataset_size)
        
        return Response({
            'model_name': model_name,
            'hyperparameters': hyperparameters
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def explain_model_choice(request):
    """Get explanation for why a specific model was chosen"""
    try:
        data = request.data
        model_name = data.get('model_name')
        dataset_info = data.get('dataset_info')
        
        if not model_name or not dataset_info:
            return Response({'error': 'Missing required parameters'}, status=status.HTTP_400_BAD_REQUEST)
        
        explanation = ml_service.explain_model_choice(model_name, dataset_info)
        
        return Response({
            'model_name': model_name,
            'explanation': explanation
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
