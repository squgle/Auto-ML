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
from .models import Report, Chart
from modeling.models import MLModel, ModelEvaluation, ModelComparison
from ingestion.models import Dataset
from modeling.services import MLService
from django.http import HttpResponse
# Remove WeasyPrint and template imports
# from django.template.loader import render_to_string
# from weasyprint import HTML
# import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

ml_service = MLService()

@api_view(['POST'])
def generate_report(request):
    """Generate a comprehensive report"""
    try:
        data = request.data
        report_type = data.get('report_type', 'comprehensive')
        dataset_id = data.get('dataset_id')
        model_id = data.get('model_id')
        evaluation_id = data.get('evaluation_id')
        
        report_data = {
            'title': f"{report_type.title()} Report",
            'report_type': report_type,
            'content': {},
            'charts': [],
            'summary': ''
        }
        
        if dataset_id:
            dataset = Dataset.objects.get(id=dataset_id)
            report_data['dataset'] = dataset
            
            # Read dataset for analysis
            if dataset.file.name.endswith('.csv'):
                df = pd.read_csv(dataset.file.path)
            elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
                df = pd.read_excel(dataset.file.path)
            
            # Data analysis content
            report_data['content']['dataset_info'] = {
                'name': dataset.name,
                'rows': len(df),
                'columns': len(df.columns),
                'target_column': dataset.target_column,
                'problem_type': dataset.problem_type,
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Generate charts for dataset
            charts = ml_service.generate_charts(df, dataset.target_column)
            report_data['charts'].extend(charts.values())
        
        if model_id:
            model = MLModel.objects.get(id=model_id)
            evaluation = ModelEvaluation.objects.filter(model=model).first()
            report_data['model'] = model
            
            # Model evaluation content
            if evaluation:
                report_data['content']['model_evaluation'] = {
                    'model_name': model.name,
                    'model_type': model.model_type,
                    'target_column': model.target_column,
                    'hyperparameters': model.hyperparameters,
                    'training_duration': model.training_duration,
                    'metrics': {
                        'accuracy': evaluation.accuracy,
                        'precision': evaluation.precision,
                        'recall': evaluation.recall,
                        'f1_score': evaluation.f1_score,
                        'roc_auc': evaluation.roc_auc,
                        'mse': evaluation.mse,
                        'mae': evaluation.mae,
                        'r2_score': evaluation.r2_score,
                        'cv_mean': evaluation.cv_mean,
                        'cv_std': evaluation.cv_std
                    },
                    'feature_importance': evaluation.feature_importance,
                    'confusion_matrix': evaluation.confusion_matrix
                }
        
        # Generate summary
        summary_parts = []
        if 'dataset_info' in report_data['content']:
            summary_parts.append(f"Dataset '{report_data['content']['dataset_info']['name']}' contains {report_data['content']['dataset_info']['rows']} rows and {report_data['content']['dataset_info']['columns']} columns.")
        
        if 'model_evaluation' in report_data['content']:
            eval_data = report_data['content']['model_evaluation']
            if eval_data['metrics']['accuracy']:
                summary_parts.append(f"Model achieved {eval_data['metrics']['accuracy']:.3f} accuracy.")
            elif eval_data['metrics']['r2_score']:
                summary_parts.append(f"Model achieved {eval_data['metrics']['r2_score']:.3f} R² score.")
        
        report_data['summary'] = ' '.join(summary_parts)
        
        # Create report record
        report = Report.objects.create(
            title=report_data['title'],
            report_type=report_type,
            dataset=dataset if dataset_id else None,
            model=model if model_id else None,
            evaluation=evaluation if evaluation_id else None,
            content=report_data['content'],
            charts=report_data['charts'],
            summary=report_data['summary'],
            generated_by='API'
        )
        
        return Response({
            'report_id': report.id,
            'title': report.title,
            'summary': report.summary,
            'content': report.content,
            'charts': report.charts,
            'message': 'Report generated successfully'
        }, status=status.HTTP_201_CREATED)
        
    except Dataset.DoesNotExist:
        return Response({'error': 'Dataset not found'}, status=status.HTTP_404_NOT_FOUND)
    except MLModel.DoesNotExist:
        return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_reports(request):
    """Get all generated reports"""
    try:
        reports = Report.objects.all().order_by('-created_at')
        data = []
        
        for report in reports:
            data.append({
                'id': report.id,
                'title': report.title,
                'report_type': report.report_type,
                'dataset_name': report.dataset.name if report.dataset else None,
                'model_name': report.model.name if report.model else None,
                'summary': report.summary,
                'created_at': report.created_at.isoformat(),
                'generated_by': report.generated_by
            })
        
        return Response(data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_report_details(request, report_id):
    """Get detailed information about a report"""
    try:
        report = Report.objects.get(id=report_id)
        
        data = {
            'id': report.id,
            'title': report.title,
            'report_type': report.report_type,
            'content': report.content,
            'charts': report.charts,
            'summary': report.summary,
            'created_at': report.created_at.isoformat(),
            'generated_by': report.generated_by
        }
        
        if report.dataset:
            data['dataset'] = {
                'id': report.dataset.id,
                'name': report.dataset.name
            }
        
        if report.model:
            data['model'] = {
                'id': report.model.id,
                'name': report.model.name,
                'model_type': report.model.model_type
            }
        
        return Response(data, status=status.HTTP_200_OK)
        
    except Report.DoesNotExist:
        return Response({'error': 'Report not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def generate_model_comparison_report(request):
    """Generate a model comparison report"""
    try:
        data = request.data
        comparison_id = data.get('comparison_id')
        
        comparison = ModelComparison.objects.get(id=comparison_id)
        models = comparison.models.all()
        
        # Get evaluation data for all models
        model_evaluations = []
        for model in models:
            evaluation = ModelEvaluation.objects.filter(model=model).first()
            if evaluation:
                model_evaluations.append({
                    'model_name': model.name,
                    'model_type': model.model_type,
                    'evaluation': {
                        'accuracy': evaluation.accuracy,
                        'precision': evaluation.precision,
                        'recall': evaluation.recall,
                        'f1_score': evaluation.f1_score,
                        'roc_auc': evaluation.roc_auc,
                        'mse': evaluation.mse,
                        'mae': evaluation.mae,
                        'r2_score': evaluation.r2_score,
                        'cv_mean': evaluation.cv_mean,
                        'cv_std': evaluation.cv_std
                    }
                })
        
        # Create comparison charts
        charts = []
        
        # Performance comparison chart
        if model_evaluations:
            model_names = [m['model_name'] for m in model_evaluations]
            accuracies = [m['evaluation'].get('accuracy', 0) for m in model_evaluations]
            f1_scores = [m['evaluation'].get('f1_score', 0) for m in model_evaluations]
            
            # Create comparison chart data
            comparison_chart = {
                'type': 'bar',
                'data': {
                    'labels': model_names,
                    'datasets': [
                        {
                            'label': 'Accuracy',
                            'data': accuracies,
                            'backgroundColor': 'rgba(54, 162, 235, 0.5)'
                        },
                        {
                            'label': 'F1 Score',
                            'data': f1_scores,
                            'backgroundColor': 'rgba(255, 99, 132, 0.5)'
                        }
                    ]
                },
                'options': {
                    'title': 'Model Performance Comparison',
                    'scales': {
                        'y': {'beginAtZero': True, 'max': 1}
                    }
                }
            }
            charts.append(comparison_chart)
        
        # Create report content
        content = {
            'comparison_info': {
                'name': comparison.name,
                'dataset_name': comparison.dataset.name,
                'best_model': comparison.best_model.name if comparison.best_model else None,
                'total_models': len(models)
            },
            'model_evaluations': model_evaluations,
            'comparison_metrics': comparison.comparison_metrics
        }
        
        # Generate summary
        best_model_name = comparison.best_model.name if comparison.best_model else 'None'
        summary = f"Model comparison of {len(models)} models on dataset '{comparison.dataset.name}'. Best performing model: {best_model_name}."
        
        # Create report
        report = Report.objects.create(
            title=f"Model Comparison Report - {comparison.name}",
            report_type='model_comparison',
            dataset=comparison.dataset,
            content=content,
            charts=charts,
            summary=summary,
            generated_by='API'
        )
        
        return Response({
            'report_id': report.id,
            'title': report.title,
            'summary': report.summary,
            'content': content,
            'charts': charts,
            'message': 'Model comparison report generated successfully'
        }, status=status.HTTP_201_CREATED)
        
    except ModelComparison.DoesNotExist:
        return Response({'error': 'Model comparison not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_available_charts(request):
    """Get available chart types"""
    chart_types = [
        {'id': 'bar', 'name': 'Bar Chart', 'description': 'Compare values across categories'},
        {'id': 'line', 'name': 'Line Chart', 'description': 'Show trends over time'},
        {'id': 'scatter', 'name': 'Scatter Plot', 'description': 'Show relationship between two variables'},
        {'id': 'histogram', 'name': 'Histogram', 'description': 'Show distribution of values'},
        {'id': 'box', 'name': 'Box Plot', 'description': 'Show distribution and outliers'},
        {'id': 'heatmap', 'name': 'Heatmap', 'description': 'Show correlation matrix'},
        {'id': 'correlation', 'name': 'Correlation Matrix', 'description': 'Show feature correlations'},
        {'id': 'confusion_matrix', 'name': 'Confusion Matrix', 'description': 'Show classification results'},
        {'id': 'roc_curve', 'name': 'ROC Curve', 'description': 'Show classification performance'},
        {'id': 'precision_recall', 'name': 'Precision-Recall Curve', 'description': 'Show classification performance'},
        {'id': 'feature_importance', 'name': 'Feature Importance', 'description': 'Show feature importance'}
    ]
    
    return Response(chart_types, status=status.HTTP_200_OK)

def download_report_pdf(request, report_id):
    """
    Generate and download a report as PDF using ReportLab (pure Python).
    """
    try:
        report = Report.objects.get(id=report_id)
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, height - 50, report.title)

        # Meta
        p.setFont("Helvetica", 10)
        p.drawString(50, height - 70, f"Type: {report.report_type}")
        p.drawString(50, height - 85, f"Generated: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        # Summary
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 110, "Summary")
        p.setFont("Helvetica", 11)
        text = p.beginText(50, height - 130)
        for line in (report.summary or "").splitlines():
            text.textLine(line)
        p.drawText(text)

        # Content
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 180, "Content")
        p.setFont("Helvetica", 10)
        text = p.beginText(50, height - 200)
        content_str = str(report.content)
        for line in content_str.splitlines():
            text.textLine(line)
        p.drawText(text)

        p.showPage()
        p.save()
        buffer.seek(0)
        pdf = buffer.getvalue()
        buffer.close()

        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{report.title}.pdf"'
        return response
    except Report.DoesNotExist:
        return Response({'error': 'Report not found'}, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
