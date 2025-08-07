from django.db import models
from modeling.models import MLModel, ModelEvaluation
from ingestion.models import Dataset

class Report(models.Model):
    """Model to store generated reports"""
    title = models.CharField(max_length=255)
    report_type = models.CharField(max_length=50, choices=[
        ('data_analysis', 'Data Analysis'),
        ('preprocessing', 'Preprocessing'),
        ('model_evaluation', 'Model Evaluation'),
        ('model_comparison', 'Model Comparison'),
        ('comprehensive', 'Comprehensive'),
    ])
    
    # Related objects
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True, blank=True)
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, null=True, blank=True)
    evaluation = models.ForeignKey(ModelEvaluation, on_delete=models.CASCADE, null=True, blank=True)
    
    # Report content
    content = models.JSONField(default=dict)  # Store report data
    charts = models.JSONField(default=list)   # Store chart configurations
    summary = models.TextField(blank=True)
    
    # File
    pdf_file = models.FileField(upload_to='reports/', null=True, blank=True)
    html_file = models.FileField(upload_to='reports/', null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    generated_by = models.CharField(max_length=255, blank=True)
    
    def __str__(self):
        return self.title

class Chart(models.Model):
    """Model to store chart configurations and data"""
    name = models.CharField(max_length=255)
    chart_type = models.CharField(max_length=50, choices=[
        ('bar', 'Bar Chart'),
        ('line', 'Line Chart'),
        ('scatter', 'Scatter Plot'),
        ('histogram', 'Histogram'),
        ('box', 'Box Plot'),
        ('heatmap', 'Heatmap'),
        ('correlation', 'Correlation Matrix'),
        ('confusion_matrix', 'Confusion Matrix'),
        ('roc_curve', 'ROC Curve'),
        ('precision_recall', 'Precision-Recall Curve'),
        ('feature_importance', 'Feature Importance'),
    ])
    
    # Data and configuration
    data = models.JSONField(default=dict)
    config = models.JSONField(default=dict)
    
    # Related objects
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True, blank=True)
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, null=True, blank=True)
    evaluation = models.ForeignKey(ModelEvaluation, on_delete=models.CASCADE, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
