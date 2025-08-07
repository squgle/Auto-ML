from django.db import models
# Removed preprocessing import - functionality moved to MLService
from ingestion.models import Dataset

class MLModel(models.Model):
    """Model to store trained machine learning models"""
    name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=50, choices=[
        ('random_forest', 'Random Forest'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('catboost', 'CatBoost'),
        ('svm', 'Support Vector Machine'),
        ('logistic_regression', 'Logistic Regression'),
        ('linear_regression', 'Linear Regression'),
        ('neural_network', 'Neural Network'),
        ('knn', 'K-Nearest Neighbors'),
        ('decision_tree', 'Decision Tree'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('ada_boost', 'AdaBoost'),
        ('elastic_net', 'Elastic Net'),
        ('ridge', 'Ridge Regression'),
        ('lasso', 'Lasso Regression'),
    ])
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    # preprocessed_dataset field removed - preprocessing handled in MLService
    
    # Model parameters
    hyperparameters = models.JSONField(default=dict)
    feature_columns = models.JSONField(default=list)
    target_column = models.CharField(max_length=255)
    
    # Training info
    training_started = models.DateTimeField(auto_now_add=True)
    training_completed = models.DateTimeField(null=True, blank=True)
    training_duration = models.FloatField(null=True, blank=True)  # in seconds
    
    # Model file
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    
    # Status
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"

class ModelEvaluation(models.Model):
    """Model to store model evaluation metrics"""
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    
    # Classification Metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    
    # Regression Metrics
    mse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    
    # Cross-validation results
    cv_scores = models.JSONField(default=list)
    cv_mean = models.FloatField(null=True, blank=True)
    cv_std = models.FloatField(null=True, blank=True)
    
    # Confusion matrix (for classification)
    confusion_matrix = models.JSONField(default=dict)
    
    # ROC curve data (for classification)
    roc_curve = models.JSONField(default=dict)
    
    # Feature importance
    feature_importance = models.JSONField(default=dict)
    
    evaluated_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Evaluation for {self.model.name}"

class ModelComparison(models.Model):
    """Model to store model comparison results"""
    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    mlmodel = models.ManyToManyField(MLModel)
    best_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, null=True, blank=True, related_name='best_in_comparison')
    comparison_metrics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
