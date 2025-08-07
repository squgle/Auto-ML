from django.db import models
from django.contrib.auth.models import User
import os

class Dataset(models.Model):
    """Model to store uploaded datasets"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file_size = models.BigIntegerField(default=0)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    
    # Data info
    target_column = models.CharField(max_length=255, blank=True, null=True)
    problem_type = models.CharField(max_length=50, choices=[
        ('classification', 'Classification'),
        ('regression', 'Regression'),
        ('clustering', 'Clustering'),
    ], blank=True, null=True)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
        super().save(*args, **kwargs)

class DataInfo(models.Model):
    """Model to store data analysis information"""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    column_name = models.CharField(max_length=255)
    data_type = models.CharField(max_length=50)
    missing_count = models.IntegerField(default=0)
    unique_count = models.IntegerField(default=0)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    mean_value = models.FloatField(null=True, blank=True)
    std_value = models.FloatField(null=True, blank=True)
    
    class Meta:
        unique_together = ['dataset', 'column_name']
