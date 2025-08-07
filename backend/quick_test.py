#!/usr/bin/env python3
"""
Quick test to check if logistic regression is working
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
django.setup()

from modeling.services import MLService
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def quick_test():
    """Quick test of logistic regression"""
    print("🧪 Quick Logistic Regression Test")
    print("=" * 40)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    print(f"📊 Dataset: {df.shape}")
    print(f"🎯 Target: {df['target'].value_counts().to_dict()}")
    
    # Initialize service
    ml_service = MLService()
    
    try:
        # Preprocess
        df_processed, _, _ = ml_service.preprocess_data(df, 'target')
        print("✅ Preprocessing completed")
        
        # Prepare data
        X = df_processed.drop('target', axis=1)
        y = df_processed['target']
        
        # Train model
        model, time = ml_service.train_model('logistic_regression', X, y, 'classification')
        print(f"✅ Model trained in {time:.2f}s")
        
        # Evaluate
        metrics = ml_service.evaluate_model(model, X, y, 'classification')
        print(f"✅ Evaluation: accuracy={metrics['accuracy']:.3f}")
        
        print("🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if not success:
        sys.exit(1) 