#!/usr/bin/env python3
"""
Test script specifically for logistic regression model training
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
django.setup()

from modeling.services import MLService
from modeling.models import MLModel, ModelEvaluation
from ingestion.models import Dataset

def test_logistic_regression():
    """Test logistic regression model training"""
    print("🧪 Testing Logistic Regression Model Training")
    print("=" * 50)
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    print(f"📊 Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"🎯 Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Initialize ML service
    ml_service = MLService()
    
    try:
        # Test preprocessing
        print("\n🔧 Testing data preprocessing...")
        df_processed, encoders, scaler = ml_service.preprocess_data(df, 'target')
        print(f"✅ Preprocessing completed. Processed shape: {df_processed.shape}")
        
        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col != 'target']
        X = df_processed[feature_columns]
        y = df_processed['target']
        
        print(f"📈 Features: {len(feature_columns)}")
        print(f"🎯 Target unique values: {len(y.unique())}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Data split: Train {X_train.shape}, Test {X_test.shape}")
        
        # Detect problem type
        problem_type = ml_service.detect_problem_type('target', df)
        print(f"🎯 Problem type detected: {problem_type}")
        
        # Test model training
        print("\n🤖 Testing logistic regression training...")
        model_name = 'logistic_regression'
        hyperparameters = {'random_state': 42, 'max_iter': 1000}
        
        trained_model, training_time = ml_service.train_model(
            model_name, X_train, y_train, problem_type, hyperparameters
        )
        print(f"✅ Model trained successfully in {training_time:.2f} seconds")
        
        # Test model evaluation
        print("\n📊 Testing model evaluation...")
        metrics = ml_service.evaluate_model(trained_model, X_test, y_test, problem_type)
        print(f"✅ Evaluation completed:")
        for metric, value in metrics.items():
            if isinstance(value, list):
                print(f"  {metric}: {len(value)} values")
            else:
                print(f"  {metric}: {value:.4f}")
        
        # Test cross-validation
        print("\n🔄 Testing cross-validation...")
        cv_results = ml_service.cross_validate_model(trained_model, X, y)
        print(f"✅ Cross-validation completed:")
        print(f"  Mean CV score: {cv_results['mean']:.4f}")
        print(f"  CV std: {cv_results['std']:.4f}")
        
        # Test feature importance
        print("\n📈 Testing feature importance...")
        feature_importance = ml_service.get_feature_importance(trained_model, feature_columns)
        print(f"✅ Feature importance extracted: {len(feature_importance)} features")
        
        # Test model saving
        print("\n💾 Testing model saving...")
        model_path = "test_models/test_logistic_regression.joblib"
        ml_service.save_model(trained_model, model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Test model loading
        print("\n📂 Testing model loading...")
        loaded_model = ml_service.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Test predictions
        print("\n🔮 Testing predictions...")
        predictions = loaded_model.predict(X_test[:5])
        print(f"✅ Predictions: {predictions}")
        
        print("\n✅ All tests passed! Logistic regression is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_dataset():
    """Test with a real dataset from the database"""
    print("\n🧪 Testing with real dataset from database")
    print("=" * 50)
    
    try:
        # Get first available dataset
        datasets = Dataset.objects.all()
        if not datasets.exists():
            print("❌ No datasets found in database")
            return False
        
        dataset = datasets.first()
        print(f"📊 Using dataset: {dataset.name}")
        
        # Read dataset
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx') or dataset.file.name.endswith('.xls'):
            df = pd.read_excel(dataset.file.path)
        else:
            print(f"❌ Unsupported file format: {dataset.file.name}")
            return False
        
        print(f"📈 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Find a suitable target column
        target_column = None
        for col in df.columns:
            if df[col].dtype in ['object', 'category'] or len(df[col].unique()) <= 10:
                target_column = col
                break
        
        if not target_column:
            print("❌ No suitable target column found")
            return False
        
        print(f"🎯 Using target column: {target_column}")
        print(f"🎯 Target distribution: {df[target_column].value_counts().to_dict()}")
        
        # Initialize ML service
        ml_service = MLService()
        
        # Test preprocessing
        print("\n🔧 Testing data preprocessing...")
        df_processed, encoders, scaler = ml_service.preprocess_data(df, target_column)
        print(f"✅ Preprocessing completed. Processed shape: {df_processed.shape}")
        
        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col != target_column]
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        print(f"📈 Features: {len(feature_columns)}")
        print(f"🎯 Target unique values: {len(y.unique())}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Data split: Train {X_train.shape}, Test {X_test.shape}")
        
        # Detect problem type
        problem_type = ml_service.detect_problem_type(target_column, df)
        print(f"🎯 Problem type detected: {problem_type}")
        
        # Test model training
        print("\n🤖 Testing logistic regression training...")
        model_name = 'logistic_regression'
        hyperparameters = {'random_state': 42, 'max_iter': 1000}
        
        trained_model, training_time = ml_service.train_model(
            model_name, X_train, y_train, problem_type, hyperparameters
        )
        print(f"✅ Model trained successfully in {training_time:.2f} seconds")
        
        print("\n✅ Real dataset test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during real dataset testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Logistic Regression Tests")
    print("Make sure you have set up your environment correctly")
    print()
    
    # Test with synthetic data
    success1 = test_logistic_regression()
    
    # Test with real dataset
    success2 = test_with_real_dataset()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Logistic regression is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1) 