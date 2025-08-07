#!/usr/bin/env python3
"""
Test script for the intelligent model selection using LangGraph and Gemini API
"""

import os
import sys
import django
import pandas as pd
import numpy as np

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
django.setup()

from modeling.llm_router import ModelSelectionService

def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    # Create sample classification dataset
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple classification rule
    
    # Add some noise and missing values
    X[0:50, 0] = np.nan  # Add missing values
    X[100:150, 1] = np.random.randn(50) * 5  # Add outliers
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def test_intelligent_selection():
    """Test the intelligent model selection"""
    print("🤖 Testing Intelligent Model Selection")
    print("=" * 50)
    
    # Create sample dataset
    df = create_sample_dataset()
    print(f"📊 Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"🎯 Target column: {df['target'].value_counts().to_dict()}")
    
    # Initialize the service
    service = ModelSelectionService()
    
    try:
        # Get intelligent suggestions
        print("\n🔍 Getting intelligent model suggestions...")
        result = service.get_intelligent_suggestions(df, 'target')
        
        print("\n📋 Results:")
        print(f"Analysis: {result.get('analysis', 'N/A')}")
        
        if 'recommended_models' in result:
            print(f"\n🎯 Recommended Models:")
            for i, model in enumerate(result['recommended_models'], 1):
                print(f"  {i}. {model}")
        
        if 'final_recommendation' in result:
            final = result['final_recommendation']
            print(f"\n🏆 Final Recommendation:")
            print(f"  Best Model: {final.get('best_model', 'N/A')}")
            print(f"  Confidence: {final.get('confidence', 'N/A')}")
            print(f"  Reasoning: {final.get('reasoning', 'N/A')}")
        
        # Test hyperparameter optimization
        print("\n⚙️ Testing hyperparameter optimization...")
        hyperparams = service.get_model_hyperparameters('random_forest', 'classification', len(df))
        print(f"Optimized hyperparameters: {hyperparams}")
        
        # Test model explanation
        print("\n💡 Testing model explanation...")
        dataset_info = service._analyze_dataset(df, 'target')
        explanation = service.explain_model_choice('random_forest', dataset_info)
        print(f"Explanation: {explanation}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to missing API key or network issues.")

def main():
    """Main test function"""
    print("🚀 Starting Intelligent Model Selection Test")
    print("Make sure you have set up your GOOGLE_API_KEY in the environment")
    print()
    
    test_intelligent_selection()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main() 