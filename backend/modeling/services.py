import pandas as pd
import numpy as np
import joblib
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier

# Optional imports with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not available")
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Warning: Optuna not available")
    OPTUNA_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from django.utils import timezone
from .llm_router import ModelSelectionService

class MLService:
    """Comprehensive Machine Learning Service"""
    
    def __init__(self):
        self.models = {
            'classification': {
                'random_forest': RandomForestClassifier,
                'logistic_regression': LogisticRegression,
                'svm': SVC,
                'knn': KNeighborsClassifier,
                'decision_tree': DecisionTreeClassifier,
                'gradient_boosting': GradientBoostingClassifier,
                'ada_boost': AdaBoostClassifier,
            },
            'regression': {
                'random_forest': RandomForestRegressor,
                'linear_regression': LinearRegression,
                'ridge': Ridge,
                'lasso': Lasso,
                'elastic_net': ElasticNet,
                'svm': SVR,
                'knn': KNeighborsRegressor,
                'decision_tree': DecisionTreeRegressor,
                'gradient_boosting': GradientBoostingRegressor,
            }
        }
        
        # Add optional models if available
        if XGBOOST_AVAILABLE:
            self.models['classification']['xgboost'] = xgb.XGBClassifier
            self.models['regression']['xgboost'] = xgb.XGBRegressor
            
        if LIGHTGBM_AVAILABLE:
            self.models['classification']['lightgbm'] = lgb.LGBMClassifier
            self.models['regression']['lightgbm'] = lgb.LGBMRegressor
            
        if CATBOOST_AVAILABLE:
            self.models['classification']['catboost'] = cb.CatBoostClassifier
            self.models['regression']['catboost'] = cb.CatBoostRegressor
        
        # Initialize intelligent model selector
        self.model_selector = ModelSelectionService()
        
        self.default_params = {
            'random_forest': {'n_estimators': 100, 'random_state': 42},
            'logistic_regression': {'random_state': 42, 'max_iter': 1000, 'solver': 'lbfgs'},
            'svm': {'random_state': 42},
            'knn': {'n_neighbors': 5},
            'decision_tree': {'random_state': 42},
            'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
            'ada_boost': {'n_estimators': 100, 'random_state': 42},
            'linear_regression': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5},
        }
        
        # Add optional model parameters if available
        if XGBOOST_AVAILABLE:
            self.default_params['xgboost'] = {'n_estimators': 100, 'random_state': 42}
            
        if LIGHTGBM_AVAILABLE:
            self.default_params['lightgbm'] = {'n_estimators': 100, 'random_state': 42}
            
        if CATBOOST_AVAILABLE:
            self.default_params['catboost'] = {'iterations': 100, 'random_state': 42}
    
    def preprocess_data(self, df, target_column, categorical_columns=None, numerical_columns=None, missing_threshold=0.3):
        """Comprehensive data preprocessing"""
        df_processed = df.copy()
        
        # 1. Drop columns with too many missing values
        missing_fraction = df_processed.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > missing_threshold].index.tolist()
        if target_column in cols_to_drop:
            cols_to_drop.remove(target_column)  # Never drop the target column!
        df_processed.drop(columns=cols_to_drop, inplace=True)
        
        # 1. Drop columns with too many missing values
        missing_fraction = df_processed.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > missing_threshold].index.tolist()
        if target_column in cols_to_drop:
            cols_to_drop.remove(target_column)  # Never drop the target column!
        df_processed.drop(columns=cols_to_drop, inplace=True)
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].dtype in ['object', 'category']:
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col] = df_processed[col].fillna(mode_val)
            else:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
        
        # Remove outliers using IQR method
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        outliers_removed = 0
        for col in numerical_cols:
            # Calculate IQR
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify and remove outliers
            outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
            outliers_count = outlier_mask.sum()
            
            if outliers_count > 0:
                df_processed = df_processed[~outlier_mask]
                outliers_removed += outliers_count
        
        if outliers_removed > 0:
            print(f"🚨 Removed {outliers_removed} outliers using IQR method")
        
        # Encode categorical variables using intelligent selection
        if categorical_columns is None:
            categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        encoders = {}
        encoded_columns = []
        
        for col in categorical_columns:
            if col in df_processed.columns and col != target_column:
                # Choose encoding method based on data characteristics
                encoding_method, encoder = self._choose_encoding_method(df_processed, col, target_column)
                
                if encoding_method == 'onehot':
                    # OneHot encoding - creates new columns
                    encoded_cols = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                    df_processed = pd.concat([df_processed, encoded_cols], axis=1)
                    df_processed.drop(columns=[col], inplace=True)
                    encoded_columns.extend(encoded_cols.columns.tolist())
                    encoders[col] = {'type': 'onehot', 'encoder': encoder, 'columns': encoded_cols.columns.tolist()}
                    
                elif encoding_method == 'label':
                    # Label encoding - replaces original column
                    df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
                    encoders[col] = {'type': 'label', 'encoder': encoder}
                    
                elif encoding_method == 'ordinal':
                    # Ordinal encoding - for ordered categories
                    df_processed[col] = encoder.fit_transform(df_processed[col].values.reshape(-1, 1)).flatten()
                    encoders[col] = {'type': 'ordinal', 'encoder': encoder}
        
        # Scale numerical features
        if numerical_columns is None:
            numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_columns:
                numerical_columns.remove(target_column)
        
        scaler = StandardScaler()
        if numerical_columns:
            df_processed[numerical_columns] = df_processed[numerical_columns].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
            df_processed[numerical_columns] = df_processed[numerical_columns].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
            df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
            # Mark that scaling has been applied
            self._scaler_applied = True


        
        return df_processed, encoders, scaler
    
    def _choose_encoding_method(self, df, column, target_column):
        """
        Intelligently choose the best encoding method for a categorical column
        
        Decision Logic:
        1. OneHot: Low cardinality (≤10), no clear order, good for tree-based models
        2. Label: High cardinality (>10), no clear order, memory efficient  
        3. Ordinal: Has natural ordering (e.g., Low/Medium/High)
        """
        unique_values = df[column].nunique()
        unique_vals = df[column].unique()
        
        # Check for ordinal patterns (ordered categories)
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'], 
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['bad', 'average', 'good'],
            ['beginner', 'intermediate', 'advanced'],
            ['very poor', 'poor', 'fair', 'good', 'very good', 'excellent']
        ]
        
        # Convert to lowercase for pattern matching
        vals_lower = [str(v).lower().strip() for v in unique_vals if pd.notna(v)]
        
        # Check if column has ordinal pattern
        for pattern in ordinal_patterns:
            if all(val in pattern for val in vals_lower if val != 'unknown'):
                print(f"🔢 Using ORDINAL encoding for '{column}' (detected pattern: {pattern})")
                encoder = OrdinalEncoder(categories=[pattern])
                return 'ordinal', encoder
        
        # Decision based on cardinality
        if unique_values <= 10:
            # Low cardinality -> OneHot encoding
            print(f"🎯 Using ONE-HOT encoding for '{column}' ({unique_values} unique values)")
            return 'onehot', None  # pd.get_dummies doesn't need pre-fitted encoder
            
        else:
            # High cardinality -> Label encoding
            print(f"🏷️ Using LABEL encoding for '{column}' ({unique_values} unique values - high cardinality)")
            encoder = LabelEncoder()
            return 'label', encoder

    def detect_problem_type(self, target_column, df):
        """Detect if the problem is classification or regression"""
        target_data = df[target_column]
        
        if target_data.dtype in ['object', 'category']:
            return 'classification'
        elif len(target_data.unique()) <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def suggest_preprocessing_steps(self, df, target_column):
        """Suggest preprocessing steps based on data analysis"""
        suggestions = []
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            suggestions.append({
                'step': 'Handle Missing Values',
                'description': f'Found {missing_data.sum()} missing values across {len(missing_data[missing_data > 0])} columns',
                'priority': 'high'
            })
        
        # Check for categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            suggestions.append({
                'step': 'Encode Categorical Variables',
                'description': f'Found {len(categorical_cols)} categorical columns that need encoding',
                'priority': 'high'
            })
        
        # Check for numerical scaling
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            suggestions.append({
                'step': 'Scale Numerical Features',
                'description': f'Found {len(numerical_cols)} numerical columns that may benefit from scaling',
                'priority': 'medium'
            })
        
        # Check for outliers
        outlier_cols = []
        for col in numerical_cols:
            if col != target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    outlier_cols.append(col)
        
        if outlier_cols:
            suggestions.append({
                'step': 'Handle Outliers',
                'description': f'Found outliers in {len(outlier_cols)} columns',
                'priority': 'medium'
            })
        
        return suggestions
    
    def suggest_models(self, problem_type, dataset_size, feature_count):
        """Suggest appropriate models based on problem type and dataset characteristics"""
        suggestions = {
            'classification': {
                'small_dataset': ['logistic_regression', 'decision_tree', 'knn'],
                'medium_dataset': ['random_forest', 'xgboost', 'lightgbm', 'svm'],
                'large_dataset': ['xgboost', 'lightgbm', 'catboost', 'gradient_boosting'],
                'high_dimensions': ['logistic_regression', 'svm', 'random_forest'],
                'low_dimensions': ['knn', 'decision_tree', 'ada_boost']
            },
            'regression': {
                'small_dataset': ['linear_regression', 'ridge', 'lasso'],
                'medium_dataset': ['random_forest', 'xgboost', 'lightgbm'],
                'large_dataset': ['xgboost', 'lightgbm', 'catboost'],
                'high_dimensions': ['ridge', 'lasso', 'elastic_net'],
                'low_dimensions': ['linear_regression', 'svm', 'knn']
            }
        }
        
        # Determine dataset size category
        if dataset_size < 1000:
            size_category = 'small_dataset'
        elif dataset_size < 10000:
            size_category = 'medium_dataset'
        else:
            size_category = 'large_dataset'
        
        # Determine dimension category
        if feature_count > 50:
            dim_category = 'high_dimensions'
        else:
            dim_category = 'low_dimensions'
        
        recommended_models = suggestions[problem_type][size_category] + suggestions[problem_type][dim_category]
        return list(set(recommended_models))  # Remove duplicates
    
    def get_intelligent_model_suggestions(self, df, target_column):
        """Get intelligent model suggestions using LangGraph and Gemini API"""
        try:
            return self.model_selector.get_intelligent_suggestions(df, target_column)
        except Exception as e:
            print(f"Error in intelligent model selection: {e}")
            # Fallback to traditional method
            problem_type = self.detect_problem_type(target_column, df)
            return {
                'recommended_models': self.suggest_models(problem_type, len(df), len(df.columns) - 1),
                'analysis': {'error': 'Intelligent selection failed, using fallback'},
                'evaluation': {},
                'final_recommendation': {'best_model': 'random_forest', 'confidence': 0.5}
            }
    
    def get_optimized_hyperparameters(self, model_name, problem_type, dataset_size):
        """Get optimized hyperparameters using LLM"""
        try:
            return self.model_selector.get_model_hyperparameters(model_name, problem_type, dataset_size)
        except Exception as e:
            print(f"Error getting optimized hyperparameters: {e}")
            return self.default_params.get(model_name, {})
    
    def explain_model_choice(self, model_name, dataset_info):
        """Get explanation for model choice using LLM"""
        try:
            return self.model_selector.explain_model_choice(model_name, dataset_info)
        except Exception as e:
            print(f"Error explaining model choice: {e}")
            return f"{model_name} was selected based on traditional ML heuristics."
    
    def train_model(self, model_name, X_train, y_train, problem_type, hyperparameters=None):
        """Train a specific model"""
        if hyperparameters is None:
            hyperparameters = self.default_params.get(model_name, {})
        
        # Check if model type exists
        if problem_type not in self.models:
            raise ValueError(f"Problem type '{problem_type}' not supported")
        
        if model_name not in self.models[problem_type]:
            raise ValueError(f"Model type '{model_name}' not supported for {problem_type}")
        
        # Special handling for logistic regression
        if model_name == 'logistic_regression':
            # Check for perfect separability
            if len(np.unique(y_train)) < 2:
                raise ValueError("Logistic regression requires at least 2 classes in the target variable")
            
            # Check for data quality issues
            if X_train.isnull().any().any():
                raise ValueError("Logistic regression cannot handle missing values in features")
            
            # Ensure data is properly scaled for logistic regression
            if not hasattr(self, '_scaler_applied'):
                print("Warning: Data should be scaled for logistic regression")
        
        model_class = self.models[problem_type][model_name]
        model = model_class(**hyperparameters)
        
        start_time = timezone.now()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            if model_name == 'logistic_regression':
                # Try with different solver if lbfgs fails
                if 'lbfgs' in str(e).lower():
                    print("Warning: lbfgs solver failed, trying with liblinear solver")
                    hyperparameters['solver'] = 'liblinear'
                    model = model_class(**hyperparameters)
                    model.fit(X_train, y_train)
                else:
                    raise e
            else:
                raise e
        
        training_time = (timezone.now() - start_time).total_seconds()
        
        return model, training_time
    
    def evaluate_model(self, model, X_test, y_test, problem_type):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        metrics = {}
        
        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
        else:  # regression
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2_score'] = r2_score(y_test, y_pred)
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=cv)
        return {
            'scores': cv_scores.tolist(),
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
    
    def optimize_hyperparameters(self, model_name, X_train, y_train, problem_type, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE:
            print("Warning: Optuna not available, skipping hyperparameter optimization")
            return self.default_params.get(model_name, {})
            
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'random_state': 42
                }
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': 42
                }
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': 42
                }
            else:
                return 0.0
            
            model_class = self.models[problem_type][model_name]
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {}
        
        feature_importance = dict(zip(feature_names, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, model, filepath):
        """Save trained model"""
        from django.conf import settings
        import os
        
        # Create full path within media directory
        full_path = os.path.join(settings.MEDIA_ROOT, filepath)
        print(f"DEBUG: Saving model to {full_path}")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(model, full_path)
        print(f"DEBUG: Model saved successfully")
    
    def load_model(self, filepath):
        """Load trained model"""
        from django.conf import settings
        import os
        
        # Create full path within media directory
        full_path = os.path.join(settings.MEDIA_ROOT, filepath)
        return joblib.load(full_path)
    
    def generate_charts(self, df, target_column, model=None, y_test=None, y_pred=None):
        """Generate various charts for analysis"""
        charts = {}
        
        # Data distribution
        if target_column in df.columns:
            if df[target_column].dtype in ['object', 'category']:
                fig = px.histogram(df, x=target_column, title=f'Distribution of {target_column}')
            else:
                fig = px.histogram(df, x=target_column, title=f'Distribution of {target_column}')
            charts['target_distribution'] = fig.to_dict()
        
        # Correlation matrix for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            fig = px.imshow(corr_matrix, title='Correlation Matrix')
            charts['correlation_matrix'] = fig.to_dict()
        
        # Feature importance if model is provided
        if model and hasattr(model, 'feature_importances_'):
            feature_names = [col for col in df.columns if col != target_column]
            importance = self.get_feature_importance(model, feature_names)
            if importance:
                fig = px.bar(x=list(importance.keys())[:10], y=list(importance.values())[:10],
                           title='Top 10 Feature Importance')
                charts['feature_importance'] = fig.to_dict()
        
        # Confusion matrix for classification
        if model and y_test is not None and y_pred is not None:
            if hasattr(model, 'predict_proba'):
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, title='Confusion Matrix')
                charts['confusion_matrix'] = fig.to_dict()
        
        return charts 