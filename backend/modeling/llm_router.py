import os
import json
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai
from decouple import config

# Configure Gemini API
GOOGLE_API_KEY = config('GOOGLE_API_KEY', default='your-google-api-key')
genai.configure(api_key=GOOGLE_API_KEY)

class ModelSelectionState:
    """State for the model selection workflow"""
    def __init__(self, dataset_info: Dict, problem_type: str, dataset_size: int, feature_count: int):
        self.dataset_info = dataset_info
        self.problem_type = problem_type
        self.dataset_size = dataset_size
        self.feature_count = feature_count
        self.analysis = {}
        self.recommended_models = []
        self.reasoning = ""
        self.final_recommendation = None

class IntelligentModelSelector:
    """Intelligent model selection using LangGraph and Gemini API"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Define the workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for model selection"""
        
        # Create the state graph
        workflow = StateGraph(ModelSelectionState)
        
        # Add nodes
        workflow.add_node("analyze_data", self._analyze_data)
        workflow.add_node("suggest_models", self._suggest_models)
        workflow.add_node("evaluate_models", self._evaluate_models)
        workflow.add_node("make_final_recommendation", self._make_final_recommendation)
        
        # Define the flow
        workflow.set_entry_point("analyze_data")
        workflow.add_edge("analyze_data", "suggest_models")
        workflow.add_edge("suggest_models", "evaluate_models")
        workflow.add_edge("evaluate_models", "make_final_recommendation")
        workflow.add_edge("make_final_recommendation", END)
        
        return workflow.compile()
    
    def _analyze_data(self, state: ModelSelectionState) -> ModelSelectionState:
        """Analyze the dataset characteristics"""
        
        prompt = f"""
        Analyze the following dataset characteristics and provide insights:
        
        Dataset Info: {json.dumps(state.dataset_info, indent=2)}
        Problem Type: {state.problem_type}
        Dataset Size: {state.dataset_size} rows
        Feature Count: {state.feature_count} features
        
        Please analyze:
        1. Data quality issues (missing values, outliers, etc.)
        2. Feature characteristics (numerical vs categorical, distributions)
        3. Problem complexity
        4. Computational requirements
        
        Provide a detailed analysis in JSON format.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state.analysis = json.loads(response.content)
        
        return state
    
    def _suggest_models(self, state: ModelSelectionState) -> ModelSelectionState:
        """Suggest appropriate models based on analysis"""
        
        prompt = f"""
        Based on the analysis: {json.dumps(state.analysis, indent=2)}
        
        Problem Type: {state.problem_type}
        Dataset Size: {state.dataset_size}
        Feature Count: {state.feature_count}
        
        Suggest 3-5 most appropriate machine learning models for this dataset.
        Consider:
        1. Problem type (classification/regression)
        2. Dataset size (small/medium/large)
        3. Feature count (low/high dimensional)
        4. Data quality issues
        5. Computational efficiency
        
        Available models for {state.problem_type}:
        - Random Forest
        - XGBoost
        - LightGBM
        - CatBoost
        - Logistic Regression (classification) / Linear Regression (regression)
        - Support Vector Machine
        - K-Nearest Neighbors
        - Decision Tree
        - Gradient Boosting
        - AdaBoost
        - Ridge/Lasso/Elastic Net (regression)
        
        Return a JSON array of recommended models with reasoning for each.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state.recommended_models = json.loads(response.content)
        
        return state
    
    def _evaluate_models(self, state: ModelSelectionState) -> ModelSelectionState:
        """Evaluate the suggested models"""
        
        prompt = f"""
        Evaluate the recommended models: {json.dumps(state.recommended_models, indent=2)}
        
        For each model, provide:
        1. Expected performance score (0-1)
        2. Training time estimate (fast/medium/slow)
        3. Memory requirements (low/medium/high)
        4. Interpretability score (0-1)
        5. Pros and cons
        
        Consider the dataset characteristics:
        - Problem type: {state.problem_type}
        - Dataset size: {state.dataset_size}
        - Feature count: {state.feature_count}
        - Analysis: {json.dumps(state.analysis, indent=2)}
        
        Return a JSON object with detailed evaluation for each model.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state.reasoning = json.loads(response.content)
        
        return state
    
    def _make_final_recommendation(self, state: ModelSelectionState) -> ModelSelectionState:
        """Make the final model recommendation"""
        
        prompt = f"""
        Based on the model evaluation: {json.dumps(state.reasoning, indent=2)}
        
        Make a final recommendation for the best model to use.
        Consider:
        1. Expected performance
        2. Training time
        3. Interpretability
        4. Dataset characteristics
        
        Return a JSON object with:
        - best_model: the recommended model name
        - confidence: confidence score (0-1)
        - reasoning: detailed explanation
        - alternative_models: list of backup options
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state.final_recommendation = json.loads(response.content)
        
        return state
    
    def select_models(self, dataset_info: Dict, problem_type: str, dataset_size: int, feature_count: int) -> Dict:
        """Main method to select models using the LangGraph workflow"""
        
        # Initialize state
        initial_state = ModelSelectionState(dataset_info, problem_type, dataset_size, feature_count)
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "analysis": final_state.analysis,
            "recommended_models": final_state.recommended_models,
            "evaluation": final_state.reasoning,
            "final_recommendation": final_state.final_recommendation
        }

class ModelSelectionService:
    """Service for intelligent model selection"""
    
    def __init__(self):
        self.selector = IntelligentModelSelector()
    
    def get_intelligent_suggestions(self, df, target_column: str) -> Dict:
        """Get intelligent model suggestions using LangGraph and Gemini"""
        
        # Analyze dataset
        dataset_info = self._analyze_dataset(df, target_column)
        problem_type = self._detect_problem_type(df[target_column])
        dataset_size = len(df)
        feature_count = len(df.columns) - 1
        
        # Get intelligent suggestions
        result = self.selector.select_models(
            dataset_info, problem_type, dataset_size, feature_count
        )
        
        return result
    
    def _analyze_dataset(self, df, target_column: str) -> Dict:
        """Analyze dataset characteristics"""
        
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_column": target_column,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "numerical_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "target_distribution": df[target_column].value_counts().to_dict() if df[target_column].dtype == 'object' else {
                "min": float(df[target_column].min()),
                "max": float(df[target_column].max()),
                "mean": float(df[target_column].mean()),
                "std": float(df[target_column].std())
            }
        }
        
        return analysis
    
    def _detect_problem_type(self, target_series) -> str:
        """Detect if the problem is classification or regression"""
        
        if target_series.dtype in ['object', 'category']:
            return 'classification'
        elif len(target_series.unique()) <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def get_model_hyperparameters(self, model_name: str, problem_type: str, dataset_size: int) -> Dict:
        """Get optimized hyperparameters for a specific model"""
        
        prompt = f"""
        Provide optimized hyperparameters for {model_name} model.
        
        Context:
        - Problem type: {problem_type}
        - Dataset size: {dataset_size} rows
        
        Return a JSON object with hyperparameters for the {model_name} model.
        Include only the most important parameters that significantly affect performance.
        """
        
        response = self.selector.llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)
    
    def explain_model_choice(self, model_name: str, dataset_info: Dict) -> str:
        """Get explanation for why a specific model was chosen"""
        
        prompt = f"""
        Explain why {model_name} is a good choice for this dataset.
        
        Dataset characteristics:
        {json.dumps(dataset_info, indent=2)}
        
        Provide a clear, concise explanation suitable for a data scientist.
        """
        
        response = self.selector.llm.invoke([HumanMessage(content=prompt)])
        return response.content 