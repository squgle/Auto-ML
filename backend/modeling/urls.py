from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_model, name='train_model'),
    path('models/', views.get_models, name='get_models'),
    path('models/<int:model_id>/', views.get_model_details, name='get_model_details'),
    path('models/<int:model_id>/metrics/', views.get_model_metrics, name='get_model_metrics'),
    path('compare/', views.compare_models, name='compare_models'),
    path('suggestions/<int:dataset_id>/', views.get_model_suggestions, name='get_model_suggestions'),
    path('hyperparameters/', views.get_optimized_hyperparameters, name='get_optimized_hyperparameters'),
    path('explain/', views.explain_model_choice, name='explain_model_choice'),
]
