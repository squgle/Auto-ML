from django.urls import path
from . import views

urlpatterns = [
    path('session/create/', views.create_chat_session, name='create_chat_session'),
    path('session/<str:session_id>/history/', views.get_chat_history, name='get_chat_history'),
    path('message/send/', views.send_message, name='send_message'),
    path('analyze/', views.analyze_dataset, name='analyze_dataset'),
    path('visualize/', views.generate_visualization, name='generate_visualization'),
    path('preprocessing/suggestions/', views.get_preprocessing_suggestions, name='get_preprocessing_suggestions'),
] 