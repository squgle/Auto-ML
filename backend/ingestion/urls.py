from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('datasets/', views.get_datasets, name='get_datasets'),
    path('datasets/<int:dataset_id>/', views.get_dataset_details, name='get_dataset_details'),
    path('datasets/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
]
