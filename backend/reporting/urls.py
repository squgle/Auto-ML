from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_report, name='generate_report'),
    path('reports/', views.get_reports, name='get_reports'),
    path('reports/<int:report_id>/', views.get_report_details, name='get_report_details'),
    path('reports/<int:report_id>/download_pdf/', views.download_report_pdf, name='download_report_pdf'),
    path('comparison/', views.generate_model_comparison_report, name='generate_model_comparison_report'),
    path('charts/', views.get_available_charts, name='get_available_charts'),
]
