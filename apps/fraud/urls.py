from django.urls import path
from . import views

urlpatterns = [
    # Enterprise Fintech Core Views
    path('transactions/', views.transactions_list_view, name='transactions_list'),
    path('transactions/<str:tx_id>/', views.transaction_detail_view, name='transaction_detail'),
    path('alerts/', views.alerts_list_view, name='alerts_list'),
    path('cases/', views.cases_list_view, name='cases_list'),
    path('cases/<str:case_id>/', views.case_detail_view, name='case_detail'),
    path('analytics/', views.analytics_hub_view, name='analytics_hub'),
    path('models/', views.models_hub_view, name='models_hub'),
    path('datasets/', views.datasets_hub_view, name='datasets_hub'),
    path('reports/', views.reports_hub_view, name='reports_hub'),
    path('live/', views.live_monitoring_view, name='live_monitoring'),

    # Backward Compatibility Endpoints
    path('upload', views.upload_data_view, name='upload'),
    path('dnn', views.dnn_view, name='dnn'),
    path('rfc', views.rfc_view, name='rfc'),
    path('prediction', views.prediction_view, name='prediction'),
]
