from django.urls import path
from . import api_views

urlpatterns = [
    # Transactions
    path('transactions/', api_views.api_search_transactions, name='api_search_transactions'),
    path('transactions/<str:tx_id>/', api_views.api_transaction_detail, name='api_transaction_detail'),
    path('transactions/ingest', api_views.api_ingest_transaction, name='api_ingest_transaction'),
    path('transactions/ingest/', api_views.api_ingest_transaction, name='api_ingest_transaction_slash'),

    # Alerts
    path('alerts/', api_views.api_list_alerts, name='api_list_alerts'),
    path('alerts/<str:alert_id>/transition/', api_views.api_transition_alert, name='api_transition_alert'),

    # Investigations
    path('cases/', api_views.api_cases_collection, name='api_cases_collection'),
    path('cases/<str:case_id>/', api_views.api_case_detail, name='api_case_detail'),
    path('cases/<str:case_id>/notes/', api_views.api_case_add_note, name='api_case_add_note'),
    path('cases/<str:case_id>/resolve/', api_views.api_case_resolve, name='api_case_resolve'),

    # Analytics
    path('analytics/overview/', api_views.api_analytics_overview, name='api_analytics_overview'),
    path('analytics/risk-distribution/', api_views.api_analytics_risk_distribution, name='api_analytics_risk_distribution'),
    path('analytics/time-series/', api_views.api_analytics_time_series, name='api_analytics_time_series'),
    path('analytics/models/', api_views.api_analytics_models, name='api_analytics_models'),

    # Reports
    path('reports/summary/', api_views.api_reports_summary, name='api_reports_summary'),
    path('reports/export/', api_views.api_reports_export_csv, name='api_reports_export_csv'),

    # Phase 7: Real-Time Live Monitoring & Synthetic Stream
    path('live/status/', api_views.api_live_status, name='api_live_status'),
    path('live/transactions/', api_views.api_live_transactions, name='api_live_transactions'),
    path('live/alerts/', api_views.api_live_alerts, name='api_live_alerts'),
    path('live/metrics/', api_views.api_live_metrics, name='api_live_metrics'),
    path('live/start/', api_views.api_live_start, name='api_live_start'),
    path('live/pause/', api_views.api_live_pause, name='api_live_pause'),
    path('live/resume/', api_views.api_live_resume, name='api_live_resume'),
    path('live/stop/', api_views.api_live_stop, name='api_live_stop'),
    path('live/step/', api_views.api_live_step, name='api_live_step'),
    path('live/stream/', api_views.api_live_stream_delta, name='api_live_stream_delta'),
]

