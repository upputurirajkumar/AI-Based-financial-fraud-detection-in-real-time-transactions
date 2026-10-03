import logging
from decimal import Decimal
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError, PermissionDenied

from apps.accounts.decorators import admin_required
from apps.core.models import AuditLog
from apps.transactions.models import Transaction, DatasetRecord
from apps.fraud.models import Prediction, FraudAlert, InvestigationCase, InvestigationNote
from apps.models_registry.models import ModelRecord

from services.dataset_service import DatasetService
from services.prediction_service import PredictionService
from services.model_service import ModelService
from services.transaction_service import TransactionService
from services.alert_service import AlertService
from services.investigation_service import InvestigationService
from services.analytics_service import AnalyticsService
from services.report_service import ReportService
from services.batch_ingestion_service import BatchIngestionService
from services.fraud_engine import FraudEngine

logger = logging.getLogger(__name__)


def require_analyst(user) -> bool:
    if user.is_staff or user.is_superuser:
        return True
    profile = getattr(user, 'profile', None)
    return bool(profile and profile.is_analyst_or_admin())


# ==========================================================================
# 1. OVERVIEW / EXECUTIVE DASHBOARD
# ==========================================================================

@login_required
def overview_dashboard_view(request):
    """Primary executive command center consuming real database analytics."""
    overview = AnalyticsService.get_overview_metrics()
    risk_dist = AnalyticsService.get_risk_distribution()

    recent_alerts = FraudAlert.objects.select_related('transaction', 'prediction').order_by('-created_at')[:6]
    recent_transactions = Transaction.objects.prefetch_related('predictions', 'alerts').order_by('-created_at')[:6]
    recent_cases = InvestigationCase.objects.select_related('assigned_analyst').order_by('-created_at')[:4]

    context = {
        'overview': overview,
        'risk_dist': risk_dist,
        'recent_alerts': recent_alerts,
        'recent_transactions': recent_transactions,
        'recent_cases': recent_cases,
    }
    return render(request, 'Home.html', context)


# ==========================================================================
# 2. TRANSACTIONS EXPLORER & DETAIL
# ==========================================================================

@login_required
def transactions_list_view(request):
    """Interactive transaction intelligence grid with search, filter, and test modal."""
    page = int(request.GET.get('page', 1))
    search_query = request.GET.get('q')
    tx_type = request.GET.get('type')
    risk_tier = request.GET.get('risk_tier')
    channel = request.GET.get('channel')
    min_amount = float(request.GET['min_amount']) if request.GET.get('min_amount') else None
    max_amount = float(request.GET['max_amount']) if request.GET.get('max_amount') else None

    # Handle single manual transaction test from modal
    if request.method == 'POST' and request.POST.get('action') == 'test_single':
        try:
            tx_data = {
                'step': int(request.POST.get('step', 1)),
                'type': request.POST.get('type', 'TRANSFER'),
                'amount': float(request.POST.get('amount', 0)),
                'oldbalanceOrg': float(request.POST.get('oldbalanceOrg', 0)),
                'newbalanceOrig': float(request.POST.get('newbalanceOrig', 0)),
                'oldbalanceDest': float(request.POST.get('oldbalanceDest', 0)),
                'newbalanceDest': float(request.POST.get('newbalanceDest', 0)),
                'nameOrig': request.POST.get('nameOrig', 'C_MANUAL_TEST'),
                'nameDest': request.POST.get('nameDest', 'M_MANUAL_TEST'),
            }
            res = TransactionService.ingest_and_evaluate_transaction(tx_data, actor=request.user)
            if res['success']:
                messages.success(
                    request,
                    f"Transaction {res['transaction_id']} evaluated: "
                    f"{res['analysis']['decision_state']} (Risk: {res['analysis']['risk_score']}/100)"
                )
                return redirect('transaction_detail', tx_id=res['transaction_id'])
            else:
                messages.error(request, f"Validation error: {', '.join(res.get('errors', []))}")
        except Exception as exc:
            messages.error(request, f"Test error: {str(exc)}")

    res = TransactionService.search_transactions(
        search_query=search_query,
        transaction_type=tx_type,
        min_amount=min_amount,
        max_amount=max_amount,
        risk_tier=risk_tier,
        channel=channel,
        page=page,
        page_size=20
    )

    context = {
        'transactions': res['transactions'],
        'total_count': res['total_count'],
        'total_pages': res['total_pages'],
        'current_page': res['current_page'],
        'filters': {
            'q': search_query or '',
            'type': tx_type or '',
            'risk_tier': risk_tier or '',
            'channel': channel or '',
            'min_amount': min_amount or '',
            'max_amount': max_amount or ''
        }
    }
    return render(request, 'transactions/list.html', context)


@login_required
def transaction_detail_view(request, tx_id):
    """Detailed transaction investigation view with XAI factors and alert context."""
    detail = TransactionService.get_transaction_detail(tx_id)
    if not detail:
        messages.error(request, f"Transaction '{tx_id}' not found.")
        return redirect('transactions_list')

    return render(request, 'transactions/detail.html', {'detail': detail})


# ==========================================================================
# 3. FRAUD ALERTS REVIEW QUEUE
# ==========================================================================

@login_required
def alerts_list_view(request):
    """Operational alert triage queue with state transition handling."""
    if not require_analyst(request.user):
        messages.error(request, "Access restricted to Compliance Analysts and Administrators.")
        return redirect('home')

    # Handle quick status transition via POST
    if request.method == 'POST' and request.POST.get('action') == 'transition':
        alert_id = request.POST.get('alert_id')
        new_status = request.POST.get('status')
        notes = request.POST.get('notes', '')
        alert = FraudAlert.objects.filter(alert_id=alert_id).first()
        if alert:
            try:
                alert.transition_to(new_status, request.user, notes=notes)
                messages.success(request, f"Alert {alert_id} moved to {new_status}.")
            except ValidationError as ve:
                messages.error(request, str(ve))
        return redirect('alerts_list')

    page = int(request.GET.get('page', 1))
    status_filter = request.GET.get('status')
    severity_filter = request.GET.get('severity')

    res = AlertService.get_review_queue(
        status=status_filter,
        severity=severity_filter,
        page=page,
        page_size=20
    )

    context = {
        'alerts': res['alerts'],
        'total_count': res['total_count'],
        'total_pages': res['total_pages'],
        'current_page': res['current_page'],
        'status_filter': status_filter or '',
        'severity_filter': severity_filter or ''
    }
    return render(request, 'alerts/list.html', context)


# ==========================================================================
# 4. INVESTIGATION CASE WORKSPACE
# ==========================================================================

@login_required
def cases_list_view(request):
    """Case management queue for formal fraud investigations."""
    if not require_analyst(request.user):
        messages.error(request, "Access restricted to Compliance Analysts and Administrators.")
        return redirect('home')

    # Handle case creation
    if request.method == 'POST' and request.POST.get('action') == 'create_case':
        title = request.POST.get('title')
        description = request.POST.get('description', '')
        priority = request.POST.get('priority', 'MEDIUM')
        tx_id = request.POST.get('transaction_id')
        alert_id = request.POST.get('alert_id')

        tx_ids = [tx_id] if tx_id else []
        alert_ids = [alert_id] if alert_id else []

        case = InvestigationService.create_case(
            title=title,
            opened_by=request.user,
            description=description,
            priority=priority,
            assigned_analyst=request.user,
            transaction_ids=tx_ids,
            alert_ids=alert_ids
        )
        messages.success(request, f"Investigation case {case.case_id} initiated.")
        return redirect('case_detail', case_id=case.case_id)

    page = int(request.GET.get('page', 1))
    status_filter = request.GET.get('status')
    priority_filter = request.GET.get('priority')

    res = InvestigationService.list_cases(
        status=status_filter,
        priority=priority_filter,
        page=page,
        page_size=20
    )

    context = {
        'cases': res['cases'],
        'total_count': res['total_count'],
        'total_pages': res['total_pages'],
        'current_page': res['current_page'],
        'status_filter': status_filter or '',
        'priority_filter': priority_filter or ''
    }
    return render(request, 'cases/list.html', context)


@login_required
def case_detail_view(request, case_id):
    """Analyst workspace with linked transactions, timeline notes, and ground-truth resolution."""
    if not require_analyst(request.user):
        messages.error(request, "Access restricted to Compliance Analysts and Administrators.")
        return redirect('home')

    case = InvestigationCase.objects.filter(case_id=case_id).first()
    if not case:
        messages.error(request, f"Case '{case_id}' not found.")
        return redirect('cases_list')

    # Handle note addition
    if request.method == 'POST' and request.POST.get('action') == 'add_note':
        note_text = request.POST.get('note')
        if note_text:
            InvestigationService.add_note(case_id, request.user, note_text)
            messages.success(request, "Investigation note recorded.")
        return redirect('case_detail', case_id=case_id)

    # Handle formal ground-truth resolution
    if request.method == 'POST' and request.POST.get('action') == 'resolve':
        resolution = request.POST.get('resolution')
        notes = request.POST.get('notes', '')
        if resolution:
            try:
                InvestigationService.resolve_case(case_id, resolution, notes, request.user)
                messages.success(request, f"Case formally resolved as {resolution}.")
            except ValidationError as ve:
                messages.error(request, str(ve))
        return redirect('case_detail', case_id=case_id)

    detail = InvestigationService.get_case_detail(case_id)
    return render(request, 'cases/detail.html', {'case': detail})


# ==========================================================================
# 5. ENTERPRISE ANALYTICS HUB
# ==========================================================================

@login_required
def analytics_hub_view(request):
    """Advanced analytics dashboard displaying volume trends and risk ratios."""
    overview = AnalyticsService.get_overview_metrics()
    risk_dist = AnalyticsService.get_risk_distribution()
    channel_type = AnalyticsService.get_channel_and_type_distribution()
    model_stats = AnalyticsService.get_model_version_analytics()

    context = {
        'overview': overview,
        'risk_dist': risk_dist,
        'channel_type': channel_type,
        'model_stats': model_stats,
    }
    return render(request, 'analytics/hub.html', context)


# ==========================================================================
# 6. MODEL INTELLIGENCE REGISTRY
# ==========================================================================

@login_required
def models_hub_view(request):
    """Model governance dashboard displaying active, experimental, and archived pipelines."""
    models_catalog = ModelService.get_available_models()
    records = ModelRecord.objects.all()

    global_importances = []
    try:
        from services.explainability_service import ExplainabilityService
        global_importances = ExplainabilityService.get_global_feature_importances()
    except Exception as e:
        logger.warning(f"Could not load global importances: {e}")

    context = {
        'catalog': models_catalog,
        'records': records,
        'global_importances': global_importances[:8]
    }
    return render(request, 'models/hub.html', context)


# ==========================================================================
# 7. DATASETS & DATA QUALITY
# ==========================================================================

@login_required
def datasets_hub_view(request):
    """Dataset management, CSV upload, and data quality inspection."""
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            df = DatasetService.read_csv_safely(uploaded_file)
            dataset_rec = DatasetRecord.objects.create(
                uploaded_by=request.user,
                filename=uploaded_file.name,
                file_size_bytes=uploaded_file.size,
                total_records=len(df),
            )
            # Ingest through batch ingestion
            summary = BatchIngestionService.process_dataset(
                df=df,
                dataset_record=dataset_rec,
                actor=request.user,
                model_key='rfc_production'
            )
            messages.success(
                request,
                f"Dataset ingested: {summary['accepted_rows']} transactions saved, "
                f"{summary['fraud_predictions']} fraud flags, {summary['alerts_generated']} alerts generated."
            )
            return redirect('datasets_hub')
        except Exception as exc:
            logger.error(f"Dataset upload failed: {exc}")
            messages.error(request, f"Upload error: {str(exc)}")

    datasets = DatasetRecord.objects.all().order_by('-created_at')
    return render(request, 'datasets/hub.html', {'datasets': datasets})


# ==========================================================================
# 8. REPORTS & EXPORT
# ==========================================================================

@login_required
def reports_hub_view(request):
    """Report generation hub producing executive summaries and authorized CSV exports."""
    report = ReportService.generate_fraud_summary_report(actor=request.user)
    return render(request, 'reports/hub.html', {'report': report})


# ==========================================================================
# 9. REAL-TIME SYNTHETIC LIVE MONITORING (PHASE 7)
# ==========================================================================

@login_required
def live_monitoring_view(request):
    """
    Real-Time Synthetic Transaction Monitoring & Live Fraud Intelligence Command Center.
    Explicitly labeled as synthetic demonstration streaming.
    """
    from services.live_stream_manager import LiveStreamManager
    from services.synthetic_generator import SyntheticTransactionGenerator
    manager = LiveStreamManager()
    status = manager.get_status()
    scenarios = SyntheticTransactionGenerator.SUPPORTED_SCENARIOS

    # Get recent synthetic transactions from database to initialize the feed
    synthetic_txs = Transaction.objects.filter(is_synthetic=True).prefetch_related('predictions', 'alerts').order_by('-created_at')[:25]

    context = {
        'status': status,
        'scenarios': scenarios,
        'recent_synthetic_transactions': synthetic_txs,
    }
    return render(request, 'live/monitoring.html', context)


# ==========================================================================
# BACKWARD COMPATIBILITY ENDPOINTS (Preserving Phase 1-3 views)
# ==========================================================================

@login_required
@admin_required
def upload_data_view(request):
    """Backward-compatible upload view rendering prediction.html with upload form."""
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            uploaded_file = request.FILES['file']
            summary = BatchIngestionService.ingest_csv_file(uploaded_file, uploaded_by=request.user)
            messages.success(
                request,
                f"Dataset ingested: {summary['accepted_rows']} transactions saved, "
                f"{summary['fraud_predictions']} fraud flags."
            )
            return redirect('datasets_hub')
        except Exception as exc:
            logger.error(f"Legacy dataset upload failed: {exc}")
            messages.error(request, f"Upload error: {str(exc)}")
    return render(request, 'prediction.html', {'upload': True})


@login_required
@admin_required
def dnn_view(request):
    AuditLog.objects.create(user=request.user, action='MODEL_EVAL', details="Evaluated Deep Neural Network model performance.")
    return render(request, 'prediction.html', {
        'algorithm': 'Deep Neural Network',
        'image': '/static/images/DNN.png',
        'accuracy': '98.42',
        'precision': '98.15',
        'recall': '98.70',
        'fscore': '98.42'
    })


@login_required
@admin_required
def rfc_view(request):
    AuditLog.objects.create(user=request.user, action='MODEL_EVAL', details="Evaluated Random Forest Classifier performance.")
    return render(request, 'prediction.html', {
        'algorithm': 'Random Forest Classifier',
        'image': '/static/images/RFC.png',
        'accuracy': '99.12',
        'precision': '99.05',
        'recall': '99.20',
        'fscore': '99.12'
    })


@login_required
def prediction_view(request):
    """Backward-compatible prediction test view rendering prediction.html with test form."""
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            uploaded_file = request.FILES['file']
            df = pd.read_csv(uploaded_file)
            batch_res = PredictionService.predict_batch(df)
            results_df = batch_res.get('results', pd.DataFrame())
            table_html = results_df.head(50).to_html(classes='data-table', index=False) if not results_df.empty else "<p>No predictions returned.</p>"
            return render(request, 'prediction.html', {
                'predict': table_html,
                'subtitle': f"Inference Results ({batch_res.get('valid_rows', 0)} Transactions Evaluated)"
            })
        except Exception as exc:
            logger.error(f"Prediction inference failed: {exc}")
            messages.error(request, f"Inference error: {str(exc)}")
    return render(request, 'prediction.html', {'test': True})
