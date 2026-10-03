import json
import logging
from typing import Any, Dict, List, Optional
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError, PermissionDenied

from apps.transactions.models import Transaction
from apps.fraud.models import FraudAlert, Prediction
from apps.accounts.decorators import admin_required
from services.transaction_service import TransactionService
from services.alert_service import AlertService
from services.investigation_service import InvestigationService
from services.analytics_service import AnalyticsService
from services.report_service import ReportService
from services.fraud_engine import FraudEngine


logger = logging.getLogger(__name__)


def standard_response(data: Any = None, meta: Dict[str, Any] = None, status: int = 200) -> JsonResponse:
    """Produces standardized JSON success contract: {success: true, data: ..., meta: ...}"""
    payload = {
        'success': True,
        'data': data if data is not None else {},
    }
    if meta is not None:
        payload['meta'] = meta
    return JsonResponse(payload, status=status)


def standard_error(code: str, message: str, status: int = 400, details: Any = None) -> JsonResponse:
    """Produces standardized JSON error contract: {success: false, error: {code: ..., message: ...}}"""
    payload = {
        'success': False,
        'error': {
            'code': code,
            'message': message
        }
    }
    if details is not None:
        payload['error']['details'] = details
    return JsonResponse(payload, status=status)


def require_analyst_or_admin(user) -> bool:
    """Helper verifying that user is at least a Fraud Risk Analyst or System Admin."""
    if user.is_staff or user.is_superuser:
        return True
    profile = getattr(user, 'profile', None)
    return bool(profile and profile.is_analyst_or_admin())


# ==========================================
# 1. TRANSACTION INTELLIGENCE APIS
# ==========================================

@login_required
@require_http_methods(["GET"])
def api_search_transactions(request):
    """GET /api/transactions/ - Search & filter transactions with pagination."""
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 25))
        min_amount = float(request.GET['min_amount']) if request.GET.get('min_amount') else None
        max_amount = float(request.GET['max_amount']) if request.GET.get('max_amount') else None
        has_alerts_str = request.GET.get('has_alerts')
        has_alerts = True if has_alerts_str == 'true' else (False if has_alerts_str == 'false' else None)

        res = TransactionService.search_transactions(
            search_query=request.GET.get('q'),
            transaction_type=request.GET.get('type'),
            min_amount=min_amount,
            max_amount=max_amount,
            risk_tier=request.GET.get('risk_tier'),
            decision_state=request.GET.get('decision_state'),
            channel=request.GET.get('channel'),
            has_alerts=has_alerts,
            page=page,
            page_size=page_size
        )
        return standard_response(
            data=res['transactions'],
            meta={
                'total_count': res['total_count'],
                'total_pages': res['total_pages'],
                'current_page': res['current_page'],
                'page_size': res['page_size']
            }
        )
    except Exception as exc:
        logger.error(f"api_search_transactions error: {exc}")
        return standard_error('QUERY_ERROR', str(exc), status=500)


@login_required
@require_http_methods(["GET"])
def api_transaction_detail(request, tx_id):
    """GET /api/transactions/<tx_id>/ - Complete transaction intelligence object."""
    detail = TransactionService.get_transaction_detail(tx_id)
    if not detail:
        return standard_error('NOT_FOUND', f"Transaction '{tx_id}' not found.", status=404)
    return standard_response(data=detail)


@login_required
@require_http_methods(["POST"])
def api_ingest_transaction(request):
    """POST /api/transactions/ingest/ - Ingest, evaluate, and record a transaction or standardized event contract."""
    try:
        body = json.loads(request.body.decode('utf-8'))

        # Standardized Transaction Event Contract (Phase 7)
        if 'transaction_data' in body or 'event_id' in body:
            from services.event_processing_service import EventProcessingService
            result = EventProcessingService.process_transaction_event(body, actor=request.user)
            if not result.get('success'):
                return standard_error('VALIDATION_FAILED', 'Transaction event failed schema validation.', details=result.get('errors'))
            return standard_response(data=result, status=201)

        channel = body.get('channel', 'ONLINE')
        result = TransactionService.ingest_and_evaluate_transaction(
            data=body,
            actor=request.user,
            channel=channel
        )
        if not result['success']:
            return standard_error('VALIDATION_FAILED', 'Transaction failed schema validation.', details=result.get('errors'))
        return standard_response(data=result, status=201)
    except json.JSONDecodeError:
        return standard_error('MALFORMED_JSON', 'Request body must be valid JSON.', status=400)
    except Exception as exc:
        logger.error(f"api_ingest_transaction error: {exc}")
        return standard_error('INGEST_ERROR', str(exc), status=500)



# ==========================================
# 2. FRAUD ALERTS & REVIEW QUEUE APIS
# ==========================================

@login_required
@require_http_methods(["GET"])
def api_list_alerts(request):
    """GET /api/alerts/ - Paginated alert review queue with filters."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', 'Access restricted to Compliance Analysts and Administrators.', status=403)

    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 25))
        assigned_id = int(request.GET['assigned_to']) if request.GET.get('assigned_to') else None

        queue = AlertService.get_review_queue(
            status=request.GET.get('status'),
            severity=request.GET.get('severity'),
            assigned_to_id=assigned_id,
            page=page,
            page_size=page_size
        )
        return standard_response(
            data=queue['alerts'],
            meta={
                'total_count': queue['total_count'],
                'total_pages': queue['total_pages'],
                'current_page': queue['current_page'],
                'page_size': queue['page_size']
            }
        )
    except Exception as exc:
        return standard_error('ALERT_QUEUE_ERROR', str(exc), status=500)


@login_required
@require_http_methods(["POST", "PATCH"])
def api_transition_alert(request, alert_id):
    """PATCH /api/alerts/<alert_id>/transition/ - Transition alert state."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', 'Access restricted to Compliance Analysts and Administrators.', status=403)

    from apps.fraud.models import FraudAlert
    alert = FraudAlert.objects.filter(alert_id=alert_id).first()
    if not alert:
        return standard_error('NOT_FOUND', f"Alert '{alert_id}' not found.", status=404)

    try:
        body = json.loads(request.body.decode('utf-8'))
        new_status = body.get('status')
        notes = body.get('notes', '')
        if not new_status:
            return standard_error('BAD_REQUEST', "Missing required field 'status'.", status=400)

        alert.transition_to(new_status, request.user, notes=notes)
        return standard_response(data={
            'alert_id': alert.alert_id,
            'status': alert.status,
            'updated_at': alert.updated_at.isoformat()
        })
    except ValidationError as ve:
        return standard_error('INVALID_TRANSITION', str(ve), status=400)
    except Exception as exc:
        return standard_error('TRANSITION_ERROR', str(exc), status=500)


# ==========================================
# 3. INVESTIGATION CASE MANAGEMENT APIS
# ==========================================

@login_required
@require_http_methods(["GET", "POST"])
def api_cases_collection(request):
    """GET/POST /api/cases/ - List or create investigation cases."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', 'Access restricted to Compliance Analysts and Administrators.', status=403)

    if request.method == 'GET':
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 25))
        res = InvestigationService.list_cases(
            status=request.GET.get('status'),
            priority=request.GET.get('priority'),
            resolution=request.GET.get('resolution'),
            page=page,
            page_size=page_size
        )
        return standard_response(
            data=res['cases'],
            meta={
                'total_count': res['total_count'],
                'total_pages': res['total_pages'],
                'current_page': res['current_page']
            }
        )

    # POST - Create Case
    try:
        body = json.loads(request.body.decode('utf-8'))
        title = body.get('title')
        if not title:
            return standard_error('BAD_REQUEST', "Field 'title' is required.", status=400)

        case = InvestigationService.create_case(
            title=title,
            opened_by=request.user,
            description=body.get('description', ''),
            priority=body.get('priority', 'MEDIUM'),
            transaction_ids=body.get('transaction_ids', []),
            alert_ids=body.get('alert_ids', [])
        )
        return standard_response(data={'case_id': case.case_id, 'status': case.status}, status=201)
    except Exception as exc:
        return standard_error('CREATE_CASE_ERROR', str(exc), status=500)


@login_required
@require_http_methods(["GET"])
def api_case_detail(request, case_id):
    """GET /api/cases/<case_id>/ - Full case details with notes and linked artifacts."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', 'Access restricted to Compliance Analysts and Administrators.', status=403)

    detail = InvestigationService.get_case_detail(case_id)
    if not detail:
        return standard_error('NOT_FOUND', f"Case '{case_id}' not found.", status=404)
    return standard_response(data=detail)


@login_required
@require_http_methods(["POST"])
def api_case_add_note(request, case_id):
    """POST /api/cases/<case_id>/notes/ - Append investigation note."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', 'Access restricted to Compliance Analysts and Administrators.', status=403)

    try:
        body = json.loads(request.body.decode('utf-8'))
        note_text = body.get('note')
        if not note_text:
            return standard_error('BAD_REQUEST', "Field 'note' is required.", status=400)

        note = InvestigationService.add_note(case_id, request.user, note_text)
        return standard_response(data={'note_id': note.id, 'created_at': note.created_at.isoformat()}, status=201)
    except ValidationError as ve:
        return standard_error('NOT_FOUND', str(ve), status=404)
    except Exception as exc:
        return standard_error('NOTE_ERROR', str(exc), status=500)


@login_required
@require_http_methods(["POST", "PATCH"])
def api_case_resolve(request, case_id):
    """PATCH /api/cases/<case_id>/resolve/ - Formal ground-truth resolution."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', 'Access restricted to Compliance Analysts and Administrators.', status=403)

    try:
        body = json.loads(request.body.decode('utf-8'))
        resolution = body.get('resolution')
        notes = body.get('notes', '')
        if not resolution:
            return standard_error('BAD_REQUEST', "Field 'resolution' is required.", status=400)

        case = InvestigationService.resolve_case(case_id, resolution, notes, request.user)
        return standard_response(data={
            'case_id': case.case_id,
            'status': case.status,
            'resolution': case.resolution,
            'closed_at': case.closed_at.isoformat()
        })
    except ValidationError as ve:
        return standard_error('INVALID_RESOLUTION', str(ve), status=400)
    except Exception as exc:
        return standard_error('RESOLVE_ERROR', str(exc), status=500)


# ==========================================
# 4. ANALYTICS & REPORTING APIS
# ==========================================

@login_required
@require_http_methods(["GET"])
def api_analytics_overview(request):
    """GET /api/analytics/overview/ - Key fraud ratios, separated predictions vs ground truth."""
    data = AnalyticsService.get_overview_metrics()
    return standard_response(data=data)


@login_required
@require_http_methods(["GET"])
def api_analytics_risk_distribution(request):
    """GET /api/analytics/risk-distribution/ - Risk tiers counts and percentages."""
    data = AnalyticsService.get_risk_distribution()
    return standard_response(data=data)


@login_required
@require_http_methods(["GET"])
def api_analytics_time_series(request):
    """GET /api/analytics/time-series/ - Rolling daily/hourly timeline metrics."""
    days = int(request.GET.get('days', 7))
    interval = request.GET.get('interval', 'daily')
    data = AnalyticsService.get_time_series_analytics(days=days, interval=interval)
    return standard_response(data=data)


@login_required
@require_http_methods(["GET"])
def api_analytics_models(request):
    """GET /api/analytics/models/ - Per-model version analytical performance."""
    data = AnalyticsService.get_model_version_analytics()
    return standard_response(data=data)


@login_required
@require_http_methods(["GET"])
def api_reports_summary(request):
    """GET /api/reports/summary/ - Executive summary report."""
    report = ReportService.generate_fraud_summary_report(actor=request.user)
    return standard_response(data=report)


@login_required
@require_http_methods(["GET"])
def api_reports_export_csv(request):
    """GET /api/reports/export/ - Role-governed CSV data export."""
    try:
        csv_content = ReportService.export_transactions_csv(
            actor=request.user,
            transaction_type=request.GET.get('type'),
            risk_tier=request.GET.get('risk_tier'),
            max_rows=int(request.GET.get('limit', 1000))
        )
        response = HttpResponse(csv_content, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="transactions_export.csv"'
        return response
    except PermissionDenied as pd:
        return standard_error('FORBIDDEN', str(pd), status=403)
    except Exception as exc:
        return standard_error('EXPORT_ERROR', str(exc), status=500)


# ==========================================
# 5. LIVE MONITORING & SYNTHETIC STREAM APIS
# ==========================================

@login_required
@require_http_methods(["GET"])
def api_live_status(request):
    """GET /api/live/status/ - Retrieve current synthetic stream status, rate, and counters."""
    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    return standard_response(data=manager.get_status())


@login_required
@require_http_methods(["POST"])
def api_live_start(request):
    """POST /api/live/start/ - Start synthetic transaction stream under safety bounds."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', "Only Risk Analysts or Administrators can initiate synthetic streams.", status=403)

    try:
        body = json.loads(request.body) if request.body else {}
    except Exception:
        body = {}

    scenario = body.get('scenario', 'NORMAL')
    rate = int(body.get('rate_per_minute', 15))
    max_events = int(body.get('max_events', 100))

    from services.live_stream_manager import LiveStreamManager
    try:
        manager = LiveStreamManager()
        status = manager.start(scenario=scenario, rate_per_minute=rate, max_events=max_events, actor=request.user)
        return standard_response(data=status)
    except PermissionDenied as pd:
        return standard_error('FORBIDDEN', str(pd), status=403)
    except Exception as exc:
        logger.error("Failed to start synthetic stream: %s", exc)
        return standard_error('STREAM_START_FAILED', str(exc), status=500)


@login_required
@require_http_methods(["POST"])
def api_live_pause(request):
    """POST /api/live/pause/ - Pause synthetic stream generation."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', "Only Risk Analysts or Administrators can control streams.", status=403)

    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    status = manager.pause(actor=request.user)
    return standard_response(data=status)


@login_required
@require_http_methods(["POST"])
def api_live_resume(request):
    """POST /api/live/resume/ - Resume active synthetic stream generation."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', "Only Risk Analysts or Administrators can control streams.", status=403)

    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    status = manager.resume(actor=request.user)
    return standard_response(data=status)


@login_required
@require_http_methods(["POST"])
def api_live_stop(request):
    """POST /api/live/stop/ - Stop and finalize current synthetic stream."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', "Only Risk Analysts or Administrators can control streams.", status=403)

    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    status = manager.stop(actor=request.user)
    return standard_response(data=status)


@login_required
@require_http_methods(["POST"])
def api_live_step(request):
    """POST /api/live/step/ - Execute a single synthetic transaction step on-demand."""
    if not require_analyst_or_admin(request.user):
        return standard_error('FORBIDDEN', "Only Risk Analysts or Administrators can control streams.", status=403)

    try:
        body = json.loads(request.body) if request.body else {}
    except Exception:
        body = {}

    scenario = body.get('scenario')

    from services.live_stream_manager import LiveStreamManager
    try:
        manager = LiveStreamManager()
        result = manager.step(scenario=scenario, actor=request.user)
        return standard_response(data=result)
    except PermissionDenied as pd:
        return standard_error('FORBIDDEN', str(pd), status=403)
    except Exception as exc:
        logger.error("Failed to execute synthetic step: %s", exc)
        return standard_error('STREAM_STEP_FAILED', str(exc), status=500)


@login_required
@require_http_methods(["GET"])
def api_live_stream_delta(request):
    """GET /api/live/stream/ - Poll live events, triggered alerts, and metrics delta."""
    since = request.GET.get('since')
    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    delta = manager.get_stream_delta(since_timestamp=since)
    return standard_response(data=delta)


@login_required
@require_http_methods(["GET"])
def api_live_transactions(request):
    """GET /api/live/transactions/ - Retrieve recent live stream transactions."""
    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    with manager._lock:
        events = list(manager.recent_events)
    if not events:
        synthetic_txs = Transaction.objects.filter(is_synthetic=True).select_related().prefetch_related('predictions')[:25]
        events = []
        for tx in synthetic_txs:
            pred = tx.predictions.first()
            events.append({
                'event_id': tx.event_id,
                'transaction_id': tx.transaction_id,
                'source': tx.source,
                'scenario': tx.scenario_tag or 'NORMAL',
                'transaction_type': tx.transaction_type,
                'amount': float(tx.amount),
                'channel': tx.channel,
                'name_orig': tx.name_orig,
                'name_dest': tx.name_dest,
                'decision_state': pred.decision_state if pred else 'LEGITIMATE',
                'risk_score': pred.risk_score if pred else 0.0,
                'risk_tier': pred.risk_tier if pred else 'LOW',
                'fraud_probability': pred.fraud_probability if pred else 0.0,
                'anomaly_score': pred.anomaly_score if pred else None,
                'latency_ms': tx.latency_ms,
                'status': tx.processing_status,
                'timestamp': tx.created_at.isoformat(),
            })
    return standard_response(data=events)


@login_required
@require_http_methods(["GET"])
def api_live_alerts(request):
    """GET /api/live/alerts/ - Retrieve recent stream alerts."""
    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    with manager._lock:
        alerts = list(manager.recent_alerts)
    if not alerts:
        db_alerts = FraudAlert.objects.filter(transaction__is_synthetic=True).select_related('transaction', 'prediction')[:25]
        alerts = [{
            'alert_id': a.alert_id,
            'title': a.title,
            'severity': a.severity,
            'rule_name': a.rule_name,
            'transaction_id': a.transaction.transaction_id if a.transaction else None,
            'risk_score': a.prediction.risk_score if a.prediction else None,
            'timestamp': a.created_at.isoformat(),
        } for a in db_alerts]
    return standard_response(data=alerts)


@login_required
@require_http_methods(["GET"])
def api_live_metrics(request):
    """GET /api/live/metrics/ - Retrieve real-time stream telemetry and performance metrics."""
    from services.live_stream_manager import LiveStreamManager
    manager = LiveStreamManager()
    status = manager.get_status()
    metrics = {
        'status': status['status'],
        'rate_per_minute': status['rate_per_minute'],
        'processed_count': status['processed_count'],
        'alerts_count': status['alerts_count'],
        'high_risk_count': status['high_risk_count'],
        'failure_count': status['failure_count'],
        'average_latency_ms': status['average_latency_ms'],
        'risk_distribution': status['risk_distribution'],
        'risk_distribution_pct': status['risk_distribution_pct'],
        'is_demo_stream': True,
        'stream_label': status['stream_label'],
    }
    return standard_response(data=metrics)

