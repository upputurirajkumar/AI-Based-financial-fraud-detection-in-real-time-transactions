import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from django.db.models import Count, Avg, Sum, Q, Min, Max
from django.db.models.functions import TruncDate, TruncHour
from django.utils import timezone

from apps.transactions.models import Transaction
from apps.fraud.models import Prediction, FraudAlert, InvestigationCase

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Robust financial fraud analytics service.
    Enforces strict mathematical separation between ML Model Predicted Fraud and Human Confirmed Fraud.
    Uses database-side aggregation to ensure sub-second analytical execution.
    """

    @classmethod
    def get_overview_metrics(cls, date_from=None, date_to=None) -> Dict[str, Any]:
        """
        Calculates high-level executive fraud metrics, separating statistical predictions from ground truth.
        """
        tx_qs = Transaction.objects.all()
        pred_qs = Prediction.objects.all()
        alert_qs = FraudAlert.objects.all()
        case_qs = InvestigationCase.objects.all()

        if date_from:
            tx_qs = tx_qs.filter(created_at__gte=date_from)
            pred_qs = pred_qs.filter(created_at__gte=date_from)
            alert_qs = alert_qs.filter(created_at__gte=date_from)
            case_qs = case_qs.filter(created_at__gte=date_from)

        if date_to:
            tx_qs = tx_qs.filter(created_at__lte=date_to)
            pred_qs = pred_qs.filter(created_at__lte=date_to)
            alert_qs = alert_qs.filter(created_at__lte=date_to)
            case_qs = case_qs.filter(created_at__lte=date_to)

        # 1. Transaction Volumes
        total_tx = tx_qs.count()
        total_volume = float(tx_qs.aggregate(s=Sum('amount'))['s'] or 0.0)

        # 2. Prediction Metrics (Predicted Fraud)
        total_evaluated = pred_qs.count()
        predicted_fraud_count = pred_qs.filter(decision_state=Prediction.DECISION_FRAUD).count()
        predicted_suspicious_count = pred_qs.filter(decision_state=Prediction.DECISION_SUSPICIOUS).count()
        predicted_legit_count = pred_qs.filter(decision_state=Prediction.DECISION_LEGITIMATE).count()

        predicted_fraud_rate = round(
            (predicted_fraud_count / max(total_evaluated, 1)) * 100.0, 2
        ) if total_evaluated > 0 else 0.0

        avg_risk_score = round(
            float(pred_qs.aggregate(a=Avg('risk_score'))['a'] or 0.0), 1
        )

        # 3. Investigation & Ground Truth Metrics (Confirmed Fraud)
        total_cases = case_qs.count()
        resolved_cases = case_qs.filter(status=InvestigationCase.STATUS_RESOLVED).count()
        confirmed_fraud_cases = case_qs.filter(resolution=InvestigationCase.RESOLUTION_CONFIRMED_FRAUD).count()
        false_positive_cases = case_qs.filter(resolution=InvestigationCase.RESOLUTION_FALSE_POSITIVE).count()

        confirmed_fraud_rate = round(
            (confirmed_fraud_cases / max(resolved_cases, 1)) * 100.0, 2
        ) if resolved_cases > 0 else 0.0

        false_positive_rate = round(
            (false_positive_cases / max(resolved_cases, 1)) * 100.0, 2
        ) if resolved_cases > 0 else 0.0

        # 4. Alert Summary
        total_alerts = alert_qs.count()
        unresolved_alerts = alert_qs.filter(status__in=[FraudAlert.STATUS_NEW, FraudAlert.STATUS_ACKNOWLEDGED, FraudAlert.STATUS_UNDER_REVIEW]).count()
        critical_alerts = alert_qs.filter(severity=FraudAlert.SEVERITY_CRITICAL).count()

        return {
            'period': {
                'from': date_from.isoformat() if date_from else None,
                'to': date_to.isoformat() if date_to else None,
            },
            'transaction_volume': {
                'total_transactions': total_tx,
                'total_monetary_volume': total_volume,
                'total_evaluated': total_evaluated,
            },
            'predicted_metrics': {
                'predicted_fraud_count': predicted_fraud_count,
                'predicted_suspicious_count': predicted_suspicious_count,
                'predicted_legitimate_count': predicted_legit_count,
                'predicted_fraud_rate_pct': predicted_fraud_rate,
                'average_risk_score': avg_risk_score,
                'definition': "Calculated as (Predicted Fraud Transactions / Total Evaluated Transactions) * 100"
            },
            'confirmed_ground_truth': {
                'total_cases': total_cases,
                'resolved_cases': resolved_cases,
                'confirmed_fraud_cases': confirmed_fraud_cases,
                'false_positive_cases': false_positive_cases,
                'confirmed_fraud_rate_pct': confirmed_fraud_rate,
                'false_positive_rate_pct': false_positive_rate,
                'definition': "Calculated as (Human Confirmed Fraud Cases / Total Resolved Cases) * 100"
            },
            'alert_summary': {
                'total_alerts': total_alerts,
                'unresolved_alerts': unresolved_alerts,
                'critical_alerts': critical_alerts,
            }
        }

    @classmethod
    def get_risk_distribution(cls) -> Dict[str, Any]:
        """Calculates distribution of transactions across configurable risk tiers."""
        total = Prediction.objects.count()
        counts = {
            'LOW': Prediction.objects.filter(risk_tier='LOW').count(),
            'MEDIUM': Prediction.objects.filter(risk_tier='MEDIUM').count(),
            'HIGH': Prediction.objects.filter(risk_tier='HIGH').count(),
            'CRITICAL': Prediction.objects.filter(risk_tier='CRITICAL').count(),
        }
        percentages = {
            k: round(v / max(total, 1) * 100.0, 2) for k, v in counts.items()
        }
        return {
            'total_predictions': total,
            'counts': counts,
            'percentages': percentages,
            'bands': {
                'LOW': '0.0 - 29.9',
                'MEDIUM': '30.0 - 59.9',
                'HIGH': '60.0 - 84.9',
                'CRITICAL': '85.0 - 100.0',
            }
        }

    @classmethod
    def get_channel_and_type_distribution(cls) -> Dict[str, Any]:
        """Aggregates transaction counts and volume by channel and transaction type."""
        types = list(Transaction.objects.values('transaction_type').annotate(
            count=Count('id'),
            total_amount=Sum('amount')
        ).order_by('-count'))

        channels = list(Transaction.objects.values('channel').annotate(
            count=Count('id'),
            total_amount=Sum('amount')
        ).order_by('-count'))

        for item in types:
            item['total_amount'] = float(item['total_amount'] or 0.0)
        for item in channels:
            item['total_amount'] = float(item['total_amount'] or 0.0)

        return {
            'by_transaction_type': types,
            'by_channel': channels
        }

    @classmethod
    def get_model_version_analytics(cls) -> List[Dict[str, Any]]:
        """Returns comparison metrics partitioned by active/historical model versions."""
        versions = list(Prediction.objects.values('model_name', 'model_version').annotate(
            total_predictions=Count('id'),
            fraud_predictions=Count('id', filter=Q(decision_state='FRAUD')),
            avg_risk_score=Avg('risk_score'),
            avg_probability=Avg('fraud_probability'),
        ).order_by('-total_predictions'))

        for v in versions:
            v['avg_risk_score'] = round(float(v['avg_risk_score'] or 0.0), 1)
            v['avg_probability'] = round(float(v['avg_probability'] or 0.0), 4)
            v['fraud_rate_pct'] = round(v['fraud_predictions'] / max(v['total_predictions'], 1) * 100.0, 2)

        return versions

    @classmethod
    def get_time_series_analytics(cls, days: int = 7, interval: str = 'daily') -> List[Dict[str, Any]]:
        """Aggregates time-series volume, predicted fraud, and alerts over a rolling time window."""
        start_date = timezone.now() - timedelta(days=days)

        if interval == 'hourly':
            trunc_fn = TruncHour('created_at')
        else:
            trunc_fn = TruncDate('created_at')

        tx_series = Transaction.objects.filter(created_at__gte=start_date).annotate(
            bucket=trunc_fn
        ).values('bucket').annotate(
            total_tx=Count('id'),
            total_amount=Sum('amount')
        ).order_by('bucket')

        pred_series = Prediction.objects.filter(created_at__gte=start_date).annotate(
            bucket=trunc_fn
        ).values('bucket').annotate(
            predicted_fraud=Count('id', filter=Q(decision_state='FRAUD')),
            avg_risk=Avg('risk_score')
        ).order_by('bucket')

        pred_lookup = {p['bucket']: p for p in pred_series}

        results = []
        for t in tx_series:
            bucket = t['bucket']
            bucket_str = bucket.isoformat() if hasattr(bucket, 'isoformat') else str(bucket)
            p_data = pred_lookup.get(bucket, {})
            results.append({
                'timestamp': bucket_str,
                'total_transactions': t['total_tx'],
                'total_volume': float(t['total_amount'] or 0.0),
                'predicted_fraud': p_data.get('predicted_fraud', 0),
                'average_risk_score': round(float(p_data.get('avg_risk', 0.0) or 0.0), 1)
            })

        return results
