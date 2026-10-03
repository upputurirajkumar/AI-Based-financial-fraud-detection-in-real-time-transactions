import csv
import io
import logging
from typing import Dict, Any, Optional
from django.utils import timezone
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied

from apps.transactions.models import Transaction
from apps.fraud.models import Prediction
from apps.core.models import AuditLog
from services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class ReportService:
    """
    Structured fraud intelligence reporting and authorized data export service.
    Enforces role permissions, data sanitization, and compliance audit tracking.
    """

    MAX_EXPORT_LIMIT = 5000

    @classmethod
    def generate_fraud_summary_report(
        cls,
        date_from=None,
        date_to=None,
        actor: Optional[User] = None
    ) -> Dict[str, Any]:
        """
        Produces a comprehensive executive fraud intelligence summary report.
        """
        overview = AnalyticsService.get_overview_metrics(date_from=date_from, date_to=date_to)
        risk_dist = AnalyticsService.get_risk_distribution()
        model_analytics = AnalyticsService.get_model_version_analytics()

        report = {
            'report_metadata': {
                'title': 'Enterprise Fraud Intelligence & Risk Assessment Report',
                'generated_at': timezone.now().isoformat(),
                'generated_by': actor.username if actor else 'System Administrator',
                'reporting_period': overview['period'],
                'compliance_notice': 'This report is prepared for compliance and risk operations. Predicted fraud is separated from human-confirmed fraud.'
            },
            'executive_summary': overview,
            'risk_distribution': risk_dist,
            'model_versions': model_analytics,
            'metric_definitions': {
                'predicted_fraud_rate': "Count of transactions where ML predicted probability >= operating threshold or composite risk tier is CRITICAL, divided by total evaluated transactions.",
                'confirmed_fraud_rate': "Count of formal investigation cases with human-verified resolution CONFIRMED_FRAUD, divided by total resolved cases.",
                'false_positive_rate': "Count of formal investigation cases with human-verified resolution FALSE_POSITIVE, divided by total resolved cases.",
                'risk_tiers': "Discrete bands: LOW (0.0-29.9), MEDIUM (30.0-59.9), HIGH (60.0-84.9), CRITICAL (85.0-100.0)."
            }
        }

        if actor:
            AuditLog.objects.create(
                user=actor,
                action='REPORT_EXPORT',
                details=f"Generated executive fraud summary report for period {overview['period']}"
            )

        return report

    @classmethod
    def export_transactions_csv(
        cls,
        actor: User,
        transaction_type: Optional[str] = None,
        risk_tier: Optional[str] = None,
        max_rows: int = 1000
    ) -> str:
        """
        Generates a sanitized CSV export of transactions with strict size limits and role enforcement.
        """
        # Enforce RBAC
        profile = getattr(actor, 'profile', None)
        if not (actor.is_staff or actor.is_superuser or (profile and profile.is_analyst_or_admin())):
            raise PermissionDenied("Only authorized Compliance Analysts and Administrators can export transaction records.")

        export_limit = min(max_rows, cls.MAX_EXPORT_LIMIT)

        qs = Transaction.objects.prefetch_related('predictions').all()
        if transaction_type:
            qs = qs.filter(transaction_type=transaction_type.upper())
        if risk_tier:
            qs = qs.filter(predictions__risk_tier=risk_tier.upper())

        qs = qs[:export_limit]

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'transaction_id', 'step', 'transaction_type', 'amount', 'currency', 'channel',
            'name_orig', 'old_balance_orig', 'new_balance_orig',
            'name_dest', 'old_balance_dest', 'new_balance_dest',
            'latest_decision_state', 'latest_risk_score', 'latest_risk_tier', 'latest_fraud_prob',
            'created_at'
        ])

        for t in qs:
            latest_pred = t.predictions.first()
            writer.writerow([
                t.transaction_id,
                t.step,
                t.transaction_type,
                float(t.amount),
                t.currency,
                t.channel,
                t.name_orig,
                float(t.old_balance_orig),
                float(t.new_balance_orig),
                t.name_dest,
                float(t.old_balance_dest),
                float(t.new_balance_dest),
                latest_pred.decision_state if latest_pred else 'UNASSESSED',
                latest_pred.risk_score if latest_pred else '',
                latest_pred.risk_tier if latest_pred else '',
                latest_pred.fraud_probability if latest_pred else '',
                t.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])

        AuditLog.objects.create(
            user=actor,
            action='REPORT_EXPORT',
            details=f"Exported {qs.count()} transactions to CSV (type={transaction_type}, tier={risk_tier})"
        )

        return output.getvalue()
