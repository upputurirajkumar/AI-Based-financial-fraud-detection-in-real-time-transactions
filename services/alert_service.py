import logging
import hashlib
from typing import Dict, Any, List, Optional
from django.utils import timezone
from django.db import transaction as db_transaction
from django.core.paginator import Paginator
from django.contrib.auth.models import User

from apps.fraud.models import FraudAlert, Prediction
from apps.transactions.models import Transaction
from apps.core.models import AuditLog

logger = logging.getLogger(__name__)


class AlertService:
    """
    Enterprise alert generation, deduplication, triage, and lifecycle management service.
    Translates raw ML predictions and deterministic risk signals into actionable compliance alerts.
    """

    RULE_CRITICAL_RISK = 'CRITICAL_RISK_THRESHOLD'
    RULE_HIGH_RISK = 'HIGH_RISK_THRESHOLD'
    RULE_HIGH_PROBABILITY = 'HIGH_MODEL_PROBABILITY'
    RULE_ANOMALY_EXCEEDED = 'UNSUPERVISED_ANOMALY_EXCEEDED'
    RULE_ACCOUNT_DRAINAGE = 'ORIGIN_ACCOUNT_DRAINED'

    @classmethod
    def determine_alert_rules(cls, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Determines which specific alert rules are triggered by an analysis result.
        Returns a list of rule definitions: [{'rule_name': ..., 'title': ..., 'description': ...}]
        """
        rules = []
        risk_score = float(analysis_result.get('risk_score', 0.0) or 0.0)
        risk_level = analysis_result.get('risk_level', 'LOW')
        prob = analysis_result.get('fraud_probability')
        anomaly_info = analysis_result.get('anomaly_info', {})
        inv_meta = analysis_result.get('investigation_metadata', {})

        # Rule 1: Critical Risk Threshold
        if risk_level == 'CRITICAL' or risk_score >= 85.0:
            rules.append({
                'rule_name': cls.RULE_CRITICAL_RISK,
                'title': f"Critical Financial Risk Detected ({risk_score}/100)",
                'description': f"Transaction triggered critical composite risk score of {risk_score}."
            })
        # Rule 2: High Risk Threshold
        elif risk_level == 'HIGH' or risk_score >= 60.0:
            rules.append({
                'rule_name': cls.RULE_HIGH_RISK,
                'title': f"High Risk Transaction Flagged ({risk_score}/100)",
                'description': f"Transaction risk score of {risk_score} exceeded high monitoring threshold."
            })

        # Rule 3: High Model Probability
        if prob is not None and prob >= 0.75:
            rules.append({
                'rule_name': cls.RULE_HIGH_PROBABILITY,
                'title': f"High Model Probability ({prob:.1%})",
                'description': f"Primary classification pipeline produced high anomaly probability of {prob:.1%}."
            })

        # Rule 4: Unsupervised Anomaly
        if anomaly_info.get('is_anomalous'):
            anom_score = anomaly_info.get('anomaly_score', 0.0)
            rules.append({
                'rule_name': cls.RULE_ANOMALY_EXCEEDED,
                'title': f"Structural Anomaly Flagged ({anom_score}/100)",
                'description': f"Unsupervised detector identified significant divergence from baseline normal transactions."
            })

        # Rule 5: Account Drainage
        if inv_meta.get('orig_drained'):
            rules.append({
                'rule_name': cls.RULE_ACCOUNT_DRAINAGE,
                'title': "Total Origin Balance Depleted",
                'description': "Transaction depleted 100% of origin available funds in a single event."
            })

        return rules

    @classmethod
    def evaluate_and_generate_alerts(
        cls,
        prediction_record: Prediction,
        analysis_result: Dict[str, Any],
        transaction_record: Optional[Transaction] = None
    ) -> List[FraudAlert]:
        """
        Evaluates prediction and creates deduplicated candidate FraudAlerts.
        Uses deterministic hash deduplication: sha256(tx_id:model_version:rule_name).
        """
        tx = transaction_record or prediction_record.transaction
        tx_id = tx.transaction_id if tx else prediction_record.prediction_id
        model_version = prediction_record.model_version
        risk_level = analysis_result.get('risk_level', 'LOW')

        # Map Severity from Risk Level
        if risk_level == 'CRITICAL':
            severity = FraudAlert.SEVERITY_CRITICAL
        elif risk_level == 'HIGH':
            severity = FraudAlert.SEVERITY_HIGH
        elif risk_level == 'MEDIUM':
            severity = FraudAlert.SEVERITY_MEDIUM
        else:
            severity = FraudAlert.SEVERITY_LOW

        triggered_rules = cls.determine_alert_rules(analysis_result)
        created_alerts = []

        for rule in triggered_rules:
            rule_name = rule['rule_name']
            dedup_key = f"{tx_id}:{model_version}:{rule_name}"
            dedup_hash = hashlib.sha256(dedup_key.encode('utf-8')).hexdigest()

            # Atomic get or create to prevent race conditions during concurrent ingestion
            with db_transaction.atomic():
                existing = FraudAlert.objects.filter(deduplication_hash=dedup_hash).first()
                if existing:
                    logger.debug(f"Deduplicated existing alert {existing.alert_id} for hash {dedup_hash[:8]}")
                    created_alerts.append(existing)
                    continue

                alert = FraudAlert.objects.create(
                    transaction=tx,
                    prediction=prediction_record,
                    title=f"[{tx_id}] {rule['title']}",
                    description=rule['description'],
                    rule_name=rule_name,
                    severity=severity,
                    status=FraudAlert.STATUS_NEW,
                    deduplication_hash=dedup_hash,
                    notes=f"Auto-generated alert by rule '{rule_name}' at {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}."
                )
                logger.info(f"Created new FraudAlert {alert.alert_id} for Transaction {tx_id} [{rule_name}]")
                created_alerts.append(alert)

        return created_alerts

    @classmethod
    def evaluate_and_create_alerts(
        cls,
        transaction: Optional[Transaction] = None,
        prediction: Optional[Prediction] = None,
        risk_factors: Optional[List[Dict[str, Any]]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> List[FraudAlert]:
        """
        Convenience wrapper to evaluate and generate alerts directly from transaction and prediction records.
        """
        if prediction is None and transaction is not None:
            prediction = transaction.predictions.first()
        if prediction is None:
            return []

        if analysis_result is None:
            analysis_result = {
                'risk_score': prediction.risk_score,
                'risk_level': prediction.risk_tier,
                'fraud_probability': prediction.fraud_probability,
                'anomaly_info': {
                    'is_anomalous': prediction.is_anomalous,
                    'anomaly_score': prediction.anomaly_score or 0.0,
                },
                'investigation_metadata': {
                    'orig_drained': (
                        transaction is not None and
                        transaction.old_balance_orig > 0 and
                        transaction.amount == transaction.old_balance_orig
                    )
                }
            }

        return cls.evaluate_and_generate_alerts(
            prediction_record=prediction,
            analysis_result=analysis_result,
            transaction_record=transaction
        )

    # Backward compatibility helper for Phase 4 tests
    @classmethod
    def evaluate_for_alert(

        cls,
        prediction_record: Prediction,
        analysis_result: Dict[str, Any]
    ) -> Optional[FraudAlert]:
        alerts = cls.evaluate_and_generate_alerts(prediction_record, analysis_result)
        return alerts[0] if alerts else None

    @classmethod
    def get_review_queue(
        cls,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        assigned_to_id: Optional[int] = None,
        date_from=None,
        date_to=None,
        page: int = 1,
        page_size: int = 25
    ) -> Dict[str, Any]:
        """
        Retrieves paginated review queue of alerts with multi-dimensional filtering.
        """
        qs = FraudAlert.objects.select_related('transaction', 'prediction', 'assigned_to').all()

        if status:
            qs = qs.filter(status=status)
        if severity:
            qs = qs.filter(severity=severity)
        if assigned_to_id:
            qs = qs.filter(assigned_to_id=assigned_to_id)
        if date_from:
            qs = qs.filter(created_at__gte=date_from)
        if date_to:
            qs = qs.filter(created_at__lte=date_to)

        paginator = Paginator(qs, page_size)
        current_page = paginator.get_page(page)

        items = []
        for a in current_page:
            items.append({
                'alert_id': a.alert_id,
                'title': a.title,
                'severity': a.severity,
                'status': a.status,
                'rule_name': a.rule_name,
                'transaction_id': a.transaction.transaction_id if a.transaction else None,
                'amount': float(a.transaction.amount) if a.transaction else None,
                'risk_score': a.prediction.risk_score if a.prediction else None,
                'assigned_to': a.assigned_to.username if a.assigned_to else None,
                'created_at': a.created_at.isoformat(),
            })

        return {
            'total_count': paginator.count,
            'total_pages': paginator.num_pages,
            'current_page': current_page.number,
            'page_size': page_size,
            'alerts': items
        }
