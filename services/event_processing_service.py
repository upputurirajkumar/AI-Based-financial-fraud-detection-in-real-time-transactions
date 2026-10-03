import time
import logging
from decimal import Decimal
from typing import Dict, Any, Optional
from django.db import transaction as db_transaction
from django.conf import settings
from django.core.exceptions import PermissionDenied, ValidationError
from django.contrib.auth.models import User
from django.utils import timezone

from apps.transactions.models import Transaction
from apps.fraud.models import Prediction, FraudAlert
from apps.core.models import AuditLog
from services.fraud_engine import FraudEngine
from services.alert_service import AlertService

logger = logging.getLogger(__name__)


class EventProcessingService:
    """
    Authoritative real-time event processing engine for transaction streams.
    Enforces idempotency, measures processing latencies, validates events,
    executes Phase 4 FraudEngine risk intelligence, creates alerts, and persists records.
    """

    @classmethod
    def process_transaction_event(
        cls,
        event: Dict[str, Any],
        actor: Optional[User] = None
    ) -> Dict[str, Any]:
        """
        Processes a single transaction event through the complete risk intelligence pipeline.
        Guaranteed to be idempotent: duplicate event IDs return existing analysis.
        """
        received_at = time.time()
        start_perf = time.perf_counter()

        # 1. Environment & Demo Security Safeguards
        source = str(event.get('source', 'INTERNAL')).upper()
        if source == 'SYNTHETIC' and not getattr(settings, 'REAL_TIME_DEMO_ENABLED', True):
            raise PermissionDenied("Synthetic transaction demo streams are disabled in this environment.")

        event_id = event.get('event_id')
        transaction_id = event.get('transaction_id')
        scenario = str(event.get('scenario', 'NORMAL')).upper()
        tx_data = event.get('transaction_data', {})

        # 2. Idempotency Check
        if event_id:
            existing_tx = Transaction.objects.filter(event_id=event_id).first()
            if existing_tx:
                logger.info("Idempotent hit: event_id %s already processed as %s", event_id, existing_tx.transaction_id)
                return cls._build_idempotent_response(existing_tx)

        if transaction_id:
            existing_tx = Transaction.objects.filter(transaction_id=transaction_id).first()
            if existing_tx:
                logger.info("Idempotent hit: transaction_id %s already exists", transaction_id)
                return cls._build_idempotent_response(existing_tx)

        # 3. Payload Validation
        validation_errors = FraudEngine.validate_transaction(tx_data)
        if validation_errors:
            logger.warning("Event validation failed for event_id %s: %s", event_id, validation_errors)
            return {
                'success': False,
                'status': 'VALIDATION_FAILED',
                'event_id': event_id,
                'transaction_id': transaction_id,
                'errors': validation_errors,
                'latency_ms': round((time.perf_counter() - start_perf) * 1000, 2),
                'timestamp': timezone.now().isoformat(),
            }

        # 4. Atomic Ingestion, Inference & Persistence
        try:
            with db_transaction.atomic():
                # Step A: Persist Transaction Record
                amount_val = Decimal(str(tx_data.get('amount', 0.0)))
                old_orig_val = Decimal(str(tx_data.get('oldbalanceOrg', 0.0)))
                new_orig_val = Decimal(str(tx_data.get('newbalanceOrig', 0.0)))
                old_dest_val = Decimal(str(tx_data.get('oldbalanceDest', 0.0)))
                new_dest_val = Decimal(str(tx_data.get('newbalanceDest', 0.0)))

                tx = Transaction(
                    transaction_id=transaction_id or '',
                    step=int(tx_data.get('step', 1)),
                    transaction_type=str(tx_data.get('type', 'PAYMENT')).upper(),
                    amount=amount_val,
                    currency=str(tx_data.get('currency', 'USD')),
                    channel=str(tx_data.get('channel', 'ONLINE')).upper(),
                    name_orig=str(tx_data.get('nameOrig', 'C_UNKNOWN')),
                    old_balance_orig=old_orig_val,
                    new_balance_orig=new_orig_val,
                    name_dest=str(tx_data.get('nameDest', 'M_UNKNOWN')),
                    old_balance_dest=old_dest_val,
                    new_balance_dest=new_dest_val,
                    source=source,
                    is_synthetic=(source == 'SYNTHETIC'),
                    event_id=event_id,
                    scenario_tag=scenario,
                    processing_status=Transaction.STATUS_PROCESSING,
                    raw_metadata=event.get('metadata', tx_data.get('metadata', {}))
                )
                tx.save()

                # Step B: Execute Phase 4 FraudEngine (Latency Timed)
                inf_start = time.perf_counter()
                inference_payload = tx.to_inference_dict()
                analysis = FraudEngine.analyze_transaction(inference_payload)
                inference_duration_ms = round((time.perf_counter() - inf_start) * 1000, 2)

                # Step C: Persist Prediction Model Snapshot
                pred = Prediction.objects.create(
                    transaction=tx,
                    model_name=analysis['model_metadata'].get('model_name', 'Random Forest Classifier'),
                    model_version=analysis['model_metadata'].get('model_version', '1.0.0'),
                    decision_threshold=analysis['threshold'],
                    fraud_probability=analysis['fraud_probability'],
                    risk_score=analysis['risk_score'],
                    risk_tier=analysis['risk_level'],
                    decision_state=analysis['decision_state'],
                    is_fraud=(analysis['decision_state'] == FraudEngine.DECISION_FRAUD),
                    anomaly_score=analysis['anomaly_info'].get('anomaly_score'),
                    is_anomalous=analysis['anomaly_info'].get('is_anomalous', False),
                    explanation=analysis['explanation'].get('summary_text', ''),
                    raw_explanation_json=analysis['explanation'],
                    inference_duration_ms=inference_duration_ms,
                    evaluated_by=actor
                )

                # Step D: Alert Rule Evaluation & Persistence
                alerts = AlertService.evaluate_and_create_alerts(
                    transaction=tx,
                    prediction=pred,
                    risk_factors=analysis.get('risk_factors', []),
                    analysis_result=analysis
                )


                # Step E: Update Transaction Processing Status & Latency
                total_duration_ms = round((time.perf_counter() - start_perf) * 1000, 2)
                tx.latency_ms = total_duration_ms
                tx.processing_status = Transaction.STATUS_COMPLETED
                tx.save(update_fields=['latency_ms', 'processing_status'])

                # Step F: Audit Log for Stream Ingestion
                if actor and actor.is_authenticated:
                    AuditLog.objects.create(
                        user=actor,
                        action='STREAM_INGEST',
                        details=f"Processed real-time event {tx.transaction_id} ({source}/{scenario}) -> {pred.decision_state}"
                    )

                logger.debug(
                    "Event %s processed as %s: State=%s, Score=%.1f, Latency=%.2fms",
                    event_id, tx.transaction_id, pred.decision_state, pred.risk_score, total_duration_ms
                )

                return {
                    'success': True,
                    'status': 'COMPLETED',
                    'event_id': tx.event_id,
                    'transaction_id': tx.transaction_id,
                    'source': tx.source,
                    'scenario': tx.scenario_tag,
                    'transaction_type': tx.transaction_type,
                    'amount': float(tx.amount),
                    'channel': tx.channel,
                    'name_orig': tx.name_orig,
                    'name_dest': tx.name_dest,
                    'decision_state': pred.decision_state,
                    'risk_score': pred.risk_score,
                    'risk_tier': pred.risk_tier,
                    'fraud_probability': pred.fraud_probability,
                    'anomaly_score': pred.anomaly_score,
                    'alerts_count': len(alerts),
                    'alerts': [
                        {
                            'alert_id': a.alert_id,
                            'title': a.title,
                            'severity': a.severity,
                            'rule_name': a.rule_name
                        } for a in alerts
                    ],
                    'inference_latency_ms': inference_duration_ms,
                    'total_latency_ms': total_duration_ms,
                    'timestamp': tx.created_at.isoformat(),
                }

        except Exception as exc:
            total_duration_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            logger.error("Processing failed for event %s: %s", event_id, exc, exc_info=True)
            return {
                'success': False,
                'status': 'PROCESSING_FAILED',
                'event_id': event_id,
                'transaction_id': transaction_id,
                'errors': [str(exc)],
                'latency_ms': total_duration_ms,
                'timestamp': timezone.now().isoformat(),
            }

    @classmethod
    def _build_idempotent_response(cls, tx: Transaction) -> Dict[str, Any]:
        """Constructs safe response for an already-processed event."""
        pred = tx.predictions.first()
        alerts = tx.alerts.all()

        return {
            'success': True,
            'status': 'COMPLETED',
            'idempotent': True,
            'event_id': tx.event_id,
            'transaction_id': tx.transaction_id,
            'source': tx.source,
            'scenario': tx.scenario_tag,
            'transaction_type': tx.transaction_type,
            'amount': float(tx.amount),
            'channel': tx.channel,
            'name_orig': tx.name_orig,
            'name_dest': tx.name_dest,
            'decision_state': pred.decision_state if pred else 'UNASSESSED',
            'risk_score': pred.risk_score if pred else 0.0,
            'risk_tier': pred.risk_tier if pred else 'LOW',
            'fraud_probability': pred.fraud_probability if pred else None,
            'anomaly_score': pred.anomaly_score if pred else None,
            'alerts_count': alerts.count(),
            'alerts': [
                {
                    'alert_id': a.alert_id,
                    'title': a.title,
                    'severity': a.severity,
                    'rule_name': a.rule_name
                } for a in alerts
            ],
            'latency_ms': tx.latency_ms,
            'timestamp': tx.created_at.isoformat(),
        }
