import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from django.db import transaction as db_transaction
from django.core.paginator import Paginator
from django.contrib.auth.models import User

from apps.transactions.models import Transaction, DatasetRecord
from apps.fraud.models import Prediction, FraudAlert, InvestigationCase
from apps.core.models import AuditLog
from services.fraud_engine import FraudEngine
from services.alert_service import AlertService

logger = logging.getLogger(__name__)


class TransactionService:
    """
    Core transaction management, search, detail assembly, and ingestion service.
    """

    @classmethod
    def ingest_and_evaluate_transaction(
        cls,
        data: Dict[str, Any],
        actor: Optional[User] = None,
        dataset: Optional[DatasetRecord] = None,
        channel: str = 'ONLINE'
    ) -> Dict[str, Any]:
        """
        Persists a transaction, executes full FraudEngine inference,
        saves the versioned prediction record, and generates deduplicated alerts.
        """
        # Validate data with FraudEngine first
        validation_errors = FraudEngine.validate_transaction(data)
        if validation_errors:
            return {
                'success': False,
                'errors': validation_errors,
                'transaction': None,
                'prediction': None
            }

        with db_transaction.atomic():
            # 1. Create Transaction in Database
            tx = Transaction.objects.create(
                dataset=dataset,
                step=int(data.get('step', 1)),
                transaction_type=str(data.get('type', 'PAYMENT')).upper(),
                amount=Decimal(str(data.get('amount', 0.0))),
                currency=str(data.get('currency', 'USD')),
                channel=channel,
                name_orig=str(data.get('nameOrig', 'C000000000')),
                old_balance_orig=Decimal(str(data.get('oldbalanceOrg', 0.0))),
                new_balance_orig=Decimal(str(data.get('newbalanceOrig', 0.0))),
                name_dest=str(data.get('nameDest', 'M000000000')),
                old_balance_dest=Decimal(str(data.get('oldbalanceDest', 0.0))),
                new_balance_dest=Decimal(str(data.get('newbalanceDest', 0.0))),
                is_fraud_flag=bool(data.get('isFraud', 0)),
                is_flagged_fraud=bool(data.get('isFlaggedFraud', 0)),
                raw_metadata=data.get('metadata', {})
            )

            # 2. Run Fraud Engine
            inference_payload = tx.to_inference_dict()
            analysis = FraudEngine.analyze_transaction(inference_payload)

            # 3. Persist Prediction Record (Immutable versioned snapshot)
            pred = Prediction.objects.create(
                transaction=tx,
                model_name=analysis['model_metadata'].get('model_name', 'Random Forest Classifier'),
                model_version=analysis['model_metadata'].get('model_version', '1.0.0'),
                decision_threshold=analysis['threshold'],
                fraud_probability=analysis['fraud_probability'],
                risk_score=analysis['risk_score'],
                risk_tier=analysis['risk_level'],
                decision_state=analysis['decision_state'],
                is_fraud=analysis['decision_state'] == FraudEngine.DECISION_FRAUD,
                anomaly_score=analysis['anomaly_info'].get('anomaly_score'),
                is_anomalous=analysis['anomaly_info'].get('is_anomalous', False),
                explanation="\n".join([f['description'] for f in analysis['explanation'].get('risk_factors', [])]),
                raw_explanation_json=analysis['explanation'],
                evaluated_by=actor
            )

            # 4. Generate Deduplicated Alerts if risk criteria are met
            alerts = AlertService.evaluate_and_generate_alerts(pred, analysis, transaction_record=tx)

            # 5. Record Audit Log
            AuditLog.objects.create(
                user=actor,
                action='TRANSACTION_INGEST',
                details=f"Ingested {tx.transaction_id} -> {pred.decision_state} (Risk: {pred.risk_score})"
            )

        return {
            'success': True,
            'transaction_id': tx.transaction_id,
            'prediction_id': pred.prediction_id,
            'analysis': analysis,
            'alerts_generated': len(alerts)
        }

    @classmethod
    def search_transactions(
        cls,
        search_query: Optional[str] = None,
        transaction_type: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        risk_tier: Optional[str] = None,
        decision_state: Optional[str] = None,
        channel: Optional[str] = None,
        has_alerts: Optional[bool] = None,
        date_from=None,
        date_to=None,
        page: int = 1,
        page_size: int = 25
    ) -> Dict[str, Any]:
        """
        Database-side paginated transaction search with multi-parameter filtering.
        """
        qs = Transaction.objects.prefetch_related('predictions', 'alerts').all()

        if search_query:
            from django.db.models import Q
            qs = qs.filter(
                Q(transaction_id__icontains=search_query) |
                Q(name_orig__icontains=search_query) |
                Q(name_dest__icontains=search_query)
            )

        if transaction_type:
            qs = qs.filter(transaction_type=transaction_type.upper())

        if min_amount is not None:
            qs = qs.filter(amount__gte=Decimal(str(min_amount)))

        if max_amount is not None:
            qs = qs.filter(amount__lte=Decimal(str(max_amount)))

        if channel:
            qs = qs.filter(channel=channel.upper())

        if risk_tier:
            qs = qs.filter(predictions__risk_tier=risk_tier.upper())

        if decision_state:
            qs = qs.filter(predictions__decision_state=decision_state.upper())

        if has_alerts is True:
            qs = qs.filter(alerts__isnull=False).distinct()
        elif has_alerts is False:
            qs = qs.filter(alerts__isnull=True)

        if date_from:
            qs = qs.filter(created_at__gte=date_from)
        if date_to:
            qs = qs.filter(created_at__lte=date_to)

        paginator = Paginator(qs, page_size)
        current_page = paginator.get_page(page)

        records = []
        for t in current_page:
            latest_pred = t.predictions.first()
            records.append({
                'id': t.id,
                'transaction_id': t.transaction_id,
                'step': t.step,
                'transaction_type': t.transaction_type,
                'amount': float(t.amount),
                'currency': t.currency,
                'channel': t.channel,
                'name_orig': t.name_orig,
                'name_dest': t.name_dest,
                'old_balance_orig': float(t.old_balance_orig),
                'new_balance_orig': float(t.new_balance_orig),
                'old_balance_dest': float(t.old_balance_dest),
                'new_balance_dest': float(t.new_balance_dest),
                'created_at': t.created_at.isoformat(),
                'latest_prediction': {
                    'prediction_id': latest_pred.prediction_id if latest_pred else None,
                    'decision_state': latest_pred.decision_state if latest_pred else 'UNASSESSED',
                    'risk_score': latest_pred.risk_score if latest_pred else None,
                    'risk_tier': latest_pred.risk_tier if latest_pred else None,
                    'fraud_probability': latest_pred.fraud_probability if latest_pred else None,
                    'model_version': latest_pred.model_version if latest_pred else None,
                } if latest_pred else None,
                'alert_count': t.alerts.count()
            })

        return {
            'total_count': paginator.count,
            'total_pages': paginator.num_pages,
            'current_page': current_page.number,
            'page_size': page_size,
            'transactions': records
        }

    @classmethod
    def get_transaction_detail(cls, tx_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Assembles comprehensive transaction intelligence object including historical predictions,
        associated alerts, investigation cases, and audit trail.
        """
        tx = Transaction.objects.filter(transaction_id=tx_identifier).first()
        if not tx:
            # Fallback to integer primary key
            if str(tx_identifier).isdigit():
                tx = Transaction.objects.filter(id=int(tx_identifier)).first()
            if not tx:
                return None

        # 1. Historical Predictions (Preserving multi-model version history)
        predictions = []
        for p in tx.predictions.all().order_by('-created_at'):
            predictions.append({
                'prediction_id': p.prediction_id,
                'model_name': p.model_name,
                'model_version': p.model_version,
                'decision_state': p.decision_state,
                'risk_score': p.risk_score,
                'risk_tier': p.risk_tier,
                'fraud_probability': p.fraud_probability,
                'decision_threshold': p.decision_threshold,
                'anomaly_score': p.anomaly_score,
                'is_anomalous': p.is_anomalous,
                'explanation': p.raw_explanation_json,
                'created_at': p.created_at.isoformat()
            })

        # 2. Associated Alerts
        alerts = []
        for a in tx.alerts.all().order_by('-created_at'):
            alerts.append({
                'alert_id': a.alert_id,
                'title': a.title,
                'severity': a.severity,
                'status': a.status,
                'rule_name': a.rule_name,
                'assigned_to': a.assigned_to.username if a.assigned_to else None,
                'created_at': a.created_at.isoformat()
            })

        # 3. Related Cases
        cases = []
        for c in tx.investigations.all().order_by('-created_at'):
            cases.append({
                'case_id': c.case_id,
                'title': c.title,
                'status': c.status,
                'priority': c.priority,
                'resolution': c.resolution,
                'assigned_analyst': c.assigned_analyst.username if c.assigned_analyst else None,
                'opened_at': c.opened_at.isoformat()
            })

        return {
            'transaction': {
                'id': tx.id,
                'transaction_id': tx.transaction_id,
                'step': tx.step,
                'type': tx.transaction_type,
                'amount': float(tx.amount),
                'currency': tx.currency,
                'channel': tx.channel,
                'name_orig': tx.name_orig,
                'old_balance_orig': float(tx.old_balance_orig),
                'new_balance_orig': float(tx.new_balance_orig),
                'name_dest': tx.name_dest,
                'old_balance_dest': float(tx.old_balance_dest),
                'new_balance_dest': float(tx.new_balance_dest),
                'created_at': tx.created_at.isoformat(),
            },
            'predictions': predictions,
            'alerts': alerts,
            'investigations': cases,
            'audit_history': list(AuditLog.objects.filter(details__icontains=tx.transaction_id).values('action', 'created_at', 'details')[:10])
        }
