import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
import pandas as pd
from django.db import transaction as db_transaction
from django.contrib.auth.models import User

from apps.transactions.models import Transaction, DatasetRecord
from apps.fraud.models import Prediction, FraudAlert
from apps.core.models import AuditLog
from services.fraud_engine import FraudEngine
from services.alert_service import AlertService
from services.data_quality_service import DataQualityService

logger = logging.getLogger(__name__)


class BatchIngestionService:
    """
    Orchestrates end-to-end dataset ingestion:
    Schema audit -> Transaction creation -> ML prediction -> Alert generation -> Progress reporting.
    """

    @classmethod
    def process_dataset(
        cls,
        df: pd.DataFrame,
        dataset_record: Optional[DatasetRecord] = None,
        actor: Optional[User] = None,
        model_key: str = FraudEngine.DEFAULT_MODEL_KEY
    ) -> Dict[str, Any]:
        """
        Processes an entire DataFrame batch, inserting valid transactions and recording predictions/alerts.
        """
        total_rows = len(df)
        accepted_rows = 0
        rejected_rows = 0
        fraud_predictions = 0
        suspicious_predictions = 0
        legit_predictions = 0
        alerts_created = 0
        errors = []

        # 1. Automated Data Quality Audit
        quality_report = DataQualityService.audit_dataset(df)
        if dataset_record:
            dataset_record.data_quality_summary = quality_report
            dataset_record.total_records = total_rows
            dataset_record.save()

        # 2. Row-by-Row Ingestion with Error Isolation
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            val_errors = FraudEngine.validate_transaction(row_dict)

            if val_errors:
                rejected_rows += 1
                errors.append({'row_index': int(idx), 'errors': val_errors})
                continue

            try:
                with db_transaction.atomic():
                    # Create Transaction
                    tx = Transaction.objects.create(
                        dataset=dataset_record,
                        step=int(row_dict.get('step', 1)),
                        transaction_type=str(row_dict.get('type', 'PAYMENT')).upper(),
                        amount=Decimal(str(row_dict.get('amount', 0.0))),
                        channel=str(row_dict.get('channel', 'ONLINE')).upper(),
                        name_orig=str(row_dict.get('nameOrig', f'C{idx:08d}')),
                        old_balance_orig=Decimal(str(row_dict.get('oldbalanceOrg', 0.0))),
                        new_balance_orig=Decimal(str(row_dict.get('newbalanceOrig', 0.0))),
                        name_dest=str(row_dict.get('nameDest', f'M{idx:08d}')),
                        old_balance_dest=Decimal(str(row_dict.get('oldbalanceDest', 0.0))),
                        new_balance_dest=Decimal(str(row_dict.get('newbalanceDest', 0.0))),
                        is_fraud_flag=bool(row_dict.get('isFraud', 0)),
                        is_flagged_fraud=bool(row_dict.get('isFlaggedFraud', 0))
                    )

                    # Model Inference via FraudEngine
                    analysis = FraudEngine.analyze_transaction(tx.to_inference_dict(), model_key=model_key)

                    # Save Prediction Record
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

                    # Generate Alerts
                    alerts = AlertService.evaluate_and_generate_alerts(pred, analysis, transaction_record=tx)
                    alerts_created += len(alerts)

                    # Tally
                    accepted_rows += 1
                    if analysis['decision_state'] == FraudEngine.DECISION_FRAUD:
                        fraud_predictions += 1
                    elif analysis['decision_state'] == FraudEngine.DECISION_SUSPICIOUS:
                        suspicious_predictions += 1
                    else:
                        legit_predictions += 1

            except Exception as exc:
                rejected_rows += 1
                errors.append({'row_index': int(idx), 'errors': [str(exc)]})
                logger.error(f"Row {idx} ingestion error: {exc}")

        # Update dataset record status
        if dataset_record:
            dataset_record.processed_records = accepted_rows
            dataset_record.failed_records = rejected_rows
            dataset_record.is_processed = True
            dataset_record.save()

        # Audit Log
        AuditLog.objects.create(
            user=actor,
            action='BATCH_PREDICTION',
            details=f"Batch processed {total_rows} rows: {accepted_rows} accepted, {fraud_predictions} fraud, {alerts_created} alerts generated."
        )

        return {
            'total_rows': total_rows,
            'accepted_rows': accepted_rows,
            'rejected_rows': rejected_rows,
            'fraud_predictions': fraud_predictions,
            'suspicious_predictions': suspicious_predictions,
            'legitimate_predictions': legit_predictions,
            'alerts_generated': alerts_created,
            'data_quality_report': quality_report,
            'errors': errors[:50]  # Return sample of errors
        }
