import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .model_service import ModelService, ModelLoadingError, ModelIncompatibilityError
from .risk_service import RiskService
from .explainability_service import ExplainabilityService

logger = logging.getLogger(__name__)


class FraudEngine:
    """
    Core AI Fraud Detection & Risk Intelligence Engine.
    Orchestrates input validation, ML classification, unsupervised anomaly scoring,
    risk intelligence, explainable AI factor attribution, and standardized output contracts.
    """

    DEFAULT_MODEL_KEY = 'rfc_production'
    ANOMALY_MODEL_KEY = 'autoencoder_experimental'

    REQUIRED_NUMERIC_FIELDS = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
    ]

    DECISION_LEGITIMATE = 'LEGITIMATE'
    DECISION_SUSPICIOUS = 'SUSPICIOUS'
    DECISION_FRAUD = 'FRAUD'

    @classmethod
    def validate_transaction(cls, transaction: Dict[str, Any]) -> List[str]:
        """Validates that a transaction has all required numeric fields with valid values."""
        errors = []
        for field in cls.REQUIRED_NUMERIC_FIELDS:
            if field not in transaction or transaction[field] is None or pd.isna(transaction[field]):
                errors.append(f"Missing or NaN required field: '{field}'")
            else:
                try:
                    val = float(transaction[field])
                    if np.isnan(val) or np.isinf(val):
                        errors.append(f"Field '{field}' cannot be NaN or Infinite")
                    elif val < 0:
                        errors.append(f"Invalid negative financial value in '{field}': {val}")
                except (ValueError, TypeError):
                    errors.append(f"Field '{field}' must be numeric, received: {transaction[field]}")
        return errors

    @classmethod
    def analyze_transaction(
        cls,
        transaction: Dict[str, Any],
        model_key: str = DEFAULT_MODEL_KEY,
        custom_threshold: Optional[float] = None,
        include_anomaly: bool = True,
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete fraud detection and risk intelligence workflow for a single transaction.
        """
        now_ts = datetime.utcnow().isoformat()
        tx_id = str(transaction.get('transaction_id') or transaction.get('nameOrig') or f"tx-{int(datetime.utcnow().timestamp()*1000)}")

        # 1. Input Validation
        validation_errors = cls.validate_transaction(transaction)
        if validation_errors:
            return {
                'status': 'INVALID_INPUT',
                'transaction_id': tx_id,
                'prediction': 'ERROR',
                'decision_state': 'ERROR',
                'errors': validation_errors,
                'fraud_probability': None,
                'risk_score': None,
                'risk_level': 'UNKNOWN',
                'threshold': None,
                'model_metadata': {'error': 'Validation failure prevented model execution'},
                'anomaly_info': {'status': 'SKIPPED'},
                'explanation': {'risk_factors': []},
                'timestamp': now_ts
            }

        # 2. Canonical DataFrame Row Preparation
        row_df = pd.DataFrame([{
            'step': int(transaction.get('step', 1)),
            'type': str(transaction.get('type', 'PAYMENT')).upper(),
            'amount': float(transaction.get('amount', 0.0)),
            'nameOrig': str(transaction.get('nameOrig', 'C000000000')),
            'oldbalanceOrg': float(transaction.get('oldbalanceOrg', 0.0)),
            'newbalanceOrig': float(transaction.get('newbalanceOrig', 0.0)),
            'nameDest': str(transaction.get('nameDest', 'M000000000')),
            'oldbalanceDest': float(transaction.get('oldbalanceDest', 0.0)),
            'newbalanceDest': float(transaction.get('newbalanceDest', 0.0)),
        }])

        # 3. Model Metadata & Threshold
        metadata = ModelService.get_model_metadata(model_key)
        operating_threshold = custom_threshold if custom_threshold is not None else metadata.get('decision_threshold', 0.23)
        model_name = metadata.get('model_name', 'Random Forest Classifier')
        model_version = metadata.get('model_version', '1.0.0')

        # 4. Supervised ML Prediction
        fraud_prob = None
        model_status = 'AVAILABLE'
        try:
            pipeline = ModelService.load_model(model_key)
            if hasattr(pipeline, 'predict_proba'):
                probs = pipeline.predict_proba(row_df)
                fraud_prob = round(float(probs[0][1]), 4)
            elif hasattr(pipeline, 'score_samples'):
                raw_score = pipeline.score_samples(row_df)[0]
                fraud_prob = round(float(raw_score / 100.0), 4)
        except (ModelLoadingError, ModelIncompatibilityError, Exception) as exc:
            logger.warning(f"Supervised ML model '{model_key}' error: {exc}")
            model_status = f"UNAVAILABLE ({str(exc)})"

        # 5. Optional Unsupervised Anomaly Detection
        anomaly_info = {'status': 'DISABLED', 'anomaly_score': None, 'is_anomalous': False}
        anomaly_score = None
        if include_anomaly:
            try:
                anomaly_pipeline = ModelService.load_model(cls.ANOMALY_MODEL_KEY)
                if hasattr(anomaly_pipeline, 'score_samples'):
                    score = float(anomaly_pipeline.score_samples(row_df)[0])
                    is_anom = bool(score >= 70.0)
                    anomaly_score = score
                    anomaly_info = {
                        'status': 'AVAILABLE',
                        'anomaly_score': round(score, 2),
                        'anomaly_threshold': 70.0,
                        'is_anomalous': is_anom,
                        'method': 'Isolation Reconstruction Anomaly Detection'
                    }
            except Exception as anom_exc:
                logger.debug(f"Anomaly model evaluation skipped: {anom_exc}")
                anomaly_info = {'status': 'UNAVAILABLE', 'anomaly_score': None, 'is_anomalous': False}

        # 6. Risk Intelligence Layer
        risk_assessment = RiskService.calculate_risk_score(
            transaction,
            model_prob=fraud_prob,
            anomaly_score=anomaly_score
        )
        risk_score = risk_assessment['risk_score']
        risk_level = risk_assessment['risk_tier']

        # 7. Tri-State Fraud Decision Logic
        # Model classification vs risk score separation
        model_classified_fraud = bool(fraud_prob >= operating_threshold) if fraud_prob is not None else False

        if model_classified_fraud or risk_level == RiskService.TIER_CRITICAL:
            decision_state = cls.DECISION_FRAUD
        elif risk_level in (RiskService.TIER_MEDIUM, RiskService.TIER_HIGH) or anomaly_info.get('is_anomalous'):
            decision_state = cls.DECISION_SUSPICIOUS
        else:
            decision_state = cls.DECISION_LEGITIMATE

        # 8. Explainable AI Factor Attribution
        explanation = {'status': 'SKIPPED', 'risk_factors': []}
        if include_explanation:
            explanation = ExplainabilityService.explain_transaction(
                transaction,
                fraud_probability=fraud_prob,
                model_key=model_key
            )

        # 9. Investigation Intelligence Metadata Object (For Phase 5 UI)
        investigation_data = {
            'step': int(transaction.get('step', 1)),
            'type': str(transaction.get('type', 'PAYMENT')).upper(),
            'amount': float(transaction.get('amount', 0.0)),
            'nameOrig': str(transaction.get('nameOrig', 'C000000000')),
            'oldbalanceOrg': float(transaction.get('oldbalanceOrg', 0.0)),
            'newbalanceOrig': float(transaction.get('newbalanceOrig', 0.0)),
            'nameDest': str(transaction.get('nameDest', 'M000000000')),
            'oldbalanceDest': float(transaction.get('oldbalanceDest', 0.0)),
            'newbalanceDest': float(transaction.get('newbalanceDest', 0.0)),
            'orig_balance_discrepancy': round(abs((float(transaction.get('oldbalanceOrg', 0.0)) - float(transaction.get('amount', 0.0))) - float(transaction.get('newbalanceOrig', 0.0))), 2),
            'dest_balance_discrepancy': round(abs((float(transaction.get('oldbalanceDest', 0.0)) + float(transaction.get('amount', 0.0))) - float(transaction.get('newbalanceDest', 0.0))), 2),
            'orig_drained': bool(float(transaction.get('oldbalanceOrg', 0.0)) > 0 and float(transaction.get('newbalanceOrig', 0.0)) == 0),
        }

        # 10. Standardized Prediction Contract
        return {
            'status': 'SUCCESS',
            'transaction_id': tx_id,
            'prediction': decision_state,
            'decision_state': decision_state,
            'fraud_probability': fraud_prob,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'threshold': round(float(operating_threshold), 3),
            'model_metadata': {
                'model_name': model_name,
                'model_version': model_version,
                'status': model_status,
                'framework': metadata.get('framework', 'scikit-learn'),
                'feature_version': '2.0.0'
            },
            'anomaly_info': anomaly_info,
            'explanation': explanation,
            'investigation_metadata': investigation_data,
            'timestamp': now_ts
        }

    @classmethod
    def analyze_batch(
        cls,
        df: pd.DataFrame,
        model_key: str = DEFAULT_MODEL_KEY,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes batch inference across a DataFrame, collecting metrics and preserving all rows.
        """
        total_rows = len(df)
        valid_rows = 0
        invalid_rows = 0
        fraud_count = 0
        suspicious_count = 0
        legit_count = 0
        records = []
        processing_errors = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            analysis = cls.analyze_transaction(
                row_dict,
                model_key=model_key,
                custom_threshold=custom_threshold,
                include_anomaly=True,
                include_explanation=True
            )

            if analysis['status'] == 'INVALID_INPUT':
                invalid_rows += 1
                processing_errors.append({'row_index': int(idx), 'errors': analysis['errors']})
            else:
                valid_rows += 1
                state = analysis['decision_state']
                if state == cls.DECISION_FRAUD:
                    fraud_count += 1
                elif state == cls.DECISION_SUSPICIOUS:
                    suspicious_count += 1
                else:
                    legit_count += 1

            records.append({**row_dict, **analysis})

        return {
            'total_rows': total_rows,
            'valid_rows': valid_rows,
            'invalid_rows': invalid_rows,
            'fraud_count': fraud_count,
            'suspicious_count': suspicious_count,
            'legitimate_count': legit_count,
            'fraud_rate': round(fraud_count / max(valid_rows, 1) * 100.0, 2),
            'processing_errors': processing_errors,
            'records': records
        }
