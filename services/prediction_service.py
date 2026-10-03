import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from .fraud_engine import FraudEngine

logger = logging.getLogger(__name__)


class PredictionService:
    """
    High-level facade exposing unified inference methods backed by the modern FraudEngine.
    Maintains backward compatibility with all Phase 1-3 call sites while enabling Phase 4 capabilities.
    """

    DEFAULT_MODEL_KEY = FraudEngine.DEFAULT_MODEL_KEY

    @classmethod
    def validate_transaction_record(cls, record: Dict[str, Any]) -> List[str]:
        return FraudEngine.validate_transaction(record)

    @classmethod
    def predict_single(
        cls,
        transaction: Dict[str, Any],
        model_key: str = DEFAULT_MODEL_KEY,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs full analysis on a single transaction."""
        analysis = FraudEngine.analyze_transaction(
            transaction,
            model_key=model_key,
            custom_threshold=custom_threshold
        )
        # Adapt for backward compatibility expectations
        return {
            'status': analysis['status'],
            'prediction': analysis['prediction'],
            'is_fraud': analysis['decision_state'] == FraudEngine.DECISION_FRAUD,
            'decision_state': analysis['decision_state'],
            'fraud_probability': analysis['fraud_probability'],
            'decision_threshold': analysis['threshold'],
            'risk_score': analysis['risk_score'],
            'risk_tier': analysis['risk_level'],
            'risk_triggers': [f['description'] for f in analysis['explanation'].get('risk_factors', [])],
            'model_name': analysis['model_metadata'].get('model_name'),
            'model_version': analysis['model_metadata'].get('model_version'),
            'anomaly_info': analysis['anomaly_info'],
            'explanation': analysis['explanation'],
            'investigation_metadata': analysis.get('investigation_metadata', {}),
            'timestamp': analysis['timestamp']
        }

    @classmethod
    def predict_batch(
        cls,
        df: pd.DataFrame,
        model_key: str = DEFAULT_MODEL_KEY,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs batch analysis using FraudEngine."""
        batch_out = FraudEngine.analyze_batch(
            df,
            model_key=model_key,
            custom_threshold=custom_threshold
        )
        return {
            'total_rows': batch_out['total_rows'],
            'valid_rows': batch_out['valid_rows'],
            'invalid_rows': batch_out['invalid_rows'],
            'fraud_predictions': batch_out['fraud_count'],
            'suspicious_predictions': batch_out['suspicious_count'],
            'legitimate_predictions': batch_out['legitimate_count'],
            'fraud_rate': batch_out['fraud_rate'],
            'processing_errors': batch_out['processing_errors'],
            'records': batch_out['records']
        }

    @classmethod
    def predict_dataframe(cls, df: pd.DataFrame, model_key: str = DEFAULT_MODEL_KEY) -> List[Dict[str, Any]]:
        batch_out = cls.predict_batch(df, model_key=model_key)
        return batch_out['records']
