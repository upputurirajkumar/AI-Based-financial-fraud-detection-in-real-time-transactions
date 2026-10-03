import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Standardized, leakage-free preprocessing pipeline for financial transactions."""

    CANONICAL_FEATURES = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'isFlaggedFraud'
    ]

    REQUIRED_TRANSACTION_COLS = [
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]

    @classmethod
    def extract_features_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts and verifies canonical features from a raw DataFrame."""
        df_clean = df.copy()

        # Handle isFlaggedFraud if absent
        if 'isFlaggedFraud' not in df_clean.columns:
            df_clean['isFlaggedFraud'] = 0

        # Ensure all canonical columns exist
        missing = [col for col in cls.CANONICAL_FEATURES if col not in df_clean.columns]
        if missing:
            raise ValueError(f"Input data is missing required feature columns: {missing}")

        features_df = df_clean[cls.CANONICAL_FEATURES].copy()

        # Cast to numeric, coerce errors to 0.0
        for col in cls.CANONICAL_FEATURES:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)

        return features_df

    @classmethod
    def prepare_single_transaction(cls, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepares a single transaction dictionary for model scoring."""
        amount = float(data.get('amount', 0.0))
        old_orig = float(data.get('oldbalanceOrg', 0.0))
        new_orig = float(data.get('newbalanceOrig', 0.0))
        old_dest = float(data.get('oldbalanceDest', 0.0))
        new_dest = float(data.get('newbalanceDest', 0.0))
        is_flagged = int(data.get('isFlaggedFraud', 0))

        row = {
            'amount': amount,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'isFlaggedFraud': is_flagged
        }
        return pd.DataFrame([row], columns=cls.CANONICAL_FEATURES)
