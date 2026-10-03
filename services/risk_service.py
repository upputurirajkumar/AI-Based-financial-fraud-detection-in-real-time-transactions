from typing import Dict, Any, Optional
import numpy as np


class RiskService:
    """
    Deterministic Risk Intelligence Layer converting statistical ML predictions,
    unsupervised anomaly scores, and domain financial indicators into calibrated 0–100 risk scores.
    """

    # Configurable Risk Tiers
    TIER_LOW = 'LOW'
    TIER_MEDIUM = 'MEDIUM'
    TIER_HIGH = 'HIGH'
    TIER_CRITICAL = 'CRITICAL'

    # Risk Band Boundaries
    THRESHOLD_MEDIUM = 30.0
    THRESHOLD_HIGH = 60.0
    THRESHOLD_CRITICAL = 85.0

    @classmethod
    def get_tier_for_score(cls, score: float) -> str:
        """Maps a numeric risk score (0-100) into a discrete risk tier."""
        if score >= cls.THRESHOLD_CRITICAL:
            return cls.TIER_CRITICAL
        elif score >= cls.THRESHOLD_HIGH:
            return cls.TIER_HIGH
        elif score >= cls.THRESHOLD_MEDIUM:
            return cls.TIER_MEDIUM
        else:
            return cls.TIER_LOW

    @classmethod
    def calculate_risk_score(
        cls,
        transaction: Dict[str, Any],
        model_prob: Optional[float] = None,
        anomaly_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates deterministic risk score (0-100), risk tier, and triggered risk rules.
        """
        # Safely extract and convert numeric fields
        try:
            amount = float(transaction.get('amount', 0.0))
            old_orig = float(transaction.get('oldbalanceOrg', 0.0))
            new_orig = float(transaction.get('newbalanceOrig', 0.0))
            old_dest = float(transaction.get('oldbalanceDest', 0.0))
            new_dest = float(transaction.get('newbalanceDest', 0.0))
        except (ValueError, TypeError):
            amount = 0.0
            old_orig = 0.0
            new_orig = 0.0
            old_dest = 0.0
            new_dest = 0.0

        tx_type = str(transaction.get('type', '')).upper()

        score = 0.0
        triggers = []

        # 1. Statistical ML Model Component (0 to 60 points)
        if model_prob is not None:
            # Bound probability safely between 0.0 and 1.0
            safe_prob = min(max(float(model_prob), 0.0), 1.0)
            model_component = safe_prob * 60.0
            score += model_component

            if safe_prob >= 0.70:
                triggers.append(f"ML Classifier predicted high fraud probability ({safe_prob:.1%})")
            elif safe_prob >= 0.23:
                triggers.append(f"ML Classifier probability ({safe_prob:.1%}) exceeds decision threshold (23.0%)")

        # 2. Origin Balance Discrepancy & Account Drainage (0 to 20 points)
        orig_discrepancy = abs((old_orig - amount) - new_orig)
        if orig_discrepancy > 1.0:
            score += 15.0
            triggers.append(f"Origin ledger discrepancy detected (${orig_discrepancy:,.2f})")

        if old_orig > 0 and new_orig == 0 and abs(old_orig - amount) < 1.0:
            score += 10.0
            triggers.append("Origin account completely emptied by transaction")

        # 3. High-Value Transfer & Mule Destination Indicator (0 to 15 points)
        if tx_type in ('TRANSFER', 'CASH_OUT'):
            if amount >= 200000.0:
                score += 10.0
                triggers.append("High-value transfer exceeding $200,000 threshold")
            elif amount >= 50000.0:
                score += 5.0

            if old_dest == 0.0 and new_dest == 0.0 and amount > 5000.0:
                score += 10.0
                triggers.append("Destination zero balance persistence (layering indicator)")

        # 4. Unsupervised Anomaly Component (0 to 10 points)
        if anomaly_score is not None:
            safe_anomaly = min(max(float(anomaly_score), 0.0), 100.0)
            if safe_anomaly >= 70.0:
                anomaly_component = (safe_anomaly / 100.0) * 10.0
                score += anomaly_component
                triggers.append(f"Unsupervised anomaly detector flagged abnormal transaction profile ({safe_anomaly:.1f}/100)")

        # Cap score deterministically in [0.0, 100.0]
        final_score = round(min(max(score, 0.0), 100.0), 1)
        tier = cls.get_tier_for_score(final_score)
        is_fraud = bool(final_score >= cls.THRESHOLD_HIGH)

        return {
            'risk_score': final_score,
            'risk_tier': tier,
            'is_fraud': is_fraud,
            'triggers': triggers
        }
