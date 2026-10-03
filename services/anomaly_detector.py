import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .ml_pipeline import create_preprocessor

logger = logging.getLogger(__name__)


class TransactionAnomalyDetector:
    """
    Unsupervised anomaly detection engine trained strictly on legitimate financial transactions.
    Computes reconstruction/isolation anomaly scores to detect novel fraud patterns.
    """

    def __init__(self, contamination: float = 0.025, random_state: int = 42):
        self.preprocessor = create_preprocessor()
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1
        )
        self.is_fitted = False
        self.threshold = 0.0  # Decision function offset

    def fit(self, X_legitimate: pd.DataFrame):
        """Fits preprocessor and anomaly detector strictly on normal/legitimate transactions."""
        logger.info(f"Fitting anomaly detector on {len(X_legitimate)} legitimate transactions...")
        X_trans = self.preprocessor.fit_transform(X_legitimate)
        self.detector.fit(X_trans)
        self.is_fitted = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized anomaly score in range [0, 100].
        Higher score indicates higher anomaly likelihood.
        """
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before scoring samples.")

        X_trans = self.preprocessor.transform(X)
        # decision_function yields negative values for outliers, positive for inliers
        raw_scores = -self.detector.decision_function(X_trans)
        # Scale to [0, 100] approximately
        # Typically raw scores range roughly from -0.2 to +0.3
        anomaly_scores = np.clip((raw_scores + 0.2) / 0.5 * 100.0, 0.0, 100.0)
        return np.round(anomaly_scores, 2)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1 for anomalies, 0 for normal transactions."""
        if not self.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before predicting.")

        X_trans = self.preprocessor.transform(X)
        preds = self.detector.predict(X_trans)
        # IsolationForest returns -1 for outliers, +1 for inliers
        return (preds == -1).astype(int)
