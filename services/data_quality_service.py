import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataQualityService:
    """Performs rigorous automated data quality audits, schema checks, and leakage detection."""

    EXPECTED_COLUMNS = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
    ]

    NUMERICAL_COLUMNS = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest'
    ]

    CATEGORICAL_COLUMNS = ['type']
    IDENTIFIER_COLUMNS = ['nameOrig', 'nameDest']
    TARGET_COLUMN = 'isFraud'
    AUXILIARY_COLUMNS = ['isFlaggedFraud']

    @classmethod
    def audit_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates comprehensive data quality audit report for tabular financial datasets."""
        total_rows = len(df)
        total_cols = len(df.columns)

        # 1. Missing Values
        missing_counts = df.isnull().sum().to_dict()
        missing_pcts = {k: round(v / total_rows * 100.0, 2) for k, v in missing_counts.items()}
        total_missing = int(sum(missing_counts.values()))

        # 2. Duplicate Transactions
        duplicate_rows = int(df.duplicated().sum())

        # 3. Target Distribution (if present)
        target_stats = {}
        if cls.TARGET_COLUMN in df.columns:
            vc = df[cls.TARGET_COLUMN].value_counts().to_dict()
            legit_count = int(vc.get(0, 0))
            fraud_count = int(vc.get(1, 0))
            fraud_pct = round(fraud_count / total_rows * 100.0, 3) if total_rows > 0 else 0.0
            imbalance_ratio = round(legit_count / max(fraud_count, 1), 1)

            target_stats = {
                'target_name': cls.TARGET_COLUMN,
                'mapping': {'0': 'Legitimate', '1': 'Fraudulent'},
                'legitimate_count': legit_count,
                'fraud_count': fraud_count,
                'fraud_percentage': fraud_pct,
                'imbalance_ratio': f"{imbalance_ratio}:1"
            }

        # 4. Numerical Column Statistics & Outliers (IQR)
        num_stats = {}
        invalid_values = {}
        for col in cls.NUMERICAL_COLUMNS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())

                # Financial validity checks (negative balances or amounts are invalid)
                negative_count = int((series < 0).sum())
                nan_inf_count = int(series.isna().sum() + np.isinf(series).sum())

                num_stats[col] = {
                    'min': round(float(series.min()), 2),
                    'max': round(float(series.max()), 2),
                    'mean': round(float(series.mean()), 2),
                    'std': round(float(series.std()), 2),
                    'median': round(float(series.median()), 2),
                    'outliers_count': outliers_count,
                    'outliers_pct': round(outliers_count / max(total_rows, 1) * 100.0, 2)
                }

                if negative_count > 0 or nan_inf_count > 0:
                    invalid_values[col] = {
                        'negative_count': negative_count,
                        'nan_inf_count': nan_inf_count
                    }

        # 5. Categorical Distributions
        cat_stats = {}
        for col in cls.CATEGORICAL_COLUMNS:
            if col in df.columns:
                cat_stats[col] = df[col].value_counts().to_dict()

        # 6. Leakage & Suspicious Patterns Warning
        leakage_warnings = []
        if 'isFlaggedFraud' in df.columns:
            leakage_warnings.append(
                "isFlaggedFraud is a heuristic downstream alert rule, not a primary transaction attribute. "
                "Must not be used as sole target or direct label indicator."
            )

        if total_missing > 0:
            leakage_warnings.append(f"Dataset contains {total_missing} missing cells requiring imputation.")

        return {
            'shape': {'rows': total_rows, 'columns': total_cols},
            'columns': list(df.columns),
            'missing_summary': {
                'total_missing': total_missing,
                'per_column': missing_counts,
                'per_column_percentage': missing_pcts
            },
            'duplicate_rows': duplicate_rows,
            'target_analysis': target_stats,
            'numerical_stats': num_stats,
            'invalid_financial_values': invalid_values,
            'categorical_distributions': cat_stats,
            'leakage_warnings': leakage_warnings
        }
