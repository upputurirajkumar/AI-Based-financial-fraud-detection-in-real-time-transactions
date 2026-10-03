from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from .feature_engineering_service import DomainFeatureTransformer

# Fixed category order for transaction types to ensure deterministic encoding
KNOWN_TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

NUMERICAL_COLS = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'orig_balance_change', 'dest_balance_change', 'orig_discrepancy', 'dest_discrepancy',
    'amount_to_old_orig_ratio'
]

BINARY_COLS = [
    'orig_emptied', 'dest_zero_balances', 'is_high_value'
]


def create_preprocessor() -> Pipeline:
    """
    Creates a unified, reproducible preprocessing pipeline:
    DomainFeatureTransformer -> ColumnTransformer (OneHotEncoder + RobustScaler).
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(
                categories=[KNOWN_TRANSACTION_TYPES],
                handle_unknown='ignore',
                sparse_output=False
            ), ['type']),
            ('num', RobustScaler(), NUMERICAL_COLS),
            ('passthrough', 'passthrough', BINARY_COLS)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('domain_features', DomainFeatureTransformer()),
        ('preprocessor', column_transformer)
    ])

    return pipeline


def build_full_pipeline(classifier) -> Pipeline:
    """Combines domain preprocessing and any Scikit-learn compatible classifier into a single artifact."""
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(
                categories=[KNOWN_TRANSACTION_TYPES],
                handle_unknown='ignore',
                sparse_output=False
            ), ['type']),
            ('num', RobustScaler(), NUMERICAL_COLS),
            ('passthrough', 'passthrough', BINARY_COLS)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline(steps=[
        ('domain_features', DomainFeatureTransformer()),
        ('preprocessor', column_transformer),
        ('classifier', classifier)
    ])

    return full_pipeline
