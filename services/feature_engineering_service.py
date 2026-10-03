import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer extracting domain fraud features
    from raw financial transactions without data leakage.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms input DataFrame into engineered feature matrix.
        Accepts DataFrame or array. Returns DataFrame with engineered columns.
        """
        if isinstance(X, np.ndarray):
            # If array passed, convert to DataFrame using standard raw column names
            cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])
        else:
            df = X.copy()

        # Handle numeric casts safely
        amount = pd.to_numeric(df.get('amount', 0.0), errors='coerce').fillna(0.0)
        old_orig = pd.to_numeric(df.get('oldbalanceOrg', 0.0), errors='coerce').fillna(0.0)
        new_orig = pd.to_numeric(df.get('newbalanceOrig', 0.0), errors='coerce').fillna(0.0)
        old_dest = pd.to_numeric(df.get('oldbalanceDest', 0.0), errors='coerce').fillna(0.0)
        new_dest = pd.to_numeric(df.get('newbalanceDest', 0.0), errors='coerce').fillna(0.0)
        tx_type = df.get('type', pd.Series(['UNKNOWN'] * len(df))).astype(str).str.upper()

        # 1. Balance Change Features
        orig_balance_change = old_orig - new_orig
        dest_balance_change = new_dest - old_dest

        # 2. Mathematical Balance Discrepancy (Flags hidden/anomalous ledger updates)
        # Expected: old_orig - amount == new_orig => error = (old_orig - amount) - new_orig
        orig_discrepancy = (old_orig - amount) - new_orig
        # Expected: old_dest + amount == new_dest => error = (old_dest + amount) - new_dest
        dest_discrepancy = (old_dest + amount) - new_dest

        # 3. Ratio of Transaction Amount to Account Balance
        amount_to_old_orig_ratio = amount / (old_orig + 1.0)

        # 4. Behavioral Binary Flags
        orig_emptied = ((old_orig > 0) & (new_orig == 0)).astype(int)
        dest_zero_balances = ((old_dest == 0) & (new_dest == 0)).astype(int)
        is_high_value = ((amount >= 200000.0) & tx_type.isin(['TRANSFER', 'CASH_OUT'])).astype(int)

        engineered_df = pd.DataFrame({
            'type': tx_type,
            'amount': amount,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'orig_balance_change': orig_balance_change,
            'dest_balance_change': dest_balance_change,
            'orig_discrepancy': orig_discrepancy,
            'dest_discrepancy': dest_discrepancy,
            'amount_to_old_orig_ratio': amount_to_old_orig_ratio,
            'orig_emptied': orig_emptied,
            'dest_zero_balances': dest_zero_balances,
            'is_high_value': is_high_value,
        }, index=df.index)

        return engineered_df
