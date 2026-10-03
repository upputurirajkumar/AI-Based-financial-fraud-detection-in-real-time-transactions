import numpy as np
import pandas as pd
from pathlib import Path


def generate_paysim_dataset(n_samples: int = 10000, fraud_ratio: float = 0.025, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a statistically authentic synthetic PaySim transaction dataset
    matching real-world financial fraud dynamics with realistic class imbalance.
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    records = []

    # 1. Generate Legitimate Transactions (n_legit)
    # Types: PAYMENT (40%), CASH_OUT (30%), CASH_IN (15%), TRANSFER (10%), DEBIT (5%)
    legit_types = rng.choice(
        ['PAYMENT', 'CASH_OUT', 'CASH_IN', 'TRANSFER', 'DEBIT'],
        size=n_legit,
        p=[0.40, 0.30, 0.15, 0.10, 0.05]
    )

    for i, t_type in enumerate(legit_types):
        step = int(rng.randint(1, 744))
        orig_id = f"C{rng.randint(100000000, 999999999)}"

        if t_type == 'PAYMENT':
            dest_id = f"M{rng.randint(100000000, 999999999)}"
            amount = round(float(rng.exponential(scale=5000.0) + 10.0), 2)
            old_orig = round(float(amount + rng.exponential(scale=10000.0)), 2)
            new_orig = round(float(max(old_orig - amount, 0.0)), 2)
            old_dest = 0.0  # Merchants in PaySim often show 0
            new_dest = 0.0
        elif t_type == 'CASH_OUT':
            dest_id = f"C{rng.randint(100000000, 999999999)}"
            amount = round(float(rng.exponential(scale=50000.0) + 50.0), 2)
            old_orig = round(float(amount + rng.exponential(scale=20000.0)), 2)
            new_orig = round(float(max(old_orig - amount, 0.0)), 2)
            old_dest = round(float(rng.exponential(scale=40000.0)), 2)
            new_dest = round(float(old_dest + amount), 2)
        elif t_type == 'CASH_IN':
            dest_id = f"C{rng.randint(100000000, 999999999)}"
            amount = round(float(rng.exponential(scale=40000.0) + 50.0), 2)
            old_orig = round(float(rng.exponential(scale=50000.0)), 2)
            new_orig = round(float(old_orig + amount), 2)
            old_dest = round(float(rng.exponential(scale=60000.0) + amount), 2)
            new_dest = round(float(max(old_dest - amount, 0.0)), 2)
        elif t_type == 'TRANSFER':
            dest_id = f"C{rng.randint(100000000, 999999999)}"
            amount = round(float(rng.exponential(scale=40000.0) + 100.0), 2)
            old_orig = round(float(amount + rng.exponential(scale=30000.0)), 2)
            new_orig = round(float(max(old_orig - amount, 0.0)), 2)
            old_dest = round(float(rng.exponential(scale=30000.0)), 2)
            new_dest = round(float(old_dest + amount), 2)
        else:  # DEBIT
            dest_id = f"C{rng.randint(100000000, 999999999)}"
            amount = round(float(rng.exponential(scale=3000.0) + 10.0), 2)
            old_orig = round(float(amount + rng.exponential(scale=8000.0)), 2)
            new_orig = round(float(max(old_orig - amount, 0.0)), 2)
            old_dest = round(float(rng.exponential(scale=15000.0)), 2)
            new_dest = round(float(old_dest + amount), 2)

        records.append({
            'step': step,
            'type': t_type,
            'amount': amount,
            'nameOrig': orig_id,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': dest_id,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'isFraud': 0,
            'isFlaggedFraud': 0
        })

    # 2. Generate Fraudulent Transactions (n_fraud)
    # In PaySim, fraud occurs almost exclusively in TRANSFER and CASH_OUT
    fraud_types = rng.choice(['TRANSFER', 'CASH_OUT'], size=n_fraud, p=[0.55, 0.45])

    for i, t_type in enumerate(fraud_types):
        step = int(rng.randint(1, 744))
        orig_id = f"C{rng.randint(100000000, 999999999)}"
        dest_id = f"C{rng.randint(100000000, 999999999)}"

        # Fraud typically involves draining high balances or full account drain
        amount = round(float(rng.uniform(10000.0, 1500000.0)), 2)
        old_orig = amount  # Classic PaySim signature: origin drained completely
        new_orig = 0.0

        if t_type == 'TRANSFER':
            # Destination often has 0 balance and remains 0 (money laundered immediately)
            old_dest = 0.0
            new_dest = 0.0 if rng.rand() < 0.6 else amount
        else:  # CASH_OUT
            old_dest = round(float(rng.uniform(0.0, 50000.0)), 2)
            new_dest = round(float(old_dest + amount), 2)

        is_flagged = 1 if (amount > 200000.0 and rng.rand() < 0.25) else 0

        records.append({
            'step': step,
            'type': t_type,
            'amount': amount,
            'nameOrig': orig_id,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': dest_id,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'isFraud': 1,
            'isFlaggedFraud': is_flagged
        })

    df = pd.DataFrame(records)
    # Shuffle records
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == '__main__':
    dataset_path = Path('data/raw/paysim_dataset.csv')
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_paysim_dataset(n_samples=10000, fraud_ratio=0.025, random_state=42)
    df.to_csv(dataset_path, index=False)
    print(f"Generated {len(df)} transactions -> {dataset_path}")
    print(f"Fraud count: {df['isFraud'].sum()} ({df['isFraud'].mean():.2%})")
