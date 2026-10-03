import random
import uuid
from typing import Dict, Any, Optional
from django.utils import timezone


class SyntheticTransactionGenerator:
    """
    Controlled generator producing realistic financial transaction structures
    modeled on the project's PaySim schema and banking fraud topologies.

    Every generated transaction is explicitly marked as SYNTHETIC to guarantee
    that simulated demonstration data is never conflated with live financial accounts.
    """

    SCENARIO_NORMAL = 'NORMAL'
    SCENARIO_HIGH_AMOUNT = 'HIGH_AMOUNT'
    SCENARIO_HIGH_RISK_PATTERN = 'HIGH_RISK_PATTERN'
    SCENARIO_ANOMALOUS = 'ANOMALOUS'
    SCENARIO_RAPID_SEQUENCE = 'RAPID_SEQUENCE'
    SCENARIO_MIXED_STREAM = 'MIXED_STREAM'

    SUPPORTED_SCENARIOS = [
        SCENARIO_NORMAL,
        SCENARIO_HIGH_AMOUNT,
        SCENARIO_HIGH_RISK_PATTERN,
        SCENARIO_ANOMALOUS,
        SCENARIO_RAPID_SEQUENCE,
        SCENARIO_MIXED_STREAM,
    ]

    _step_counter = 1
    _seq_account_orig = "C_SEQ_VICTIM_9081"
    _seq_account_dest = "M_SEQ_MULE_1044"

    @classmethod
    def generate_event(cls, scenario: str = SCENARIO_NORMAL, custom_step: Optional[int] = None) -> Dict[str, Any]:
        """
        Generates a standardized transaction event dict conforming to the project schema.
        """
        if scenario == cls.SCENARIO_MIXED_STREAM:
            # 80% Normal, 15% High Risk, 5% Anomalous
            roll = random.random()
            if roll < 0.80:
                resolved_scenario = cls.SCENARIO_NORMAL
            elif roll < 0.95:
                resolved_scenario = cls.SCENARIO_HIGH_RISK_PATTERN
            else:
                resolved_scenario = cls.SCENARIO_ANOMALOUS
        else:
            resolved_scenario = scenario if scenario in cls.SUPPORTED_SCENARIOS else cls.SCENARIO_NORMAL

        step = custom_step if custom_step is not None else cls._step_counter
        cls._step_counter = (cls._step_counter % 744) + 1  # 31 days PaySim cycle

        event_id = f"EVT-{uuid.uuid4().hex[:12].upper()}"
        transaction_id = f"TXN-SYNTH-{uuid.uuid4().hex[:10].upper()}"

        tx_payload = cls._build_scenario_payload(resolved_scenario, step)

        return {
            'event_id': event_id,
            'transaction_id': transaction_id,
            'timestamp': timezone.now().isoformat(),
            'source': 'SYNTHETIC',
            'environment': 'DEMO',
            'scenario': resolved_scenario,
            'transaction_data': tx_payload,
        }

    @classmethod
    def _build_scenario_payload(cls, scenario: str, step: int) -> Dict[str, Any]:
        if scenario == cls.SCENARIO_NORMAL:
            return cls._build_normal_payload(step)
        elif scenario == cls.SCENARIO_HIGH_AMOUNT:
            return cls._build_high_amount_payload(step)
        elif scenario == cls.SCENARIO_HIGH_RISK_PATTERN:
            return cls._build_high_risk_payload(step)
        elif scenario == cls.SCENARIO_ANOMALOUS:
            return cls._build_anomalous_payload(step)
        elif scenario == cls.SCENARIO_RAPID_SEQUENCE:
            return cls._build_rapid_sequence_payload(step)
        else:
            return cls._build_normal_payload(step)

    @classmethod
    def _build_normal_payload(cls, step: int) -> Dict[str, Any]:
        """Typical benign retail or merchant transaction with consistent balances."""
        tx_type = random.choice(['PAYMENT', 'PAYMENT', 'PAYMENT', 'CASH_IN', 'DEBIT'])
        amount = round(random.uniform(12.50, 480.00), 2)
        old_orig = round(random.uniform(amount + 50.0, 15000.0), 2)
        
        if tx_type == 'CASH_IN':
            new_orig = round(old_orig + amount, 2)
            name_orig = f"C_BENIGN_{random.randint(10000, 99999)}"
            name_dest = f"C_BANK_POS_{random.randint(1000, 9999)}"
            old_dest = round(random.uniform(500.0, 10000.0), 2)
            new_dest = round(max(0.0, old_dest - amount), 2)
            channel = 'ATM'
        elif tx_type == 'DEBIT':
            new_orig = round(max(0.0, old_orig - amount), 2)
            name_orig = f"C_BENIGN_{random.randint(10000, 99999)}"
            name_dest = f"M_UTILITY_{random.randint(1000, 9999)}"
            old_dest = 0.0
            new_dest = 0.0
            channel = 'ONLINE'
        else:  # PAYMENT
            new_orig = round(max(0.0, old_orig - amount), 2)
            name_orig = f"C_SHOPPER_{random.randint(10000, 99999)}"
            name_dest = f"M_MERCHANT_{random.randint(1000, 9999)}"
            old_dest = 0.0
            new_dest = 0.0
            channel = random.choice(['ONLINE', 'MOBILE', 'POS'])

        return {
            'step': step,
            'type': tx_type,
            'amount': amount,
            'currency': 'USD',
            'channel': channel,
            'nameOrig': name_orig,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': name_dest,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'metadata': {
                'demo_scenario': 'NORMAL',
                'description': 'Typical low-value retail consumer payment',
            }
        }

    @classmethod
    def _build_high_amount_payload(cls, step: int) -> Dict[str, Any]:
        """High-value transfer with valid matching ledger changes (tests volume rule)."""
        tx_type = random.choice(['TRANSFER', 'CASH_OUT'])
        amount = round(random.uniform(320000.0, 850000.0), 2)
        old_orig = round(amount + random.uniform(50000.0, 200000.0), 2)
        new_orig = round(old_orig - amount, 2)
        old_dest = round(random.uniform(10000.0, 100000.0), 2)
        new_dest = round(old_dest + amount, 2)

        return {
            'step': step,
            'type': tx_type,
            'amount': amount,
            'currency': 'USD',
            'channel': 'WIRE' if tx_type == 'TRANSFER' else 'ATM',
            'nameOrig': f"C_CORPORATE_{random.randint(1000, 9999)}",
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': f"C_ESCROW_{random.randint(1000, 9999)}",
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'metadata': {
                'demo_scenario': 'HIGH_AMOUNT',
                'description': 'Elevated high-value corporate treasury transfer',
            }
        }

    @classmethod
    def _build_high_risk_payload(cls, step: int) -> Dict[str, Any]:
        """Canonical fraud pattern: complete balance liquidation with drained origin and zero destination."""
        amount = round(random.uniform(180000.0, 620000.0), 2)
        old_orig = amount  # Drains 100% of origin balance
        new_orig = 0.0
        old_dest = 0.0
        new_dest = 0.0  # Zero destination footprint typical of cash-out mules

        return {
            'step': step,
            'type': 'TRANSFER',
            'amount': amount,
            'currency': 'USD',
            'channel': 'ONLINE',
            'nameOrig': f"C_VICTIM_{random.randint(1000, 9999)}",
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': f"M_MULE_{random.randint(1000, 9999)}",
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'metadata': {
                'demo_scenario': 'HIGH_RISK_PATTERN',
                'description': 'Complete account drainage via unauthorized wire transfer',
            }
        }

    @classmethod
    def _build_anomalous_payload(cls, step: int) -> Dict[str, Any]:
        """Out-of-distribution balance mismatch designed to trigger unsupervised anomaly detection."""
        amount = round(random.uniform(50000.0, 150000.0), 2)
        old_orig = 10000.0  # Balance significantly smaller than transaction amount
        new_orig = 0.0
        old_dest = 20000.0
        new_dest = 20000.0  # Zero destination balance increase despite large transfer

        return {
            'step': step,
            'type': 'TRANSFER',
            'amount': amount,
            'currency': 'USD',
            'channel': 'MOBILE',
            'nameOrig': f"C_ANOMALY_{random.randint(1000, 9999)}",
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': f"C_RECEIVER_{random.randint(1000, 9999)}",
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'metadata': {
                'demo_scenario': 'ANOMALOUS',
                'description': 'Severe balance discrepancy exceeding initial account value',
            }
        }

    @classmethod
    def _build_rapid_sequence_payload(cls, step: int) -> Dict[str, Any]:
        """Successive transfers originating from a single target account within identical window."""
        amount = round(random.uniform(40000.0, 95000.0), 2)
        old_orig = 300000.0
        new_orig = max(0.0, old_orig - amount)

        return {
            'step': step,
            'type': 'TRANSFER',
            'amount': amount,
            'currency': 'USD',
            'channel': 'ONLINE',
            'nameOrig': cls._seq_account_orig,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': f"M_AGENT_{random.randint(100, 999)}",
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'metadata': {
                'demo_scenario': 'RAPID_SEQUENCE',
                'description': 'Rapid repeated transfer sequence from target account',
            }
        }
