from django.test import TestCase
import numpy as np
import pandas as pd
from services.fraud_engine import FraudEngine
from services.risk_service import RiskService
from services.explainability_service import ExplainabilityService
from services.alert_service import AlertService
from apps.fraud.models import Prediction, FraudAlert
from apps.transactions.models import Transaction


class Phase4FraudIntelligenceTests(TestCase):
    """Rigorous unit tests for Phase 4 AI Fraud Detection Engine, Risk Intelligence, and XAI."""

    def setUp(self):
        self.legit_tx = {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 50.0,
            'oldbalanceOrg': 500.0,
            'newbalanceOrig': 450.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'nameOrig': 'C10001',
            'nameDest': 'M20001'
        }

        self.fraud_tx = {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 300000.0,
            'oldbalanceOrg': 300000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'nameOrig': 'C99999',
            'nameDest': 'C88888'
        }

    # 1. Fraud Engine & Standardized Prediction Contract Tests
    def test_fraud_engine_valid_legitimate_transaction(self):
        """Legitimate routine transaction should yield LEGITIMATE decision state and low risk."""
        result = FraudEngine.analyze_transaction(self.legit_tx)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['decision_state'], FraudEngine.DECISION_LEGITIMATE)
        self.assertLess(result['risk_score'], 30.0)
        self.assertEqual(result['risk_level'], 'LOW')
        self.assertIn('explanation', result)
        self.assertIn('anomaly_info', result)
        self.assertIn('investigation_metadata', result)

    def test_fraud_engine_valid_fraud_transaction(self):
        """High-value transfer with drained origin and zero destination should trigger FRAUD."""
        result = FraudEngine.analyze_transaction(self.fraud_tx)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['decision_state'], FraudEngine.DECISION_FRAUD)
        self.assertGreaterEqual(result['risk_score'], 85.0)
        self.assertEqual(result['risk_level'], 'CRITICAL')
        self.assertGreaterEqual(result['fraud_probability'], result['threshold'])

    def test_fraud_engine_input_validation_catches_invalid_inputs(self):
        """Engine must reject negative numbers, NaNs, and missing fields cleanly without throwing."""
        bad_cases = [
            ({'amount': -100.0, 'oldbalanceOrg': 100.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0}, "negative"),
            ({'amount': float('nan'), 'oldbalanceOrg': 100.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0}, "NaN"),
            ({'amount': 100.0, 'oldbalanceOrg': 100.0}, "missing fields"),
        ]
        for case, label in bad_cases:
            res = FraudEngine.analyze_transaction(case)
            self.assertEqual(res['status'], 'INVALID_INPUT', f"Failed for {label}")
            self.assertEqual(res['decision_state'], 'ERROR')
            self.assertGreater(len(res['errors']), 0)

    # 2. Risk Intelligence Layer & Boundary Tests
    def test_risk_service_tier_boundaries(self):
        """Risk bands must strictly categorize exact boundary scores."""
        self.assertEqual(RiskService.get_tier_for_score(0.0), 'LOW')
        self.assertEqual(RiskService.get_tier_for_score(29.9), 'LOW')
        self.assertEqual(RiskService.get_tier_for_score(30.0), 'MEDIUM')
        self.assertEqual(RiskService.get_tier_for_score(59.9), 'MEDIUM')
        self.assertEqual(RiskService.get_tier_for_score(60.0), 'HIGH')
        self.assertEqual(RiskService.get_tier_for_score(84.9), 'HIGH')
        self.assertEqual(RiskService.get_tier_for_score(85.0), 'CRITICAL')
        self.assertEqual(RiskService.get_tier_for_score(100.0), 'CRITICAL')

    def test_risk_service_probability_bounding(self):
        """Out of bound probabilities must be bounded safely into [0, 1]."""
        score_low = RiskService.calculate_risk_score(self.legit_tx, model_prob=-0.5)
        score_high = RiskService.calculate_risk_score(self.legit_tx, model_prob=2.5)
        self.assertGreaterEqual(score_low['risk_score'], 0.0)
        self.assertLessEqual(score_high['risk_score'], 100.0)

    # 3. Decision Threshold Tests
    def test_threshold_sensitivity(self):
        """Custom threshold must govern fraud decisioning strictly."""
        # Using a mid-probability transaction
        tx = {
            'step': 1,
            'type': 'CASH_OUT',
            'amount': 25000.0,
            'oldbalanceOrg': 25000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 10000.0,
            'newbalanceDest': 35000.0,
            'nameOrig': 'C555',
            'nameDest': 'C666'
        }
        # With threshold 0.10 -> classified FRAUD
        res_low = FraudEngine.analyze_transaction(tx, custom_threshold=0.01)
        self.assertIn(res_low['decision_state'], [FraudEngine.DECISION_FRAUD, FraudEngine.DECISION_SUSPICIOUS])

        # With impossible threshold 0.999 -> not classified as model fraud
        res_high = FraudEngine.analyze_transaction(tx, custom_threshold=0.999)
        self.assertNotEqual(res_high['decision_state'], FraudEngine.DECISION_FRAUD)

    # 4. Explainable AI & Non-Accusatory Language Tests
    def test_explainability_global_importances(self):
        """Global feature importances must return non-empty list of sorted importances."""
        importances = ExplainabilityService.get_global_feature_importances()
        self.assertGreater(len(importances), 0)
        # Verify sorted descending
        for i in range(len(importances) - 1):
            self.assertGreaterEqual(importances[i]['importance_score'], importances[i + 1]['importance_score'])

    def test_explainability_local_factors_and_safety_language(self):
        """Local explanation must identify key factors using non-accusatory compliance language."""
        explanation = ExplainabilityService.explain_transaction(self.fraud_tx, fraud_probability=0.95)
        self.assertEqual(explanation['status'], 'AVAILABLE')
        self.assertIn('disclaimer', explanation)
        self.assertIn('decision-support', explanation['disclaimer'].lower())

        # Verify no accusatory words in descriptions
        for factor in explanation['risk_factors']:
            desc = factor['description'].lower()
            self.assertNotIn("criminal", desc)
            self.assertNotIn("proves fraud", desc)
            self.assertNotIn("is a criminal", desc)

    # 5. Anomaly Detection Tests
    def test_anomaly_detection_provides_normalized_score(self):
        """Anomaly detector must provide score in [0, 100] and boolean anomaly flag."""
        res = FraudEngine.analyze_transaction(self.legit_tx, include_anomaly=True)
        anom = res['anomaly_info']
        self.assertEqual(anom['status'], 'AVAILABLE')
        self.assertGreaterEqual(anom['anomaly_score'], 0.0)
        self.assertLessEqual(anom['anomaly_score'], 100.0)
        self.assertIsInstance(anom['is_anomalous'], bool)

    # 6. Alert Decision Foundation Tests
    def test_alert_candidate_generation_for_high_risk_only(self):
        """AlertService must create candidate alert for CRITICAL/HIGH risk and skip LOW risk."""
        # Create database prediction for fraud
        pred_fraud = Prediction.objects.create(
            risk_score=95.0,
            risk_tier='CRITICAL',
            decision_state='FRAUD',
            is_fraud=True
        )
        alert = AlertService.evaluate_for_alert(
            pred_fraud,
            {'risk_level': 'CRITICAL', 'decision_state': 'FRAUD', 'risk_score': 95.0}
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, 'CRITICAL')
        self.assertEqual(alert.status, FraudAlert.STATUS_OPEN)

        # Prediction for low risk should NOT generate an alert
        pred_legit = Prediction.objects.create(
            risk_score=10.0,
            risk_tier='LOW',
            decision_state='LEGITIMATE',
            is_fraud=False
        )
        no_alert = AlertService.evaluate_for_alert(
            pred_legit,
            {'risk_level': 'LOW', 'decision_state': 'LEGITIMATE', 'risk_score': 10.0}
        )
        self.assertIsNone(no_alert)
