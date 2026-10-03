from django.test import TestCase
import pandas as pd
from services.preprocessing_service import PreprocessingService
from services.risk_service import RiskService
from services.model_service import ModelService, ModelLoadingError


class ArchitectureServiceTests(TestCase):
    """Verifies service layer modularity, feature schema, and risk scoring."""

    def test_preprocessing_service_enforces_canonical_features(self):
        """Preprocessing must output exact canonical features in correct order."""
        data = {
            'amount': [5000.0],
            'oldbalanceOrg': [10000.0],
            'newbalanceOrig': [5000.0],
            'oldbalanceDest': [0.0],
            'newbalanceDest': [5000.0],
            'extra_unwanted_col': ['sample'],
            'type': ['PAYMENT']
        }
        df = pd.DataFrame(data)
        features = PreprocessingService.extract_features_dataframe(df)

        self.assertEqual(list(features.columns), PreprocessingService.CANONICAL_FEATURES)
        self.assertEqual(features['isFlaggedFraud'].iloc[0], 0.0)

    def test_risk_service_high_risk_triggers(self):
        """Transfer draining account over $200k must trigger critical/high risk tier."""
        tx = {
            'type': 'TRANSFER',
            'amount': 250000.0,
            'oldbalanceOrg': 250000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        assessment = RiskService.calculate_risk_score(tx, model_prob=0.95)
        self.assertTrue(assessment['is_fraud'])
        self.assertIn(assessment['risk_tier'], [RiskService.TIER_HIGH, RiskService.TIER_CRITICAL])
        self.assertGreaterEqual(assessment['risk_score'], 80.0)

    def test_model_service_catalog_and_error_handling(self):
        """ModelService must report missing models clearly without crashing."""
        catalog = ModelService.get_available_models()
        self.assertIn('rfc', catalog)
        self.assertIn('dnn', catalog)

        with self.assertRaises(ModelLoadingError):
            ModelService.load_model('non_existent_key')
