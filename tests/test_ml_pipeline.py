from django.test import TestCase
import numpy as np
import pandas as pd
from services.data_quality_service import DataQualityService
from services.feature_engineering_service import DomainFeatureTransformer
from services.ml_pipeline import create_preprocessor, build_full_pipeline
from services.model_service import ModelService
from services.prediction_service import PredictionService


class MLPipelineModernizationTests(TestCase):
    """Rigorous unit tests for Phase 3 Data Engineering & ML Pipeline."""

    def setUp(self):
        # Sample clean dataset for testing
        self.sample_df = pd.DataFrame({
            'step': [1, 2, 3],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100.0, 50000.0, 250000.0],
            'nameOrig': ['C1001', 'C1002', 'C1003'],
            'oldbalanceOrg': [200.0, 50000.0, 250000.0],
            'newbalanceOrig': [100.0, 0.0, 0.0],
            'nameDest': ['M2001', 'C2002', 'C2003'],
            'oldbalanceDest': [0.0, 0.0, 10000.0],
            'newbalanceDest': [0.0, 0.0, 260000.0],
            'isFraud': [0, 1, 1],
            'isFlaggedFraud': [0, 0, 1]
        })

    # 1. Data Quality & Validation Tests
    def test_data_quality_audit_computes_correct_metrics(self):
        """DataQualityService must report accurate counts and class distributions."""
        report = DataQualityService.audit_dataset(self.sample_df)
        self.assertEqual(report['shape']['rows'], 3)
        self.assertEqual(report['shape']['columns'], 11)
        self.assertEqual(report['target_analysis']['fraud_count'], 2)
        self.assertEqual(report['target_analysis']['legitimate_count'], 1)
        self.assertEqual(report['missing_summary']['total_missing'], 0)

    def test_invalid_negative_financial_values_detected(self):
        """Negative amounts or balances must be flagged as invalid."""
        bad_df = self.sample_df.copy()
        bad_df.loc[0, 'amount'] = -500.0
        report = DataQualityService.audit_dataset(bad_df)
        self.assertIn('amount', report['invalid_financial_values'])
        self.assertEqual(report['invalid_financial_values']['amount']['negative_count'], 1)

    # 2. Preprocessing & Feature Engineering Tests
    def test_feature_engineering_discrepancy_calculations(self):
        """Feature engineering must calculate mathematical balance discrepancy correctly."""
        transformer = DomainFeatureTransformer()
        feat_df = transformer.transform(self.sample_df)

        # In row 0: old 200, amount 100, new 100 -> discrepancy = (200 - 100) - 100 = 0
        self.assertAlmostEqual(feat_df.loc[0, 'orig_discrepancy'], 0.0)
        # In row 1: origin drained -> orig_emptied = 1
        self.assertEqual(feat_df.loc[1, 'orig_emptied'], 1)
        # In row 2: amount 250k on CASH_OUT -> is_high_value = 1
        self.assertEqual(feat_df.loc[2, 'is_high_value'], 1)

    def test_preprocessing_handles_unknown_categorical_values_safely(self):
        """Preprocessor must safely ignore unseen transaction types without crashing."""
        preprocessor = create_preprocessor()
        preprocessor.fit(self.sample_df)

        unseen_df = pd.DataFrame([{
            'step': 1,
            'type': 'FOREIGN_EXCHANGE_UNKNOWN',
            'amount': 100.0,
            'oldbalanceOrg': 200.0,
            'newbalanceOrig': 100.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }])

        # Should transform without raising KeyError or ValueError
        trans_matrix = preprocessor.transform(unseen_df)
        self.assertEqual(trans_matrix.shape[0], 1)
        # Verify categorical one-hot columns are all 0 for unknown category
        cat_one_hot = trans_matrix[0, :5]
        self.assertTrue(np.all(cat_one_hot == 0.0))

    # 3. Model Inference & Threshold Selection Tests
    def test_production_model_loads_and_predicts(self):
        """Production pipeline artifact must load and produce calibrated probabilities."""
        pipeline = ModelService.load_model('rfc_production')
        self.assertIsNotNone(pipeline)

        probs = pipeline.predict_proba(self.sample_df)
        self.assertEqual(probs.shape, (3, 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_prediction_service_single_record_inference(self):
        """PredictionService.predict_single must return structured output with threshold."""
        tx = {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 500000.0,
            'oldbalanceOrg': 500000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        res = PredictionService.predict_single(tx, model_key='rfc_production')
        self.assertEqual(res['status'], 'SUCCESS')
        self.assertIn(res['prediction'], ['FRAUD', 'LEGITIMATE'])
        self.assertGreaterEqual(res['fraud_probability'], 0.0)
        self.assertLessEqual(res['fraud_probability'], 1.0)
        self.assertIn('decision_threshold', res)
        self.assertIn('risk_score', res)

    def test_prediction_service_batch_inference_with_mixed_rows(self):
        """Batch prediction must process valid rows and audit invalid rows without dropping them."""
        mixed_df = pd.DataFrame([
            # Valid row
            {'step': 1, 'type': 'PAYMENT', 'amount': 100.0, 'oldbalanceOrg': 200.0, 'newbalanceOrig': 100.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0},
            # Invalid negative amount
            {'step': 2, 'type': 'PAYMENT', 'amount': -50.0, 'oldbalanceOrg': 200.0, 'newbalanceOrig': 100.0, 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0},
            # Missing balance field
            {'step': 3, 'type': 'TRANSFER', 'amount': 1000.0, 'oldbalanceOrg': 1000.0, 'newbalanceOrig': 0.0, 'oldbalanceDest': None, 'newbalanceDest': 0.0}
        ])

        batch_res = PredictionService.predict_batch(mixed_df, model_key='rfc_production')
        self.assertEqual(batch_res['total_rows'], 3)
        self.assertEqual(batch_res['valid_rows'], 1)
        self.assertEqual(batch_res['invalid_rows'], 2)
        self.assertEqual(len(batch_res['processing_errors']), 2)
        self.assertEqual(len(batch_res['records']), 3)
