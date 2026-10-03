import json
from decimal import Decimal
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError, PermissionDenied

from apps.accounts.models import UserProfile
from apps.transactions.models import Transaction, DatasetRecord
from apps.fraud.models import Prediction, FraudAlert, InvestigationCase, InvestigationNote
from services.transaction_service import TransactionService
from services.alert_service import AlertService
from services.investigation_service import InvestigationService
from services.analytics_service import AnalyticsService
from services.report_service import ReportService
from services.batch_ingestion_service import BatchIngestionService


class Phase5IntelligenceBackendTests(TestCase):
    """
    Comprehensive test suite for Phase 5:
    Transaction Domain, Persistent Predictions, Alert Deduplication,
    Case Management, Separation of Predicted vs Confirmed Fraud, and REST APIs.
    """

    def setUp(self):
        self.client = Client()

        # Users with distinct RBAC roles
        self.admin_user = User.objects.create_superuser(username='admin_boss', password='password123', email='admin@test.com')
        self.analyst_user = User.objects.create_user(username='risk_analyst', password='password123', email='analyst@test.com')
        self.analyst_user.profile.role = UserProfile.ROLE_ANALYST
        self.analyst_user.profile.save()

        self.standard_user = User.objects.create_user(username='standard_user', password='password123', email='user@test.com')
        self.standard_user.profile.role = UserProfile.ROLE_USER
        self.standard_user.profile.save()

    # 1. Transaction Domain & Multi-Model Prediction Traceability
    def test_transaction_creation_and_auto_id_generation(self):
        """Transaction must generate unique TXN- prefixed ID and canonical inference dict."""
        tx = Transaction.objects.create(
            step=1,
            transaction_type=Transaction.TYPE_TRANSFER,
            amount=Decimal('50000.00'),
            name_orig='C12345678',
            old_balance_orig=Decimal('50000.00'),
            new_balance_orig=Decimal('0.00'),
            name_dest='C87654321',
            old_balance_dest=Decimal('0.00'),
            new_balance_dest=Decimal('50000.00')
        )
        self.assertTrue(tx.transaction_id.startswith('TXN-'))
        inf_dict = tx.to_inference_dict()
        self.assertEqual(inf_dict['transaction_id'], tx.transaction_id)
        self.assertEqual(inf_dict['amount'], 50000.0)

    def test_historical_multi_version_predictions_preserved(self):
        """A single transaction can have multiple predictions across different model versions."""
        tx = Transaction.objects.create(
            step=1,
            transaction_type=Transaction.TYPE_PAYMENT,
            amount=Decimal('100.00'),
            name_orig='C100',
            old_balance_orig=Decimal('500.00'),
            new_balance_orig=Decimal('400.00'),
            name_dest='M200',
            old_balance_dest=Decimal('0.00'),
            new_balance_dest=Decimal('0.00')
        )

        # Version 1.0 prediction
        pred_v1 = Prediction.objects.create(
            transaction=tx,
            model_name='Random Forest Classifier',
            model_version='1.0.0',
            decision_state='LEGITIMATE',
            risk_score=10.0,
            risk_tier='LOW'
        )

        # Version 1.1 prediction on same transaction
        pred_v1_1 = Prediction.objects.create(
            transaction=tx,
            model_name='Random Forest Classifier',
            model_version='1.1.0',
            decision_state='SUSPICIOUS',
            risk_score=35.0,
            risk_tier='MEDIUM'
        )

        detail = TransactionService.get_transaction_detail(tx.transaction_id)
        self.assertEqual(len(detail['predictions']), 2)
        versions = [p['model_version'] for p in detail['predictions']]
        self.assertIn('1.0.0', versions)
        self.assertIn('1.1.0', versions)

    # 2. Alert System & Deduplication
    def test_alert_generation_and_deduplication(self):
        """Identical transaction, model version, and rule must NOT produce duplicate alerts."""
        tx = Transaction.objects.create(
            step=1,
            transaction_type=Transaction.TYPE_TRANSFER,
            amount=Decimal('400000.00'),
            name_orig='C999',
            old_balance_orig=Decimal('400000.00'),
            new_balance_orig=Decimal('0.00'),
            name_dest='C888',
            old_balance_dest=Decimal('0.00'),
            new_balance_dest=Decimal('0.00')
        )
        pred = Prediction.objects.create(
            transaction=tx,
            model_name='Random Forest Classifier',
            model_version='1.0.0',
            risk_score=95.0,
            risk_tier='CRITICAL',
            decision_state='FRAUD'
        )
        analysis = {
            'risk_score': 95.0,
            'risk_level': 'CRITICAL',
            'fraud_probability': 0.98,
            'investigation_metadata': {'orig_drained': True}
        }

        # First alert generation
        alerts_1 = AlertService.evaluate_and_generate_alerts(pred, analysis, transaction_record=tx)
        self.assertGreater(len(alerts_1), 0)
        initial_count = FraudAlert.objects.filter(transaction=tx).count()

        # Second alert generation with identical data -> deduplication must prevent duplicate rows
        alerts_2 = AlertService.evaluate_and_generate_alerts(pred, analysis, transaction_record=tx)
        final_count = FraudAlert.objects.filter(transaction=tx).count()
        self.assertEqual(initial_count, final_count)

    def test_alert_state_machine_transitions(self):
        """Alerts must adhere strictly to valid state transition rules."""
        pred = Prediction.objects.create(risk_score=75.0, risk_tier='HIGH', decision_state='FRAUD')
        alert = FraudAlert.objects.create(
            prediction=pred,
            title="High Risk Transfer",
            severity=FraudAlert.SEVERITY_HIGH,
            status=FraudAlert.STATUS_NEW
        )

        # Valid: NEW -> UNDER_REVIEW
        alert.transition_to(FraudAlert.STATUS_UNDER_REVIEW, self.analyst_user, "Review initiated")
        self.assertEqual(alert.status, FraudAlert.STATUS_UNDER_REVIEW)

        # Invalid: UNDER_REVIEW -> NEW (cannot revert to NEW)
        with self.assertRaises(ValidationError):
            alert.transition_to(FraudAlert.STATUS_NEW, self.analyst_user, "Invalid transition")

    # 3. Investigation Case Management & Ground-Truth Resolution
    def test_investigation_case_workflow_and_resolution(self):
        """Case must track notes, synchronize alerts, and record ground-truth resolution."""
        pred = Prediction.objects.create(risk_score=90.0, risk_tier='CRITICAL', decision_state='FRAUD')
        alert = FraudAlert.objects.create(prediction=pred, title="Critical Alert", status=FraudAlert.STATUS_NEW)

        # 1. Create Case
        case = InvestigationService.create_case(
            title="Suspicious Mule Transfer Case",
            opened_by=self.analyst_user,
            priority=InvestigationCase.PRIORITY_HIGH,
            alert_ids=[alert.alert_id]
        )
        self.assertEqual(case.status, InvestigationCase.STATUS_OPEN)

        # 2. Add Investigation Note
        note = InvestigationService.add_note(case.case_id, self.analyst_user, "Called account holder; unauthorized transfer confirmed.")
        self.assertEqual(note.case, case)

        # 3. Ground Truth Resolution
        InvestigationService.resolve_case(
            case_id=case.case_id,
            resolution=InvestigationCase.RESOLUTION_CONFIRMED_FRAUD,
            resolution_notes="Customer confirmed account takeover.",
            analyst=self.analyst_user
        )
        case.refresh_from_db()
        self.assertEqual(case.status, InvestigationCase.STATUS_RESOLVED)
        self.assertEqual(case.resolution, InvestigationCase.RESOLUTION_CONFIRMED_FRAUD)

        # Linked alert should be transitioned to RESOLVED
        alert.refresh_from_db()
        self.assertEqual(alert.status, FraudAlert.STATUS_RESOLVED)

    # 4. Strict Separation of Predicted Fraud vs Confirmed Fraud
    def test_analytics_separates_predicted_fraud_from_confirmed_fraud(self):
        """Predicted fraud rate must NOT be conflated with human confirmed fraud rate."""
        # Create 10 transactions: 2 predicted fraud, 8 predicted legit
        for i in range(10):
            is_fraud_pred = (i < 2)
            tx = Transaction.objects.create(
                amount=Decimal('100.00'),
                transaction_type=Transaction.TYPE_PAYMENT,
                name_orig=f'C{i}',
                old_balance_orig=Decimal('500.00'),
                new_balance_orig=Decimal('400.00'),
                name_dest=f'M{i}',
                old_balance_dest=Decimal('0.00'),
                new_balance_dest=Decimal('0.00')
            )
            Prediction.objects.create(
                transaction=tx,
                decision_state=Prediction.DECISION_FRAUD if is_fraud_pred else Prediction.DECISION_LEGITIMATE,
                risk_score=90.0 if is_fraud_pred else 10.0,
                risk_tier='CRITICAL' if is_fraud_pred else 'LOW'
            )

        # Create 2 investigation cases: 1 confirmed fraud, 1 false positive
        case1 = InvestigationCase.objects.create(
            title="Case 1", opened_by=self.analyst_user, status=InvestigationCase.STATUS_RESOLVED,
            resolution=InvestigationCase.RESOLUTION_CONFIRMED_FRAUD
        )
        case2 = InvestigationCase.objects.create(
            title="Case 2", opened_by=self.analyst_user, status=InvestigationCase.STATUS_RESOLVED,
            resolution=InvestigationCase.RESOLUTION_FALSE_POSITIVE
        )

        overview = AnalyticsService.get_overview_metrics()

        # Predicted Fraud Rate: 2 / 10 = 20.0%
        self.assertEqual(overview['predicted_metrics']['predicted_fraud_count'], 2)
        self.assertEqual(overview['predicted_metrics']['predicted_fraud_rate_pct'], 20.0)

        # Confirmed Fraud Rate: 1 / 2 = 50.0%
        self.assertEqual(overview['confirmed_ground_truth']['confirmed_fraud_cases'], 1)
        self.assertEqual(overview['confirmed_ground_truth']['confirmed_fraud_rate_pct'], 50.0)
        self.assertEqual(overview['confirmed_ground_truth']['false_positive_rate_pct'], 50.0)

    # 5. Batch Ingestion with Valid and Invalid Rows
    def test_batch_ingestion_service_processes_mixed_data(self):
        """Batch ingestion must accept valid rows, reject invalid rows, and record error accounting."""
        import pandas as pd
        df = pd.DataFrame([
            # Valid row 1: Legit
            {'step': 1, 'type': 'PAYMENT', 'amount': 50.0, 'nameOrig': 'C101', 'oldbalanceOrg': 500.0, 'newbalanceOrig': 450.0, 'nameDest': 'M201', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0},
            # Valid row 2: Fraud pattern
            {'step': 1, 'type': 'TRANSFER', 'amount': 300000.0, 'nameOrig': 'C102', 'oldbalanceOrg': 300000.0, 'newbalanceOrig': 0.0, 'nameDest': 'C202', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0},
            # Invalid row 3: Negative amount
            {'step': 1, 'type': 'PAYMENT', 'amount': -100.0, 'nameOrig': 'C103', 'oldbalanceOrg': 500.0, 'newbalanceOrig': 600.0, 'nameDest': 'M203', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0}
        ])

        summary = BatchIngestionService.process_dataset(df, actor=self.analyst_user)
        self.assertEqual(summary['total_rows'], 3)
        self.assertEqual(summary['accepted_rows'], 2)
        self.assertEqual(summary['rejected_rows'], 1)
        self.assertEqual(summary['fraud_predictions'], 1)
        self.assertGreaterEqual(summary['alerts_generated'], 1)

    # 6. REST API Security & RBAC Enforcement
    def test_api_rbac_restrictions(self):
        """Standard users must be forbidden (403) from alerts, cases, and exports."""
        self.client.force_login(self.standard_user)

        # Standard user attempting to access alerts queue
        res_alerts = self.client.get('/api/alerts/')
        self.assertEqual(res_alerts.status_code, 403)

        # Standard user attempting to access cases
        res_cases = self.client.get('/api/cases/')
        self.assertEqual(res_cases.status_code, 403)

        # Standard user attempting to export CSV
        res_export = self.client.get('/api/reports/export/')
        self.assertEqual(res_export.status_code, 403)

        # Analyst user MUST be permitted (200)
        self.client.force_login(self.analyst_user)
        res_analyst_alerts = self.client.get('/api/alerts/')
        self.assertEqual(res_analyst_alerts.status_code, 200)
        self.assertTrue(res_analyst_alerts.json()['success'])

    def test_api_transaction_search_and_detail(self):
        """API must support searching transactions and returning standardized detail objects."""
        self.client.force_login(self.analyst_user)

        tx = Transaction.objects.create(
            step=1,
            transaction_type=Transaction.TYPE_TRANSFER,
            amount=Decimal('8888.00'),
            name_orig='CSEARCH_ORIG',
            old_balance_orig=Decimal('10000.00'),
            new_balance_orig=Decimal('1112.00'),
            name_dest='CSEARCH_DEST',
            old_balance_dest=Decimal('0.00'),
            new_balance_dest=Decimal('8888.00')
        )

        res = self.client.get(f'/api/transactions/?q=CSEARCH_ORIG')
        self.assertEqual(res.status_code, 200)
        json_data = res.json()
        self.assertTrue(json_data['success'])
        self.assertGreaterEqual(json_data['meta']['total_count'], 1)

        # Transaction Detail
        res_detail = self.client.get(f'/api/transactions/{tx.transaction_id}/')
        self.assertEqual(res_detail.status_code, 200)
        detail_data = res_detail.json()
        self.assertEqual(detail_data['data']['transaction']['transaction_id'], tx.transaction_id)
