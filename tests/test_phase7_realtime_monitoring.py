import json
from unittest.mock import patch
from decimal import Decimal
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from apps.accounts.models import UserProfile
from apps.transactions.models import Transaction
from apps.fraud.models import Prediction, FraudAlert
from services.synthetic_generator import SyntheticTransactionGenerator
from services.event_processing_service import EventProcessingService
from services.live_stream_manager import LiveStreamManager
from services.fraud_engine import FraudEngine


class Phase7SyntheticGeneratorTestCase(TestCase):
    """Unit tests for the controlled Synthetic Transaction Generator."""

    def test_all_scenarios_generate_valid_events(self):
        """Every supported scenario produces a standardized event dictionary."""
        for scenario in SyntheticTransactionGenerator.SUPPORTED_SCENARIOS:
            event = SyntheticTransactionGenerator.generate_event(scenario=scenario)
            self.assertIn('event_id', event)
            self.assertIn('transaction_id', event)
            self.assertIn('timestamp', event)
            self.assertEqual(event['source'], 'SYNTHETIC')
            self.assertEqual(event['environment'], 'DEMO')
            self.assertIn('transaction_data', event)

            tx_data = event['transaction_data']
            # Must pass project schema validation
            val_errors = FraudEngine.validate_transaction(tx_data)
            self.assertEqual(val_errors, [], f"Validation errors for scenario {scenario}: {val_errors}")

            # Must have non-negative amount
            self.assertGreater(tx_data['amount'], 0)
            self.assertGreaterEqual(tx_data['oldbalanceOrg'], 0)
            self.assertGreaterEqual(tx_data['newbalanceOrig'], 0)

    def test_synthetic_labeling_is_explicit(self):
        """Synthetic transactions are never presented without synthetic markers."""
        event = SyntheticTransactionGenerator.generate_event(SyntheticTransactionGenerator.SCENARIO_HIGH_RISK_PATTERN)
        self.assertEqual(event['source'], 'SYNTHETIC')
        self.assertEqual(event['environment'], 'DEMO')
        self.assertEqual(event['scenario'], 'HIGH_RISK_PATTERN')


class Phase7EventProcessingServiceTestCase(TestCase):
    """Unit tests for the authoritative real-time Event Processing Engine."""

    def setUp(self):
        self.user = User.objects.create_user(username='test_analyst', password='password123')
        profile, _ = UserProfile.objects.get_or_create(user=self.user)
        profile.role = UserProfile.ROLE_ANALYST
        profile.save()

    def test_process_valid_synthetic_event(self):
        """Processes event through validation, persistence, FraudEngine inference, and alert rules."""
        event = SyntheticTransactionGenerator.generate_event(SyntheticTransactionGenerator.SCENARIO_HIGH_RISK_PATTERN)
        result = EventProcessingService.process_transaction_event(event, actor=self.user)

        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['source'], 'SYNTHETIC')
        self.assertGreater(result['total_latency_ms'], 0)

        # Check database records
        tx = Transaction.objects.get(transaction_id=result['transaction_id'])
        self.assertTrue(tx.is_synthetic)
        self.assertEqual(tx.source, Transaction.SOURCE_SYNTHETIC)
        self.assertEqual(tx.processing_status, Transaction.STATUS_COMPLETED)
        self.assertGreater(tx.latency_ms, 0)

        # Check prediction record
        pred = tx.predictions.first()
        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred.risk_score, 0.0)

    def test_idempotency_prevents_duplicate_records(self):
        """Re-submitting the exact same event returns cached analysis without creating duplicate rows."""
        event = SyntheticTransactionGenerator.generate_event(SyntheticTransactionGenerator.SCENARIO_NORMAL)
        result1 = EventProcessingService.process_transaction_event(event, actor=self.user)
        self.assertTrue(result1['success'])

        tx_count_before = Transaction.objects.count()
        pred_count_before = Prediction.objects.count()
        alert_count_before = FraudAlert.objects.count()

        # Submit identical event a second time
        result2 = EventProcessingService.process_transaction_event(event, actor=self.user)
        self.assertTrue(result2['success'])
        self.assertTrue(result2.get('idempotent', False))
        self.assertEqual(result2['transaction_id'], result1['transaction_id'])

        # No new records created
        self.assertEqual(Transaction.objects.count(), tx_count_before)
        self.assertEqual(Prediction.objects.count(), pred_count_before)
        self.assertEqual(FraudAlert.objects.count(), alert_count_before)

    def test_validation_failure_handling(self):
        """Rejects events with invalid data and returns descriptive errors."""
        invalid_event = {
            'event_id': 'EVT-INVALID',
            'transaction_id': 'TXN-INVALID',
            'source': 'SYNTHETIC',
            'transaction_data': {
                'step': -5,  # Invalid negative step
                'type': 'INVALID_TYPE',
                'amount': -100.0,
            }
        }
        result = EventProcessingService.process_transaction_event(invalid_event, actor=self.user)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'VALIDATION_FAILED')
        self.assertTrue(len(result['errors']) > 0)


class Phase7LiveStreamManagerTestCase(TestCase):
    """Unit tests for the LiveStreamManager singleton coordinator."""

    def setUp(self):
        self.manager = LiveStreamManager()
        self.manager.stop()  # Ensure clean starting state

    def tearDown(self):
        self.manager.stop()

    def test_singleton_consistency(self):
        """Manager maintains singular global operational state."""
        mgr2 = LiveStreamManager()
        self.assertIs(self.manager, mgr2)

    def test_step_execution(self):
        """Executing a step creates and records one event in memory and database."""
        initial_processed = self.manager.processed_count
        result = self.manager.step(scenario=SyntheticTransactionGenerator.SCENARIO_NORMAL)
        self.assertTrue(result['success'])
        self.assertEqual(self.manager.processed_count, initial_processed + 1)
        self.assertGreater(len(self.manager.recent_events), 0)

    @patch('services.live_stream_manager.threading.Thread')
    def test_lifecycle_controls_mocked_thread(self, mock_thread_cls):
        """Manager transitions correctly across start, pause, resume, and stop states."""
        mock_thread = mock_thread_cls.return_value
        status = self.manager.start(scenario=SyntheticTransactionGenerator.SCENARIO_NORMAL, rate_per_minute=20, max_events=10)
        self.assertEqual(status['status'], LiveStreamManager.STATUS_RUNNING)
        self.assertTrue(mock_thread.start.called)

        status = self.manager.pause()
        self.assertEqual(status['status'], LiveStreamManager.STATUS_PAUSED)

        status = self.manager.resume()
        self.assertEqual(status['status'], LiveStreamManager.STATUS_RUNNING)

        status = self.manager.stop()
        self.assertEqual(status['status'], LiveStreamManager.STATUS_STOPPED)

    def test_delta_streaming_and_status(self):
        """Returns delta stream and telemetry snapshots."""
        self.manager.step()
        delta = self.manager.get_stream_delta()
        self.assertIn('status', delta)
        self.assertIn('events', delta)
        self.assertIn('server_timestamp', delta)
        self.assertEqual(delta['status']['is_demo_stream'], True)


class Phase7LiveMonitoringAPITestCase(TestCase):
    """Integration tests for Live Monitoring REST endpoints and UI view."""

    def setUp(self):
        self.client = Client()
        self.analyst = User.objects.create_user(username='risk_analyst', password='password123')
        profile, _ = UserProfile.objects.get_or_create(user=self.analyst)
        profile.role = UserProfile.ROLE_ANALYST
        profile.save()

        self.regular_user = User.objects.create_user(username='regular_user', password='password123')
        profile_reg, _ = UserProfile.objects.get_or_create(user=self.regular_user)
        profile_reg.role = UserProfile.ROLE_USER
        profile_reg.save()


        LiveStreamManager().stop()

    def tearDown(self):
        LiveStreamManager().stop()

    def test_unauthenticated_access_denied(self):
        """Unauthenticated requests cannot access live status or control endpoints."""
        res = self.client.get(reverse('api_live_status'))
        self.assertIn(res.status_code, [302, 401, 403])

        res = self.client.post(reverse('api_live_start'), data=json.dumps({}), content_type='application/json')
        self.assertIn(res.status_code, [302, 401, 403])

    def test_analyst_can_access_status_and_metrics(self):
        """Authorized analyst can query live status, transactions, alerts, and metrics."""
        self.client.login(username='risk_analyst', password='password123')

        # Status
        res = self.client.get(reverse('api_live_status'))
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['is_demo_stream'], True)

        # Metrics
        res = self.client.get(reverse('api_live_metrics'))
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertIn('processed_count', data['data'])

        # Transactions
        res = self.client.get(reverse('api_live_transactions'))
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()['success'])

        # Alerts
        res = self.client.get(reverse('api_live_alerts'))
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()['success'])

    def test_analyst_can_step_and_stream_delta(self):
        """Analyst can trigger on-demand step and receive stream delta."""
        self.client.login(username='risk_analyst', password='password123')

        res = self.client.post(
            reverse('api_live_step'),
            data=json.dumps({'scenario': 'HIGH_RISK_PATTERN'}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['source'], 'SYNTHETIC')

        # Poll stream delta
        res = self.client.get(reverse('api_live_stream_delta'))
        self.assertEqual(res.status_code, 200)
        delta_data = res.json()
        self.assertTrue(delta_data['success'])
        self.assertGreater(len(delta_data['data']['events']), 0)

    def test_live_monitoring_ui_view_renders_correctly(self):
        """GET /live/ renders the Live Monitoring dashboard with synthetic markers."""
        self.client.login(username='risk_analyst', password='password123')
        res = self.client.get(reverse('live_monitoring'))
        self.assertEqual(res.status_code, 200)
        content = res.content.decode('utf-8')
        self.assertIn('Live Transaction Monitoring', content)
        self.assertIn('SYNTHETIC DEMO STREAM', content)
        self.assertIn('SYNTHETIC DATA', content)
        self.assertIn('Scenario:', content)
        self.assertIn('Start Stream', content)
