import time
import threading
import logging
from collections import deque
from typing import Dict, Any, List, Optional
from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.exceptions import PermissionDenied

from apps.core.models import AuditLog
from services.synthetic_generator import SyntheticTransactionGenerator
from services.event_processing_service import EventProcessingService

logger = logging.getLogger(__name__)


class LiveStreamManager:
    """
    Singleton operational coordinator for the Real-Time Synthetic Transaction Stream.
    Manages thread execution, rate controls, safety boundaries, bounded event buffers,
    and live metrics calculation.
    """

    STATUS_STOPPED = 'STOPPED'
    STATUS_RUNNING = 'RUNNING'
    STATUS_PAUSED = 'PAUSED'

    _instance = None
    _lock = threading.RLock()


    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LiveStreamManager, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.status = self.STATUS_STOPPED
        self.scenario = SyntheticTransactionGenerator.SCENARIO_NORMAL
        self.rate_per_minute = 15  # Default: 1 event every 4 seconds
        self.max_events = getattr(settings, 'MAX_SYNTHETIC_EVENTS_PER_SESSION', 150)
        self.processed_count = 0
        self.alerts_count = 0
        self.high_risk_count = 0
        self.failure_count = 0
        self.started_at: Optional[str] = None
        self.started_by_username: str = "system"

        # Bounded client buffers (prevents unbounded memory growth)
        self.recent_events = deque(maxlen=50)
        self.recent_alerts = deque(maxlen=25)
        self.latencies = deque(maxlen=30)
        self.risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

    def start(
        self,
        scenario: str = SyntheticTransactionGenerator.SCENARIO_NORMAL,
        rate_per_minute: int = 15,
        max_events: Optional[int] = None,
        actor: Optional[User] = None
    ) -> Dict[str, Any]:
        """Starts background synthetic generation thread under safety bounds."""
        if not getattr(settings, 'REAL_TIME_DEMO_ENABLED', True):
            raise PermissionDenied("Synthetic live stream is disabled in this environment.")

        with self._lock:
            if self.status == self.STATUS_RUNNING:
                return self.get_status()

            # Enforce bounds
            max_allowed = getattr(settings, 'MAX_SYNTHETIC_EVENTS_PER_SESSION', 300)
            max_rate = getattr(settings, 'MAX_SYNTHETIC_RATE_PER_MINUTE', 60)

            self.scenario = scenario if scenario in SyntheticTransactionGenerator.SUPPORTED_SCENARIOS else SyntheticTransactionGenerator.SCENARIO_NORMAL
            self.rate_per_minute = max(1, min(int(rate_per_minute), max_rate))
            self.max_events = max(5, min(int(max_events or 100), max_allowed))
            self.status = self.STATUS_RUNNING
            self.started_at = timezone.now().isoformat()
            self.started_by_username = actor.username if actor and actor.is_authenticated else "analyst"

            self._stop_event.clear()
            self._pause_event.set()

            # Launch daemon worker
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="SyntheticStreamWorker")
            self._thread.start()

            if actor and actor.is_authenticated:
                AuditLog.objects.create(
                    user=actor,
                    action='STREAM_START',
                    details=f"Started synthetic stream: Scenario={self.scenario}, Rate={self.rate_per_minute}/min, Max={self.max_events}"
                )

            logger.info("LiveStreamManager started by %s: %s at %d evt/min", self.started_by_username, self.scenario, self.rate_per_minute)
            return self.get_status()

    def pause(self, actor: Optional[User] = None) -> Dict[str, Any]:
        """Pauses stream generation without resetting state."""
        with self._lock:
            if self.status != self.STATUS_RUNNING:
                return self.get_status()

            self.status = self.STATUS_PAUSED
            self._pause_event.clear()

            if actor and actor.is_authenticated:
                AuditLog.objects.create(user=actor, action='STREAM_PAUSE', details="Paused synthetic transaction stream")

            logger.info("LiveStreamManager paused")
            return self.get_status()

    def resume(self, actor: Optional[User] = None) -> Dict[str, Any]:
        """Resumes active generation from paused state."""
        with self._lock:
            if self.status != self.STATUS_PAUSED:
                return self.get_status()

            self.status = self.STATUS_RUNNING
            self._pause_event.set()

            if actor and actor.is_authenticated:
                AuditLog.objects.create(user=actor, action='STREAM_RESUME', details="Resumed synthetic transaction stream")

            logger.info("LiveStreamManager resumed")
            return self.get_status()

    def stop(self, actor: Optional[User] = None) -> Dict[str, Any]:
        """Terminates background worker thread and finalizes session."""
        with self._lock:
            if self.status == self.STATUS_STOPPED:
                return self.get_status()

            self.status = self.STATUS_STOPPED
            self._stop_event.set()
            self._pause_event.set()  # Unblock if currently paused

            if actor and actor.is_authenticated:
                AuditLog.objects.create(
                    user=actor,
                    action='STREAM_STOP',
                    details=f"Stopped synthetic stream. Processed {self.processed_count} events with {self.alerts_count} alerts."
                )

            logger.info("LiveStreamManager stopped")
            return self.get_status()

    def step(self, scenario: Optional[str] = None, actor: Optional[User] = None) -> Dict[str, Any]:
        """Generates and processes exactly one synthetic event on-demand (deterministic testing)."""
        if not getattr(settings, 'REAL_TIME_DEMO_ENABLED', True):
            raise PermissionDenied("Synthetic live stream is disabled in this environment.")

        target_scenario = scenario or self.scenario
        event = SyntheticTransactionGenerator.generate_event(scenario=target_scenario)
        result = EventProcessingService.process_transaction_event(event, actor=actor)

        self._record_processed_event(result)
        return result

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive operational state, telemetry, and buffer snapshots."""
        with self._lock:
            avg_lat = round(sum(self.latencies) / len(self.latencies), 2) if self.latencies else 0.0
            total_risk = sum(self.risk_distribution.values()) or 1
            risk_pcts = {
                k: round((v / total_risk) * 100, 1) for k, v in self.risk_distribution.items()
            }

            return {
                'status': self.status,
                'scenario': self.scenario,
                'rate_per_minute': self.rate_per_minute,
                'max_events': self.max_events,
                'processed_count': self.processed_count,
                'alerts_count': self.alerts_count,
                'high_risk_count': self.high_risk_count,
                'failure_count': self.failure_count,
                'average_latency_ms': avg_lat,
                'risk_distribution': self.risk_distribution,
                'risk_distribution_pct': risk_pcts,
                'environment': getattr(settings, 'ENVIRONMENT', 'DEVELOPMENT'),
                'is_demo_stream': True,
                'stream_label': 'LIVE MONITORING — SYNTHETIC DEMO STREAM',
                'started_at': self.started_at,
                'started_by': self.started_by_username,
            }

    def get_stream_delta(self, since_timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Returns events, alerts, and state updates produced since the client's last polling mark."""
        status = self.get_status()
        with self._lock:
            events_list = list(self.recent_events)
            alerts_list = list(self.recent_alerts)

        if since_timestamp:
            filtered_events = [e for e in events_list if e.get('timestamp', '') > since_timestamp]
            filtered_alerts = [a for a in alerts_list if a.get('timestamp', '') > since_timestamp]
        else:
            filtered_events = events_list[-15:]
            filtered_alerts = alerts_list[-10:]

        return {
            'status': status,
            'events': filtered_events,
            'alerts': filtered_alerts,
            'server_timestamp': timezone.now().isoformat(),
        }

    def _worker_loop(self):
        """Background thread executing controlled event generation and model pipeline evaluation."""
        logger.info("Worker thread starting for LiveStreamManager")
        while not self._stop_event.is_set():
            # Wait if paused
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            # Check maximum session boundary
            if self.processed_count >= self.max_events:
                logger.info("Stream reached max events limit (%d), auto-stopping", self.max_events)
                self.status = self.STATUS_STOPPED
                self._stop_event.set()
                break

            # Generate synthetic event
            try:
                event = SyntheticTransactionGenerator.generate_event(scenario=self.scenario)
                result = EventProcessingService.process_transaction_event(event)
                self._record_processed_event(result)
            except Exception as exc:
                logger.error("Error in stream worker iteration: %s", exc)
                self.failure_count += 1

            # Sleep interval calculated from rate
            interval_sec = max(0.5, 60.0 / float(self.rate_per_minute))
            # Split sleep into small chunks for responsive stopping/pausing
            steps = int(interval_sec / 0.1)
            for _ in range(steps):
                if self._stop_event.is_set() or not self._pause_event.is_set():
                    break
                time.sleep(0.1)

        logger.info("Worker thread completed for LiveStreamManager")

    def _record_processed_event(self, res: Dict[str, Any]):
        """Updates rolling buffers, distribution tallies, and latency counters."""
        with self._lock:
            if not res.get('success'):
                self.failure_count += 1
                return

            self.processed_count += 1
            lat = res.get('total_latency_ms', res.get('latency_ms', 0.0))
            if lat > 0:
                self.latencies.append(lat)

            risk_tier = res.get('risk_tier', 'LOW')
            if risk_tier in self.risk_distribution:
                self.risk_distribution[risk_tier] += 1

            if risk_tier in ('CRITICAL', 'HIGH') or res.get('decision_state') == 'FRAUD':
                self.high_risk_count += 1

            # Append event to recent events queue
            event_snapshot = {
                'event_id': res.get('event_id'),
                'transaction_id': res.get('transaction_id'),
                'source': res.get('source', 'SYNTHETIC'),
                'scenario': res.get('scenario', 'NORMAL'),
                'transaction_type': res.get('transaction_type'),
                'amount': res.get('amount'),
                'channel': res.get('channel'),
                'name_orig': res.get('name_orig'),
                'name_dest': res.get('name_dest'),
                'decision_state': res.get('decision_state'),
                'risk_score': res.get('risk_score'),
                'risk_tier': risk_tier,
                'fraud_probability': res.get('fraud_probability'),
                'anomaly_score': res.get('anomaly_score'),
                'latency_ms': lat,
                'status': res.get('status', 'COMPLETED'),
                'timestamp': res.get('timestamp') or timezone.now().isoformat(),
            }
            self.recent_events.append(event_snapshot)

            # Record any alerts generated
            alerts_generated = res.get('alerts', [])
            if alerts_generated:
                self.alerts_count += len(alerts_generated)
                for a in alerts_generated:
                    self.recent_alerts.append({
                        'alert_id': a.get('alert_id'),
                        'title': a.get('title'),
                        'severity': a.get('severity'),
                        'rule_name': a.get('rule_name'),
                        'transaction_id': res.get('transaction_id'),
                        'risk_score': res.get('risk_score'),
                        'timestamp': timezone.now().isoformat(),
                    })
