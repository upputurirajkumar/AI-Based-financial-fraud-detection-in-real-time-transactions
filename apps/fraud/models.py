import uuid
import hashlib
from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from apps.core.models import BaseTimestampedModel
from apps.transactions.models import Transaction


class Prediction(BaseTimestampedModel):
    """
    Stores versioned model inference results and risk metadata for financial transactions.
    Supports multi-model historical traceability where one transaction can have predictions
    from multiple distinct model versions.
    """
    DECISION_LEGITIMATE = 'LEGITIMATE'
    DECISION_SUSPICIOUS = 'SUSPICIOUS'
    DECISION_FRAUD = 'FRAUD'

    DECISION_CHOICES = [
        (DECISION_LEGITIMATE, 'Legitimate'),
        (DECISION_SUSPICIOUS, 'Suspicious'),
        (DECISION_FRAUD, 'Fraudulent'),
    ]

    prediction_id = models.CharField(max_length=64, unique=True, db_index=True, blank=True)
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, null=True, blank=True, related_name='predictions')
    model_name = models.CharField(max_length=64, default='Random Forest Classifier', db_index=True)
    model_version = models.CharField(max_length=32, default='1.0.0', db_index=True)
    decision_threshold = models.FloatField(default=0.23)

    # Core ML Outputs
    fraud_probability = models.FloatField(null=True, blank=True)
    risk_score = models.FloatField(default=0.0, db_index=True)
    risk_tier = models.CharField(max_length=20, default='LOW', db_index=True)
    decision_state = models.CharField(max_length=20, choices=DECISION_CHOICES, default=DECISION_LEGITIMATE, db_index=True)
    is_fraud = models.BooleanField(default=False, db_index=True)

    # Unsupervised Anomaly Detection
    anomaly_score = models.FloatField(null=True, blank=True)
    is_anomalous = models.BooleanField(default=False)

    # Explainable AI Metadata
    explanation = models.TextField(blank=True)
    raw_explanation_json = models.JSONField(default=dict, blank=True)

    # Performance & Diagnostics
    inference_duration_ms = models.FloatField(null=True, blank=True)
    evaluated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='evaluations')

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['model_name', 'model_version']),
            models.Index(fields=['risk_tier', 'created_at']),
            models.Index(fields=['decision_state', 'created_at']),
        ]

    def save(self, *args, **kwargs):
        if not self.prediction_id:
            self.prediction_id = f"PRD-{uuid.uuid4().hex[:12].upper()}"
        super().save(*args, **kwargs)

    def __str__(self):
        tx_ref = self.transaction.transaction_id if self.transaction else "Standalone"
        return f"[{self.prediction_id}] {tx_ref} -> {self.decision_state} ({self.risk_score}/100, {self.model_name} v{self.model_version})"


class FraudAlert(BaseTimestampedModel):
    """
    Actionable fraud alert candidate generated deterministically by business and ML rules.
    Maintains a strict state machine and prevents duplicate alerts via deduplication hashing.
    """
    SEVERITY_INFO = 'INFO'
    SEVERITY_LOW = 'LOW'
    SEVERITY_MEDIUM = 'MEDIUM'
    SEVERITY_HIGH = 'HIGH'
    SEVERITY_CRITICAL = 'CRITICAL'

    SEVERITY_CHOICES = [
        (SEVERITY_INFO, 'Informational'),
        (SEVERITY_LOW, 'Low'),
        (SEVERITY_MEDIUM, 'Medium'),
        (SEVERITY_HIGH, 'High'),
        (SEVERITY_CRITICAL, 'Critical'),
    ]

    STATUS_NEW = 'NEW'
    STATUS_ACKNOWLEDGED = 'ACKNOWLEDGED'
    STATUS_UNDER_REVIEW = 'UNDER_REVIEW'
    STATUS_ESCALATED = 'ESCALATED'
    STATUS_RESOLVED = 'RESOLVED'
    STATUS_DISMISSED = 'DISMISSED'

    # Kept for backward compatibility with Phase 1-2 tests
    STATUS_OPEN = 'NEW'

    STATUS_CHOICES = [
        (STATUS_NEW, 'New'),
        (STATUS_ACKNOWLEDGED, 'Acknowledged'),
        (STATUS_UNDER_REVIEW, 'Under Review'),
        (STATUS_ESCALATED, 'Escalated'),
        (STATUS_RESOLVED, 'Resolved'),
        (STATUS_DISMISSED, 'Dismissed'),
    ]

    VALID_TRANSITIONS = {
        STATUS_NEW: [STATUS_ACKNOWLEDGED, STATUS_UNDER_REVIEW, STATUS_DISMISSED],
        STATUS_ACKNOWLEDGED: [STATUS_UNDER_REVIEW, STATUS_ESCALATED, STATUS_DISMISSED],
        STATUS_UNDER_REVIEW: [STATUS_ESCALATED, STATUS_RESOLVED, STATUS_DISMISSED],
        STATUS_ESCALATED: [STATUS_UNDER_REVIEW, STATUS_RESOLVED, STATUS_DISMISSED],
        STATUS_RESOLVED: [STATUS_UNDER_REVIEW], # Reopen
        STATUS_DISMISSED: [STATUS_UNDER_REVIEW], # Reopen
    }

    alert_id = models.CharField(max_length=64, unique=True, db_index=True, blank=True)
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, null=True, blank=True, related_name='alerts')
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='alerts')

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    rule_name = models.CharField(max_length=100, default='HIGH_RISK_THRESHOLD', db_index=True)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default=SEVERITY_HIGH, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_NEW, db_index=True)

    deduplication_hash = models.CharField(max_length=64, unique=True, db_index=True, blank=True)
    assigned_to = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_alerts')

    reviewed_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'severity']),
            models.Index(fields=['assigned_to', 'status']),
        ]

    def save(self, *args, **kwargs):
        if not self.alert_id:
            self.alert_id = f"ALT-{uuid.uuid4().hex[:12].upper()}"
        if not self.deduplication_hash:
            tx_id = self.transaction.transaction_id if self.transaction else (self.prediction.prediction_id if self.prediction else "NONE")
            model_ver = self.prediction.model_version if self.prediction else "1.0.0"
            raw_key = f"{tx_id}:{model_ver}:{self.rule_name}"
            self.deduplication_hash = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()
        super().save(*args, **kwargs)

    def transition_to(self, new_status: str, user: User, notes: str = ""):
        """Enforces deterministic state transitions for alert lifecycle."""
        if new_status == self.status:
            return

        valid_targets = self.VALID_TRANSITIONS.get(self.status, [])
        if new_status not in valid_targets:
            raise ValidationError(f"Invalid alert state transition from '{self.status}' to '{new_status}'. Allowed: {valid_targets}")

        from django.utils import timezone
        self.status = new_status
        if new_status in (self.STATUS_ACKNOWLEDGED, self.STATUS_UNDER_REVIEW) and not self.reviewed_at:
            self.reviewed_at = timezone.now()
        if new_status in (self.STATUS_RESOLVED, self.STATUS_DISMISSED):
            self.resolved_at = timezone.now()

        if notes:
            timestamp_str = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
            append_note = f"\n[{timestamp_str}] {user.username} transitioned alert to {new_status}: {notes}"
            self.notes = (self.notes + append_note).strip()

        self.save()

    def __str__(self):
        return f"[{self.alert_id}] {self.severity} Alert: {self.title} [{self.status}]"


class InvestigationCase(BaseTimestampedModel):
    """
    Formal fraud investigation case managed by compliance and fraud operations analysts.
    Ground-truth confirmation or dismissal happens strictly through authorized human review.
    """
    PRIORITY_LOW = 'LOW'
    PRIORITY_MEDIUM = 'MEDIUM'
    PRIORITY_HIGH = 'HIGH'
    PRIORITY_URGENT = 'URGENT'

    PRIORITY_CHOICES = [
        (PRIORITY_LOW, 'Low'),
        (PRIORITY_MEDIUM, 'Medium'),
        (PRIORITY_HIGH, 'High'),
        (PRIORITY_URGENT, 'Urgent'),
    ]

    STATUS_OPEN = 'OPEN'
    STATUS_UNDER_REVIEW = 'UNDER_REVIEW'
    STATUS_ESCALATED = 'ESCALATED'
    STATUS_RESOLVED = 'RESOLVED'
    STATUS_CLOSED = 'CLOSED'

    STATUS_CHOICES = [
        (STATUS_OPEN, 'Open'),
        (STATUS_UNDER_REVIEW, 'Under Review'),
        (STATUS_ESCALATED, 'Escalated'),
        (STATUS_RESOLVED, 'Resolved'),
        (STATUS_CLOSED, 'Closed'),
    ]

    RESOLUTION_UNRESOLVED = 'UNRESOLVED'
    RESOLUTION_CONFIRMED_FRAUD = 'CONFIRMED_FRAUD'
    RESOLUTION_FALSE_POSITIVE = 'FALSE_POSITIVE'
    RESOLUTION_SUSPICIOUS_ACTIVITY = 'SUSPICIOUS_ACTIVITY'
    RESOLUTION_NO_ISSUE = 'NO_ISSUE_FOUND'
    RESOLUTION_INCONCLUSIVE = 'INCONCLUSIVE'

    RESOLUTION_CHOICES = [
        (RESOLUTION_UNRESOLVED, 'Unresolved / In Progress'),
        (RESOLUTION_CONFIRMED_FRAUD, 'Confirmed Fraud (Ground Truth Positive)'),
        (RESOLUTION_FALSE_POSITIVE, 'False Positive (Legitimate Transaction)'),
        (RESOLUTION_SUSPICIOUS_ACTIVITY, 'Suspicious Activity (SAR Filed / Account Monitored)'),
        (RESOLUTION_NO_ISSUE, 'No Issue Found'),
        (RESOLUTION_INCONCLUSIVE, 'Inconclusive / Insufficient Evidence'),
    ]

    case_id = models.CharField(max_length=64, unique=True, db_index=True, blank=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default=PRIORITY_MEDIUM, db_index=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_OPEN, db_index=True)
    resolution = models.CharField(max_length=30, choices=RESOLUTION_CHOICES, default=RESOLUTION_UNRESOLVED, db_index=True)

    assigned_analyst = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='assigned_cases')
    opened_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='opened_cases')

    transactions = models.ManyToManyField(Transaction, blank=True, related_name='investigations')
    alerts = models.ManyToManyField(FraudAlert, blank=True, related_name='investigations')

    opened_at = models.DateTimeField(auto_now_add=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'priority']),
            models.Index(fields=['assigned_analyst', 'status']),
            models.Index(fields=['resolution', 'closed_at']),
        ]

    def save(self, *args, **kwargs):
        if not self.case_id:
            self.case_id = f"CASE-{uuid.uuid4().hex[:12].upper()}"
        super().save(*args, **kwargs)

    def resolve_case(self, resolution: str, analyst: User, notes: str):
        """Resolves investigation case with explicit ground-truth outcome."""
        from django.utils import timezone
        valid_resolutions = [r[0] for r in self.RESOLUTION_CHOICES if r[0] != self.RESOLUTION_UNRESOLVED]
        if resolution not in valid_resolutions:
            raise ValidationError(f"Invalid case resolution '{resolution}'. Must be one of {valid_resolutions}.")

        self.resolution = resolution
        self.status = self.STATUS_RESOLVED
        self.closed_at = timezone.now()
        timestamp_str = self.closed_at.strftime('%Y-%m-%d %H:%M:%S')
        self.resolution_notes = (
            f"[{timestamp_str}] Resolved as {resolution} by {analyst.username}:\n{notes}\n" + self.resolution_notes
        ).strip()
        self.save()

    def __str__(self):
        return f"[{self.case_id}] {self.title} ({self.status} - {self.resolution})"


class InvestigationNote(BaseTimestampedModel):
    """Immutable audit trail notes recorded by compliance analysts during an investigation."""
    case = models.ForeignKey(InvestigationCase, on_delete=models.CASCADE, related_name='notes')
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='investigation_notes')
    note = models.TextField()
    is_system_generated = models.BooleanField(default=False)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        author_name = self.author.username if self.author else "System"
        return f"Note by {author_name} on {self.case.case_id} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class EvidenceAttachment(BaseTimestampedModel):
    """Secure metadata record for evidentiary artifacts attached to an investigation case."""
    case = models.ForeignKey(InvestigationCase, on_delete=models.CASCADE, related_name='evidence_attachments')
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='uploaded_evidence')
    filename = models.CharField(max_length=255)
    file_size_bytes = models.BigIntegerField()
    file_hash = models.CharField(max_length=64)  # SHA-256
    mime_type = models.CharField(max_length=100, default='application/octet-stream')
    description = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.case.case_id}] Evidence: {self.filename} ({self.file_size_bytes} bytes)"
