from django.db import models
from django.contrib.auth.models import User


class BaseTimestampedModel(models.Model):
    """Abstract base model tracking creation and update timestamps."""
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class AuditLog(BaseTimestampedModel):
    """Audit log recording critical actions, authentications, and administrative events."""
    ACTION_CHOICES = [
        ('LOGIN', 'User Login'),
        ('LOGOUT', 'User Logout'),
        ('REGISTRATION', 'User Registration'),
        ('DATASET_UPLOAD', 'Dataset Uploaded'),
        ('MODEL_LOAD', 'Model Loaded'),
        ('MODEL_EVAL', 'Model Evaluated'),
        ('BATCH_PREDICTION', 'Batch Prediction Executed'),
        ('SECURITY_ALERT', 'Security Violation Detected'),
    ]

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='audit_logs')
    action = models.CharField(max_length=50, choices=ACTION_CHOICES, db_index=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    details = models.TextField(blank=True)
    status_code = models.IntegerField(default=200)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['action', 'created_at']),
        ]

    def __str__(self):
        username = self.user.username if self.user else 'Anonymous'
        return f"[{self.created_at.strftime('%Y-%m-%d %H:%M:%S')}] {username} - {self.action}"
