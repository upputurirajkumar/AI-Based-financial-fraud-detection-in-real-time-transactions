from django.db import models
from apps.core.models import BaseTimestampedModel


class ModelRecord(BaseTimestampedModel):
    """Tracks registered machine learning models, versions, and performance metrics."""
    FRAMEWORK_SKLEARN = 'sklearn'
    FRAMEWORK_KERAS = 'keras'
    FRAMEWORK_ENSEMBLE = 'ensemble'

    FRAMEWORK_CHOICES = [
        (FRAMEWORK_SKLEARN, 'Scikit-Learn'),
        (FRAMEWORK_KERAS, 'TensorFlow / Keras'),
        (FRAMEWORK_ENSEMBLE, 'Hybrid Ensemble'),
    ]

    key = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20, default='1.0.0')
    framework = models.CharField(max_length=20, choices=FRAMEWORK_CHOICES)
    filename = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    description = models.TextField(blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} v{self.version} ({self.framework})"
