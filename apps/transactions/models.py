import uuid
from django.db import models
from django.contrib.auth.models import User
from apps.core.models import BaseTimestampedModel


class DatasetRecord(BaseTimestampedModel):
    """Tracks uploaded transaction datasets and their processing status."""
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    filename = models.CharField(max_length=255)
    file_size_bytes = models.BigIntegerField()
    total_records = models.IntegerField(default=0)
    processed_records = models.IntegerField(default=0)
    failed_records = models.IntegerField(default=0)
    is_processed = models.BooleanField(default=False)
    data_quality_summary = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.filename} ({self.total_records} rows)"


class Transaction(BaseTimestampedModel):
    """Financial transaction entity representing real-time ledger operations."""
    TYPE_PAYMENT = 'PAYMENT'
    TYPE_TRANSFER = 'TRANSFER'
    TYPE_CASH_OUT = 'CASH_OUT'
    TYPE_DEBIT = 'DEBIT'
    TYPE_CASH_IN = 'CASH_IN'

    TYPE_CHOICES = [
        (TYPE_PAYMENT, 'Payment'),
        (TYPE_TRANSFER, 'Transfer'),
        (TYPE_CASH_OUT, 'Cash Out'),
        (TYPE_DEBIT, 'Debit'),
        (TYPE_CASH_IN, 'Cash In'),
    ]

    CHANNEL_CHOICES = [
        ('ONLINE', 'Online Banking / Web'),
        ('MOBILE', 'Mobile Banking App'),
        ('ATM', 'Automated Teller Machine'),
        ('POS', 'Point of Sale'),
        ('WIRE', 'Wire Transfer Desk'),
    ]

    transaction_id = models.CharField(max_length=64, unique=True, db_index=True, blank=True)
    dataset = models.ForeignKey(DatasetRecord, on_delete=models.SET_NULL, null=True, blank=True, related_name='transactions')
    step = models.IntegerField(default=1, db_index=True)
    transaction_type = models.CharField(max_length=20, choices=TYPE_CHOICES, db_index=True)
    amount = models.DecimalField(max_digits=18, decimal_places=2, db_index=True)
    currency = models.CharField(max_length=10, default='USD')
    channel = models.CharField(max_length=30, choices=CHANNEL_CHOICES, default='ONLINE', db_index=True)

    name_orig = models.CharField(max_length=64, db_index=True)
    old_balance_orig = models.DecimalField(max_digits=18, decimal_places=2)
    new_balance_orig = models.DecimalField(max_digits=18, decimal_places=2)

    name_dest = models.CharField(max_length=64, db_index=True)
    old_balance_dest = models.DecimalField(max_digits=18, decimal_places=2)
    new_balance_dest = models.DecimalField(max_digits=18, decimal_places=2)

    # Real-Time Event & Synthetic Demo Metadata
    SOURCE_SYNTHETIC = 'SYNTHETIC'
    SOURCE_INTERNAL = 'INTERNAL'
    SOURCE_EXTERNAL = 'EXTERNAL'

    SOURCE_CHOICES = [
        (SOURCE_SYNTHETIC, 'Synthetic Demo Stream'),
        (SOURCE_INTERNAL, 'Internal Ledger'),
        (SOURCE_EXTERNAL, 'External Ingested'),
    ]

    STATUS_RECEIVED = 'RECEIVED'
    STATUS_PROCESSING = 'PROCESSING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'

    STATUS_CHOICES = [
        (STATUS_RECEIVED, 'Received'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_INTERNAL, db_index=True)
    is_synthetic = models.BooleanField(default=False, db_index=True)
    event_id = models.CharField(max_length=64, blank=True, null=True, unique=True, db_index=True)
    scenario_tag = models.CharField(max_length=40, blank=True, default='', db_index=True)
    latency_ms = models.FloatField(default=0.0)
    processing_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_COMPLETED, db_index=True)

    # Dataset ground truth flags if known from historical labeled source
    is_fraud_flag = models.BooleanField(default=False, db_index=True)
    is_flagged_fraud = models.BooleanField(default=False)
    raw_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['transaction_type', 'amount']),
            models.Index(fields=['name_orig']),
            models.Index(fields=['name_dest']),
            models.Index(fields=['created_at', 'transaction_type']),
        ]

    def save(self, *args, **kwargs):
        if not self.transaction_id:
            self.transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
        super().save(*args, **kwargs)

    def to_inference_dict(self) -> dict:
        """Converts database transaction into canonical dictionary required by FraudEngine."""
        return {
            'transaction_id': self.transaction_id,
            'step': self.step,
            'type': self.transaction_type,
            'amount': float(self.amount),
            'nameOrig': self.name_orig,
            'oldbalanceOrg': float(self.old_balance_orig),
            'newbalanceOrig': float(self.new_balance_orig),
            'nameDest': self.name_dest,
            'oldbalanceDest': float(self.old_balance_dest),
            'newbalanceDest': float(self.new_balance_dest),
        }

    def __str__(self):
        return f"[{self.transaction_id}] {self.transaction_type} ${self.amount:,.2f} ({self.name_orig} -> {self.name_dest})"
