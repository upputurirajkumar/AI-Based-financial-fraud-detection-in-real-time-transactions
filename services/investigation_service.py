import logging
from typing import Dict, Any, List, Optional
from django.utils import timezone
from django.db import transaction as db_transaction
from django.core.paginator import Paginator
from django.core.exceptions import ValidationError, PermissionDenied
from django.contrib.auth.models import User

from apps.fraud.models import InvestigationCase, InvestigationNote, EvidenceAttachment, FraudAlert
from apps.transactions.models import Transaction
from apps.core.models import AuditLog

logger = logging.getLogger(__name__)


class InvestigationService:
    """
    Case management, analyst collaboration, ground-truth resolution, and evidence tracking service.
    Enforces compliance and prevents unauthorized tampering with historical audit notes.
    """

    @classmethod
    def create_case(
        cls,
        title: str,
        opened_by: User,
        description: str = "",
        priority: str = InvestigationCase.PRIORITY_MEDIUM,
        assigned_analyst: Optional[User] = None,
        transaction_ids: Optional[List[str]] = None,
        alert_ids: Optional[List[str]] = None
    ) -> InvestigationCase:
        """Creates a new formal investigation case linking transactions and alerts."""
        with db_transaction.atomic():
            case = InvestigationCase.objects.create(
                title=title,
                description=description,
                priority=priority,
                opened_by=opened_by,
                assigned_analyst=assigned_analyst,
                status=InvestigationCase.STATUS_OPEN
            )

            if transaction_ids:
                tx_objs = Transaction.objects.filter(transaction_id__in=transaction_ids)
                case.transactions.set(tx_objs)

            if alert_ids:
                alert_objs = FraudAlert.objects.filter(alert_id__in=alert_ids)
                case.alerts.set(alert_objs)
                # Automatically move linked alerts to UNDER_REVIEW
                for alert in alert_objs:
                    if alert.status in (FraudAlert.STATUS_NEW, FraudAlert.STATUS_ACKNOWLEDGED):
                        alert.transition_to(FraudAlert.STATUS_UNDER_REVIEW, opened_by, f"Linked to case {case.case_id}")

            # System note
            InvestigationNote.objects.create(
                case=case,
                author=opened_by,
                note=f"Case created with priority {priority}. Linked {len(transaction_ids or [])} transactions and {len(alert_ids or [])} alerts.",
                is_system_generated=True
            )

            AuditLog.objects.create(
                user=opened_by,
                action='CASE_CREATE',
                details=f"Opened investigation case {case.case_id} ({case.title})"
            )

        return case

    @classmethod
    def add_note(cls, case_id: str, author: User, note_text: str) -> InvestigationNote:
        """Adds a permanent, chronological investigation note."""
        case = InvestigationCase.objects.filter(case_id=case_id).first()
        if not case:
            raise ValidationError(f"Investigation case '{case_id}' not found.")

        note = InvestigationNote.objects.create(
            case=case,
            author=author,
            note=note_text,
            is_system_generated=False
        )

        AuditLog.objects.create(
            user=author,
            action='CASE_NOTE',
            details=f"Added note on case {case_id}"
        )
        return note

    @classmethod
    def assign_case(cls, case_id: str, analyst: User, actor: User) -> InvestigationCase:
        """Assigns case to an authorized compliance analyst."""
        case = InvestigationCase.objects.filter(case_id=case_id).first()
        if not case:
            raise ValidationError(f"Investigation case '{case_id}' not found.")

        case.assigned_analyst = analyst
        if case.status == InvestigationCase.STATUS_OPEN:
            case.status = InvestigationCase.STATUS_UNDER_REVIEW
        case.save()

        InvestigationNote.objects.create(
            case=case,
            author=actor,
            note=f"Case assigned to {analyst.username}.",
            is_system_generated=True
        )

        AuditLog.objects.create(
            user=actor,
            action='CASE_ASSIGN',
            details=f"Assigned case {case_id} to {analyst.username}"
        )
        return case

    @classmethod
    def resolve_case(cls, case_id: str, resolution: str, resolution_notes: str, analyst: User) -> InvestigationCase:
        """
        Resolves investigation case with explicit ground-truth outcome.
        Only human compliance analysts or administrators can execute this action.
        """
        case = InvestigationCase.objects.filter(case_id=case_id).first()
        if not case:
            raise ValidationError(f"Investigation case '{case_id}' not found.")

        with db_transaction.atomic():
            case.resolve_case(resolution=resolution, analyst=analyst, notes=resolution_notes)

            # System note recording formal resolution
            InvestigationNote.objects.create(
                case=case,
                author=analyst,
                note=f"Case resolved as '{resolution}'. Notes: {resolution_notes}",
                is_system_generated=True
            )

            # Synchronize linked alerts to RESOLVED
            for alert in case.alerts.all():
                if alert.status != FraudAlert.STATUS_RESOLVED:
                    alert.transition_to(FraudAlert.STATUS_RESOLVED, analyst, f"Resolved via case {case.case_id} ({resolution})")

            AuditLog.objects.create(
                user=analyst,
                action='CASE_RESOLVE',
                details=f"Resolved case {case_id} as {resolution}"
            )

        return case

    @classmethod
    def list_cases(
        cls,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        resolution: Optional[str] = None,
        analyst_id: Optional[int] = None,
        page: int = 1,
        page_size: int = 25
    ) -> Dict[str, Any]:
        """Retrieves paginated list of investigation cases."""
        qs = InvestigationCase.objects.select_related('assigned_analyst', 'opened_by').prefetch_related('transactions', 'alerts').all()

        if status:
            qs = qs.filter(status=status)
        if priority:
            qs = qs.filter(priority=priority)
        if resolution:
            qs = qs.filter(resolution=resolution)
        if analyst_id:
            qs = qs.filter(assigned_analyst_id=analyst_id)

        paginator = Paginator(qs, page_size)
        current_page = paginator.get_page(page)

        records = []
        for c in current_page:
            records.append({
                'case_id': c.case_id,
                'title': c.title,
                'priority': c.priority,
                'status': c.status,
                'resolution': c.resolution,
                'assigned_analyst': c.assigned_analyst.username if c.assigned_analyst else None,
                'opened_by': c.opened_by.username,
                'transactions_count': c.transactions.count(),
                'alerts_count': c.alerts.count(),
                'opened_at': c.opened_at.isoformat(),
                'closed_at': c.closed_at.isoformat() if c.closed_at else None,
            })

        return {
            'total_count': paginator.count,
            'total_pages': paginator.num_pages,
            'current_page': current_page.number,
            'page_size': page_size,
            'cases': records
        }

    @classmethod
    def get_case_detail(cls, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full case details with notes, linked transactions, and alerts."""
        c = InvestigationCase.objects.filter(case_id=case_id).first()
        if not c:
            return None

        notes = []
        for n in c.notes.all().order_by('created_at'):
            notes.append({
                'id': n.id,
                'author': n.author.username if n.author else 'System',
                'note': n.note,
                'is_system_generated': n.is_system_generated,
                'created_at': n.created_at.isoformat()
            })

        evidence = []
        for e in c.evidence_attachments.all().order_by('-created_at'):
            evidence.append({
                'id': e.id,
                'filename': e.filename,
                'file_size_bytes': e.file_size_bytes,
                'mime_type': e.mime_type,
                'description': e.description,
                'uploaded_by': e.uploaded_by.username if e.uploaded_by else 'Unknown',
                'created_at': e.created_at.isoformat()
            })

        txs = []
        for t in c.transactions.all():
            latest_pred = t.predictions.first()
            txs.append({
                'transaction_id': t.transaction_id,
                'type': t.transaction_type,
                'amount': float(t.amount),
                'risk_score': latest_pred.risk_score if latest_pred else None,
                'decision_state': latest_pred.decision_state if latest_pred else None,
            })

        alerts = []
        for a in c.alerts.all():
            alerts.append({
                'alert_id': a.alert_id,
                'title': a.title,
                'severity': a.severity,
                'status': a.status,
                'rule_name': a.rule_name
            })

        return {
            'case_id': c.case_id,
            'title': c.title,
            'description': c.description,
            'priority': c.priority,
            'status': c.status,
            'resolution': c.resolution,
            'resolution_notes': c.resolution_notes,
            'assigned_analyst': c.assigned_analyst.username if c.assigned_analyst else None,
            'opened_by': c.opened_by.username,
            'opened_at': c.opened_at.isoformat(),
            'closed_at': c.closed_at.isoformat() if c.closed_at else None,
            'notes': notes,
            'evidence': evidence,
            'transactions': txs,
            'alerts': alerts
        }
