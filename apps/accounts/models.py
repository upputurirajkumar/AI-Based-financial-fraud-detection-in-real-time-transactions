from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """User profile extending default User with Role-Based Access Control (RBAC)."""
    ROLE_ADMIN = 'ADMIN'
    ROLE_ANALYST = 'ANALYST'
    ROLE_AUDITOR = 'AUDITOR'
    ROLE_USER = 'USER'

    ROLE_CHOICES = [
        (ROLE_ADMIN, 'System Administrator'),
        (ROLE_ANALYST, 'Fraud Risk Analyst'),
        (ROLE_AUDITOR, 'Compliance Auditor'),
        (ROLE_USER, 'Standard User'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_USER, db_index=True)
    mobile = models.CharField(max_length=20, blank=True)
    department = models.CharField(max_length=100, blank=True, default='Risk Management')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_administrator(self):
        return self.role == self.ROLE_ADMIN or self.user.is_superuser or self.user.is_staff

    def is_analyst_or_admin(self):
        return self.role in (self.ROLE_ADMIN, self.ROLE_ANALYST) or self.user.is_staff

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        role = UserProfile.ROLE_ADMIN if (instance.is_staff or instance.is_superuser) else UserProfile.ROLE_USER
        UserProfile.objects.create(user=instance, role=role)
    else:
        if hasattr(instance, 'profile'):
            instance.profile.save()
