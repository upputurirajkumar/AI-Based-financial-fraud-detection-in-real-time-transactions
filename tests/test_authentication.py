from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from apps.accounts.models import UserProfile


class AuthenticationTests(TestCase):
    """Verifies registration, login, logout, and anti-escalation security."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpassword123',
            email='testuser@example.com'
        )

    def test_user_profile_automatically_created(self):
        """User profile must be generated via signal and assigned standard USER role."""
        self.assertTrue(hasattr(self.user, 'profile'))
        self.assertEqual(self.user.profile.role, UserProfile.ROLE_USER)
        self.assertFalse(self.user.is_staff)

    def test_user_login_success(self):
        """Valid credentials authenticate and open user session."""
        response = self.client.post(reverse('login'), {
            'username': 'testuser',
            'password': 'testpassword123'
        })
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, '/')

    def test_user_login_invalid_password(self):
        """Invalid credentials rejected with error message."""
        response = self.client.post(reverse('login'), {
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid username or password")

    def test_registration_blocks_privilege_escalation(self):
        """Submitting role=admin during registration must be ignored; user remains standard."""
        response = self.client.post(reverse('register'), {
            'username': 'newhacker',
            'name': 'Hacker Name',
            'email': 'hacker@example.com',
            'mobile': '1234567890',
            'password': 'ComplexPassword123!',
            'cnfm_password': 'ComplexPassword123!',
            'role': 'admin'  # Malicious payload trying to escalate
        })
        self.assertEqual(response.status_code, 302)
        new_user = User.objects.get(username='newhacker')
        self.assertFalse(new_user.is_staff)
        self.assertFalse(new_user.is_superuser)
        self.assertEqual(new_user.profile.role, UserProfile.ROLE_USER)
