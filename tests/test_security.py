from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse


class SecurityAuthorizationTests(TestCase):
    """Verifies RBAC protection across administrative routes."""

    def setUp(self):
        self.client = Client()
        self.standard_user = User.objects.create_user(
            username='regular',
            password='regularpassword123',
            is_staff=False
        )
        self.admin_user = User.objects.create_user(
            username='admin',
            password='adminpassword123',
            is_staff=True
        )

    def test_anonymous_user_redirected_from_admin_views(self):
        """Unauthenticated requests must be redirected to login."""
        for route_name in ['upload', 'dnn', 'rfc', 'prediction']:
            response = self.client.get(reverse(route_name))
            self.assertEqual(response.status_code, 302)
            self.assertIn('/login', response.url)

    def test_standard_user_blocked_from_admin_views(self):
        """Standard users must be redirected away from admin views."""
        self.client.login(username='regular', password='regularpassword123')
        for route_name in ['upload', 'dnn', 'rfc']:
            response = self.client.get(reverse(route_name))
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.url, '/')

    def test_standard_user_allowed_on_prediction_view(self):
        """Standard user has access to the prediction testing interface."""
        self.client.login(username='regular', password='regularpassword123')
        response = self.client.get(reverse('prediction'))
        self.assertEqual(response.status_code, 200)

    def test_admin_user_allowed_on_all_views(self):
        """Admin user has full access to upload, dnn, rfc, and prediction."""
        self.client.login(username='admin', password='adminpassword123')
        for route_name in ['upload', 'dnn', 'rfc', 'prediction']:
            response = self.client.get(reverse(route_name))
            self.assertEqual(response.status_code, 200)
