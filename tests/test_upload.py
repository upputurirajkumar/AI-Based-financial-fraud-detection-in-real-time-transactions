import io
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.core.exceptions import ValidationError
from services.dataset_service import DatasetService


class UploadTests(TestCase):
    """Verifies file upload security, validation, and parsing."""

    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='adminuser',
            password='adminpassword123',
            is_staff=True
        )

    def test_non_csv_file_rejected(self):
        """Uploading non-CSV files must trigger a ValidationError."""
        fake_file = SimpleUploadedFile("malicious.exe", b"binary content", content_type="application/octet-stream")
        with self.assertRaises(ValidationError):
            DatasetService.validate_uploaded_file(fake_file)

    def test_empty_csv_rejected(self):
        """Empty CSV files must be rejected."""
        empty_file = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")
        with self.assertRaises(ValidationError):
            DatasetService.read_csv_safely(empty_file)

    def test_valid_csv_parsed_successfully(self):
        """Proper CSV content parsed into DataFrame without error."""
        csv_data = b"step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n1,PAYMENT,100.0,500.0,400.0,0.0,100.0\n"
        valid_file = SimpleUploadedFile("test_valid.csv", csv_data, content_type="text/csv")
        df = DatasetService.read_csv_safely(valid_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['amount'], 100.0)
