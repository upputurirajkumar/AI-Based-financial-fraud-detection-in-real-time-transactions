import io
import csv
import logging
from typing import Dict, Any, List
import pandas as pd
from django.conf import settings
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class DatasetService:
    """Validates and processes transaction datasets."""

    ALLOWED_EXTENSIONS = ['.csv']

    @classmethod
    def validate_uploaded_file(cls, uploaded_file) -> None:
        """Validates extension and file size."""
        # 1. Extension check
        name = uploaded_file.name.lower()
        if not any(name.endswith(ext) for ext in cls.ALLOWED_EXTENSIONS):
            raise ValidationError("Invalid file type. Only CSV files (.csv) are supported.")

        # 2. Size limit
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 10485760)
        if uploaded_file.size > max_size:
            raise ValidationError(
                f"File size ({uploaded_file.size / (1024*1024):.1f}MB) exceeds "
                f"the allowable maximum of {max_size / (1024*1024):.0f}MB."
            )

    @classmethod
    def read_csv_safely(cls, uploaded_file) -> pd.DataFrame:
        """Safely reads CSV into pandas DataFrame handling encodings."""
        cls.validate_uploaded_file(uploaded_file)

        try:
            content = uploaded_file.read()
            # Try UTF-8 decoding, fallback to latin-1
            try:
                decoded = content.decode('utf-8')
            except UnicodeDecodeError:
                decoded = content.decode('latin-1')

            df = pd.read_csv(io.StringIO(decoded))
            if df.empty:
                raise ValidationError("The uploaded CSV dataset contains no records.")
            return df
        except ValidationError:
            raise
        except Exception as exc:
            logger.error("CSV parse error: %s", str(exc))
            raise ValidationError(f"Failed to parse CSV file: {str(exc)}") from exc
