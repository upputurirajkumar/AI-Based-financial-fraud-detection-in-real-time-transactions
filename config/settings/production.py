import os
from .base import *

DEBUG = False

# Production security requirements
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'
SECURE_CONTENT_TYPE_NOSNIFF = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True

# Production PostgreSQL Database if DATABASE_URL provided
if os.getenv('DATABASE_URL') and not os.getenv('DATABASE_URL').startswith('sqlite'):
    import dj_database_url
    DATABASES['default'] = dj_database_url.config(
        default=os.getenv('DATABASE_URL'),
        conn_max_age=600,
        ssl_require=True
    )
