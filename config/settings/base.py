import os
from pathlib import Path
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / '.env')

# Security settings
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-audit-phase2-fraud-detection-secret-key-2026')
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
ALLOWED_HOSTS = ['*']
CSRF_TRUSTED_ORIGINS = ['https://*.run.app', 'http://localhost:3000', 'http://127.0.0.1:3000']

# Real-Time Synthetic Monitoring Configuration & Safety Bounds
ENVIRONMENT = os.getenv('ENVIRONMENT', 'DEVELOPMENT').upper()
# Explicitly disable synthetic stream in production by default
REAL_TIME_DEMO_ENABLED = os.getenv('REAL_TIME_DEMO_ENABLED', 'True').lower() in ('true', '1', 't') if ENVIRONMENT != 'PRODUCTION' else False
MAX_SYNTHETIC_EVENTS_PER_SESSION = int(os.getenv('MAX_SYNTHETIC_EVENTS_PER_SESSION', '300'))
MAX_SYNTHETIC_RATE_PER_MINUTE = int(os.getenv('MAX_SYNTHETIC_RATE_PER_MINUTE', '60'))

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Modernized modular domain apps
    'apps.core.apps.CoreConfig',
    'apps.accounts.apps.AccountsConfig',
    'apps.transactions.apps.TransactionsConfig',
    'apps.fraud.apps.FraudConfig',
    'apps.models_registry.apps.ModelsRegistryConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'

# Database configuration (Defaults to SQLite for local development)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
_static_url = os.getenv('STATIC_URL', '/static/')
if not _static_url.startswith('/'):
    _static_url = '/' + _static_url
if not _static_url.endswith('/'):
    _static_url = _static_url + '/'
STATIC_URL = _static_url

STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (Uploaded datasets, generated reports)
_media_url = os.getenv('MEDIA_URL', '/media/')
if not _media_url.startswith('/'):
    _media_url = '/' + _media_url
if not _media_url.endswith('/'):
    _media_url = _media_url + '/'
MEDIA_URL = _media_url
MEDIA_ROOT = BASE_DIR / os.getenv('MEDIA_ROOT', 'media')

# ML model directory
MODEL_DIR = BASE_DIR / os.getenv('MODEL_DIR', 'model')

# File upload security limits (Default: 10MB)
MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 10485760))
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE
FILE_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Structured logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(levelname)s in %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

# Real-Time Synthetic Transaction Stream Settings (Phase 7)
REAL_TIME_DEMO_ENABLED = os.getenv('REAL_TIME_DEMO_ENABLED', 'True').lower() in ('true', '1', 't')
MAX_SYNTHETIC_EVENTS_PER_SESSION = int(os.getenv('MAX_SYNTHETIC_EVENTS_PER_SESSION', '300'))
MAX_SYNTHETIC_RATE_PER_MINUTE = int(os.getenv('MAX_SYNTHETIC_RATE_PER_MINUTE', '60'))

