import os

# Default to development settings unless DJANGO_ENV is explicitly set to production
env = os.getenv('DJANGO_ENV', 'development')
if env == 'production':
    from .production import *
else:
    from .development import *
