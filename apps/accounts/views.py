import logging
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from .forms import SecureUserRegistrationForm, UserLoginForm
from apps.core.models import AuditLog

logger = logging.getLogger(__name__)


def home_view(request):
    """Landing page view displaying system overview."""
    from apps.fraud.views import overview_dashboard_view
    return overview_dashboard_view(request)


def register_view(request):
    """Secure user registration enforcing standard user role."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = SecureUserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            AuditLog.objects.create(
                user=user,
                action='REGISTRATION',
                ip_address=request.META.get('REMOTE_ADDR'),
                details=f"New user registered: {user.username}"
            )
            logger.info("New user registered successfully: %s", user.username)
            messages.success(request, "Registration successful! You may now log in.")
            return redirect('login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.title()}: {error}")
    else:
        form = SecureUserRegistrationForm()

    return render(request, 'register.html', {'form': form})


def login_view(request):
    """User authentication with generic error handling to prevent user enumeration."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)

            if user is not None and user.is_active:
                login(request, user)
                AuditLog.objects.create(
                    user=user,
                    action='LOGIN',
                    ip_address=request.META.get('REMOTE_ADDR'),
                    details=f"User {user.username} logged in successfully."
                )
                logger.info("User logged in: %s", username)
                messages.success(request, "Login successful.")
                next_url = request.GET.get('next', '/')
                return redirect(next_url)
            else:
                AuditLog.objects.create(
                    action='SECURITY_ALERT',
                    ip_address=request.META.get('REMOTE_ADDR'),
                    details=f"Failed login attempt for username: {username}",
                    status_code=401
                )
                logger.warning("Failed login attempt for username: %s", username)
                messages.error(request, "Invalid username or password.")
    else:
        form = UserLoginForm()

    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """Session termination view."""
    if request.user.is_authenticated:
        AuditLog.objects.create(
            user=request.user,
            action='LOGOUT',
            ip_address=request.META.get('REMOTE_ADDR'),
            details=f"User {request.user.username} logged out."
        )
        logger.info("User logged out: %s", request.user.username)
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('login')
