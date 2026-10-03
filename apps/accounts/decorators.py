from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages


def admin_required(view_func):
    """Ensure user is logged in and holds administrator or staff privileges."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.error(request, "Please log in to access this administrative feature.")
            return redirect('login')
        profile = getattr(request.user, 'profile', None)
        is_admin = request.user.is_staff or (profile and profile.is_administrator())
        if not is_admin:
            messages.error(request, "Access restricted. Administrator privileges required.")
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def analyst_or_admin_required(view_func):
    """Ensure user has risk analyst or admin role."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.error(request, "Please log in to continue.")
            return redirect('login')
        profile = getattr(request.user, 'profile', None)
        is_authorized = request.user.is_staff or (profile and profile.is_analyst_or_admin())
        if not is_authorized:
            messages.error(request, "Access restricted. Analyst or Administrator privileges required.")
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return _wrapped_view
