from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


class SecureUserRegistrationForm(forms.ModelForm):
    """Secure registration form preventing privilege escalation."""
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Enter Password', 'class': 'form-input'}),
        min_length=8
    )
    cnfm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Confirm Password', 'class': 'form-input'}),
        label="Confirm Password"
    )
    name = forms.CharField(max_length=150, required=True, widget=forms.TextInput(attrs={'placeholder': 'Full Name'}))
    mobile = forms.CharField(max_length=20, required=False, widget=forms.TextInput(attrs={'placeholder': 'Mobile Number'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'name', 'mobile', 'password']
        widgets = {
            'username': forms.TextInput(attrs={'placeholder': 'Choose Username'}),
            'email': forms.EmailInput(attrs={'placeholder': 'Email Address'}),
        }

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("Email address is already registered.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        cnfm_password = cleaned_data.get("cnfm_password")

        if password and cnfm_password and password != cnfm_password:
            self.add_error('cnfm_password', "Passwords do not match.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        user.first_name = self.cleaned_data.get("name", "")
        # NEVER grant staff or superuser via public registration
        user.is_staff = False
        user.is_superuser = False
        if commit:
            user.save()
            profile = user.profile
            profile.mobile = self.cleaned_data.get("mobile", "")
            profile.save()
        return user


class UserLoginForm(forms.Form):
    """User authentication form with generic error messaging to prevent enumeration."""
    username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Username'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Password'}))
