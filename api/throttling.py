"""
Custom throttling classes for rate limiting
"""
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle


class ChatRateThrottle(UserRateThrottle):
    """
    Rate limit for chat endpoints.
    Prevents abuse of AI chat which is resource-intensive.
    """
    scope = "chat"


class AuthRateThrottle(AnonRateThrottle):
    """
    Rate limit for authentication endpoints.
    Protects against brute force attacks on login/register.
    """
    scope = "auth"
