"""Custom exception types for the auth module."""


class TokenError(Exception):
    """Raised when JWT token validation fails."""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails for any reason."""
    pass
