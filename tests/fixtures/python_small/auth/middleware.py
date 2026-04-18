"""Authentication middleware for JWT-based auth."""

import jwt
from datetime import datetime
from typing import Optional

from .models import User
from .exceptions import TokenError, AuthenticationError

__all__ = ["AuthMiddleware", "validate_token"]

MAX_TOKEN_AGE = 3600
DEFAULT_ALGORITHM = "HS256"


class AuthMiddleware:
    """ASGI middleware for JWT authentication."""

    def __init__(self, app, secret: str, algorithm: str = "HS256"):
        self.app = app
        self.secret = secret
        self.algorithm = algorithm

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            token = self._extract_token(scope)
            if token:
                scope["user"] = validate_token(token, self.secret)
        await self.app(scope, receive, send)

    def _extract_token(self, scope) -> Optional[str]:
        """Extract bearer token from Authorization header."""
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            return auth[7:]
        return None


def validate_token(token: str, secret: str) -> Optional[User]:
    """Validate a JWT token and return the associated user.

    Checks expiry, signature, and required claims.
    Returns None if token is invalid.
    """
    try:
        payload = decode_jwt(token, secret)
        if not payload:
            return None
        user_id = payload.get("sub")
        if not user_id:
            return None
        return get_user(user_id)
    except TokenError:
        return None


def decode_jwt(token: str, secret: str) -> Optional[dict]:
    """Decode and verify a JWT token payload."""
    try:
        return jwt.decode(token, secret, algorithms=[DEFAULT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise TokenError("Token expired")
    except jwt.InvalidTokenError:
        raise TokenError("Invalid token")


def get_user(user_id: str) -> Optional[User]:
    """Fetch user from database by ID."""
    # Database lookup
    pass


@staticmethod
def _helper():
    """Internal helper, not exported."""
    pass
