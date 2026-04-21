"""User model stubs for the test fixture."""

from typing import Optional


class User:
    """Represents an authenticated user."""

    def __init__(self, user_id: str, email: str, role: str = "user"):
        self.user_id = user_id
        self.email = email
        self.role = role

    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return self.role == "admin"

    def __repr__(self) -> str:
        return f"User(id={self.user_id}, email={self.email})"
