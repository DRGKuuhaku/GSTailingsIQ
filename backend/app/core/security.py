"""
security.py
------------
Authentication & authorization helpers for TailingsIQ backend.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Union

import logging
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Password hashing & token helpers
# ---------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()  # Bearer-token Authorization header


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plaintext password matches its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta
        if expires_delta
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Return JWT payload dict or None if signature/expiry invalid."""
    try:
        return jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
    except JWTError as exc:
        logger.warning("Token verification failed: %s", exc)
        return None


# ---------------------------------------------------------------------
# Current-user dependency
# ---------------------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """FastAPI dependency that returns the authenticated *User* object."""
    from ..models.user import User  # late import to avoid circular refs

    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


# ---------------------------------------------------------------------
# Role / permission utilities
# ---------------------------------------------------------------------

def require_roles(*allowed_roles):
    """
    Dependency factory that allows only users whose .role is in *allowed_roles*.
    Usage: Depends(require_roles("admin", "super_admin"))
    """
    def dependency(current_user=Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role permissions",
            )
        return current_user

    return Depends(dependency)


def check_user_permissions(user, required_permission: str) -> bool:
    """Return True if *user* possesses *required_permission*."""
    role_permissions = {
        "super_admin": ["*"],  # all permissions
        "admin": [
            "user_management",
            "system_config",
            "data_export",
            "compliance_full",
            "monitoring_full",
        ],
        "engineer_of_record": [
            "compliance_full",
            "monitoring_full",
            "data_export",
            "tsf_management",
            "risk_assessment",
        ],
        "tsf_operator": [
            "monitoring_read",
            "data_entry",
            "alerts_manage",
        ],
        "regulator": [
            "compliance_read",
            "monitoring_read",
            "reports_access",
        ],
        "management": [
            "reports_access",
            "monitoring_read",
            "compliance_read",
        ],
        "consultant": [
            "monitoring_read",
            "data_analysis",
            "reports_access",
        ],
        "viewer": [
            "monitoring_read",
            "reports_read",
        ],
    }

    perms = role_permissions.get(user.role, [])
    return "*" in perms or required_permission in perms


# ---------------------------------------------------------------------
# Back-compat helper expected by older route files
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Convenience alias expected by admin routes
# ---------------------------------------------------------------------

def get_current_active_user(current_user = Depends(get_current_user)):
    """
    In the current codebase we don't store an 'is_active' flag, so this
    simply returns the authenticated user.  Later you can add a check like:

        if not current_user.is_active:
            raise HTTPException(status_code=403, detail="Inactive user")

    For now it unblocks the import error.
    """
    return current_user

def check_permissions(
    allowed_roles: Optional[List[str]] = None,
    required_permission: Optional[str] = None,
):
    """
    Combined role/permission dependency.

    • If *allowed_roles* is provided → enforce role membership
    • If *required_permission* is provided → enforce fine-grained permission
    """
    def dependency(current_user=Depends(get_current_user)):
        # Role gate
        if allowed_roles and current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role permissions",
            )

        # Permission gate
        if required_permission and not check_user_permissions(
            current_user, required_permission
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied",
            )

        return current_user
    
    # ---------------------------------------------------------------------
# Permission-only dependency expected by documents & compliance routes
# ---------------------------------------------------------------------

def require_permission(required_permission: str):
    """
    Returns a FastAPI dependency that ensures the *current_user*
    possesses *required_permission*.  Usage:

        @router.post("/...")
        async def foo(current_user = Depends(require_permission("data_export"))):
            ...
    """
    from fastapi import Depends, HTTPException, status

    def dependency(current_user = Depends(get_current_user)):
        if not check_user_permissions(current_user, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied",
            )
        return current_user

    # Return a FastAPI dependency
    return Depends(dependency)
