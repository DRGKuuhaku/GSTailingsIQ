from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import jwt
import logging
from ..core.database import get_db
from ..core.config import settings
from ..models.user import User, UserResponse, UserUpdate, UserRole, UserStatus
from ..services.user_service import UserService
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")
user_service = UserService()

# Pydantic models for authentication
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class LoginRequest(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.PyJWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    username = verify_token(token)
    if username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user"""
    if current_user.status != UserStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Login endpoint"""
    try:
        user = user_service.authenticate_user(
            db, 
            login_data.username, 
            login_data.password,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.from_orm(user)
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Logout endpoint"""
    try:
        # Log the logout event
        user_service.log_user_action(
            db,
            current_user.id,
            "logout",
            {},
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"message": "Logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse.from_orm(current_user)

@router.put("/profile", response_model=UserResponse)
async def update_profile(
    profile_data: UserUpdate,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    try:
        # Users can only update their own profile
        update_data = profile_data.dict(exclude_unset=True)

        # Remove fields that users shouldn't be able to change themselves
        restricted_fields = ['role', 'status', 'facilities_access']
        for field in restricted_fields:
            update_data.pop(field, None)

        # Update allowed fields
        for field, value in update_data.items():
            if hasattr(current_user, field):
                setattr(current_user, field, value)

        db.commit()
        db.refresh(current_user)

        # Log profile update
        user_service.log_user_action(
            db,
            current_user.id,
            "profile_updated",
            {"updated_fields": list(update_data.keys())},
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return UserResponse.from_orm(current_user)
    except Exception as e:
        db.rollback()
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        # Verify current password
        if not user_service.pwd_context.verify(password_data.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

        # Update password
        current_user.hashed_password = user_service.pwd_context.hash(password_data.new_password)
        current_user.last_password_change = datetime.utcnow()
        current_user.failed_login_attempts = 0  # Reset failed attempts

        db.commit()

        # Log password change
        user_service.log_user_action(
            db,
            current_user.id,
            "password_changed",
            {},
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/request-password-reset")
async def request_password_reset(
    reset_request: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Request password reset"""
    try:
        user = db.query(User).filter(User.email == reset_request.email).first()

        if user:
            # Generate reset token
            import secrets
            reset_token = secrets.token_urlsafe(32)

            # Set reset token and expiry (1 hour)
            user.password_reset_token = reset_token
            user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)

            db.commit()

            # Log password reset request
            user_service.log_user_action(
                db,
                user.id,
                "password_reset_requested",
                {"email": reset_request.email},
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent")
            )

            # TODO: Send email with reset token
            # For now, return the token (in production, this should be sent via email)
            logger.info(f"Password reset requested for {reset_request.email}, token: {reset_token}")

        # Always return success to prevent email enumeration
        return {"message": "If the email exists, a password reset link has been sent"}
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset,
    request: Request,
    db: Session = Depends(get_db)
):
    """Reset password with token"""
    try:
        user = db.query(User).filter(
            User.password_reset_token == reset_data.token,
            User.password_reset_expires > datetime.utcnow()
        ).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        # Update password
        user.hashed_password = user_service.pwd_context.hash(reset_data.new_password)
        user.password_reset_token = None
        user.password_reset_expires = None
        user.last_password_change = datetime.utcnow()
        user.failed_login_attempts = 0  # Reset failed attempts

        db.commit()

        # Log password reset
        user_service.log_user_action(
            db,
            user.id,
            "password_reset_completed",
            {},
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return {"message": "Password reset successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Password reset error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
