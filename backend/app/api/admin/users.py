from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from ...core.database import get_db
from ...core.security import get_current_user, get_current_active_user
from ...models.user import User, UserCreate, UserUpdate, UserResponse, UserRole, UserStatus
from ...services.user_service import UserService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
user_service = UserService()

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    role: Optional[UserRole] = None,
    status: Optional[UserStatus] = None,
    organization: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all users (Admin/Super Admin only)"""
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    query = db.query(User)

    if role:
        query = query.filter(User.role == role.value)
    if status:
        query = query.filter(User.status == status.value)
    if organization:
        query = query.filter(User.organization.ilike(f"%{organization}%"))

    users = query.offset(skip).limit(limit).all()
    return users

@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new user (Admin/Super Admin only)"""
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    try:
        new_user = user_service.create_user(db, user_data, created_by=current_user.id)

        # Log user creation
        user_service.log_user_action(
            db, 
            current_user.id, 
            "created_user", 
            {
                "new_user_id": new_user.id,
                "new_user_username": new_user.username,
                "new_user_role": new_user.role
            },
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return new_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user by ID (Admin/Super Admin only)"""
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update user (Admin/Super Admin only)"""
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Prevent non-super-admin from modifying super-admin users
    if user.role == UserRole.SUPER_ADMIN.value and current_user.role != UserRole.SUPER_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify super admin users"
        )

    try:
        # Update user fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(user, field):
                if field == 'role' and isinstance(value, UserRole):
                    setattr(user, field, value.value)
                elif field == 'status' and isinstance(value, UserStatus):
                    setattr(user, field, value.value)
                else:
                    setattr(user, field, value)

        db.commit()
        db.refresh(user)

        # Log user update
        user_service.log_user_action(
            db,
            current_user.id,
            "updated_user",
            {
                "updated_user_id": user.id,
                "updated_user_username": user.username,
                "updated_fields": list(update_data.keys())
            },
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete/deactivate user (Super Admin only)"""
    if current_user.role != UserRole.SUPER_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admin can delete users"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Prevent deleting super admin users
    if user.role == UserRole.SUPER_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete super admin users"
        )

    try:
        # Soft delete - set status to inactive instead of hard delete
        user.status = UserStatus.INACTIVE.value
        db.commit()

        # Log user deletion
        user_service.log_user_action(
            db,
            current_user.id,
            "deleted_user",
            {
                "deleted_user_id": user.id,
                "deleted_user_username": user.username
            },
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return {"message": "User deactivated successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Reset user password (Admin/Super Admin only)"""
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    try:
        # Generate temporary password
        import secrets
        import string
        temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

        # Hash and update password
        user.hashed_password = user_service.pwd_context.hash(temp_password)
        user.failed_login_attempts = 0  # Reset failed attempts
        db.commit()

        # Log password reset
        user_service.log_user_action(
            db,
            current_user.id,
            "reset_user_password",
            {"reset_user_id": user.id, "reset_user_username": user.username},
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )

        return {"message": "Password reset successfully", "temporary_password": temp_password}
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting password: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
