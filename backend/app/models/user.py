from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base
from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ENGINEER_OF_RECORD = "engineer_of_record"
    TSF_OPERATOR = "tsf_operator"
    REGULATOR = "regulator"
    MANAGEMENT = "management"
    CONSULTANT = "consultant"
    VIEWER = "viewer"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.VIEWER.value)
    status = Column(String(20), nullable=False, default=UserStatus.PENDING.value)
    
    # Profile Information
    organization = Column(String(255))
    position = Column(String(255))
    phone = Column(String(20))
    license_number = Column(String(100))
    
    # Access Control
    facilities_access = Column(JSON, default=[])
    permissions = Column(JSON, default={})
    
    # Security
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime(timezone=True))
    two_factor_enabled = Column(Boolean, default=False)
    
    # Audit Trail
    created_by = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_password_change = Column(DateTime(timezone=True))
    
    # Relationships
    audit_logs = relationship("UserAuditLog", back_populates="user")

class UserAuditLog(Base):
    __tablename__ = "user_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(100), nullable=False)
    details = Column(JSON, default={})
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="audit_logs")

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    role: UserRole
    organization: Optional[str] = None
    position: Optional[str] = None
    phone: Optional[str] = None
    license_number: Optional[str] = None
    facilities_access: List[str] = []

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    organization: Optional[str] = None
    position: Optional[str] = None
    phone: Optional[str] = None
    license_number: Optional[str] = None
    facilities_access: Optional[List[str]] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    status: UserStatus
    organization: Optional[str]
    position: Optional[str]
    phone: Optional[str]
    license_number: Optional[str]
    facilities_access: List[str]
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True
