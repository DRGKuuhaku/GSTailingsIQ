from typing import Optional
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import secrets
import logging
from ..models.user import User, UserAuditLog, UserCreate, UserUpdate, UserRole, UserStatus
from typing import Dict, List, Optional, Any  # ← add Dict (and others you might use)

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def create_user(self, db: Session, user_data: UserCreate, created_by: int = None) -> User:
        """Create a new user with proper validation and security"""
        try:
            # Check if username or email already exists
            existing_user = db.query(User).filter(
                (User.username == user_data.username) | 
                (User.email == user_data.email)
            ).first()
            
            if existing_user:
                if existing_user.username == user_data.username:
                    raise ValueError("Username already exists")
                else:
                    raise ValueError("Email already exists")
            
            # Hash password
            hashed_password = self.pwd_context.hash(user_data.password)
            
            # Create user
            db_user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                role=user_data.role.value,
                organization=user_data.organization,
                position=user_data.position,
                phone=user_data.phone,
                license_number=user_data.license_number,
                facilities_access=user_data.facilities_access,
                created_by=created_by,
                status=UserStatus.ACTIVE.value
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            # Log user creation
            self.log_user_action(db, db_user.id, "user_created", {
                "created_by": created_by,
                "role": user_data.role.value
            })
            
            logger.info(f"User created: {user_data.username} (ID: {db_user.id})")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    def authenticate_user(self, db: Session, username: str, password: str, 
                         ip_address: str = None, user_agent: str = None) -> Optional[User]:
        """Authenticate user with security measures"""
        try:
            user = db.query(User).filter(User.username == username).first()
            
            if not user:
                logger.warning(f"Login attempt with non-existent username: {username}")
                return None
            
            # Check if account is locked
            if user.failed_login_attempts >= self.max_failed_attempts:
                if user.last_login and (datetime.utcnow() - user.last_login) < self.lockout_duration:
                    logger.warning(f"Account locked for user: {username}")
                    return None
                else:
                    # Reset failed attempts after lockout period
                    user.failed_login_attempts = 0
            
            # Check if account is active
            if user.status != UserStatus.ACTIVE.value:
                logger.warning(f"Login attempt for inactive user: {username}")
                return None
            
            # Verify password
            if not self.pwd_context.verify(password, user.hashed_password):
                user.failed_login_attempts += 1
                db.commit()
                
                self.log_user_action(db, user.id, "failed_login", {
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "failed_attempts": user.failed_login_attempts
                })
                
                logger.warning(f"Failed login attempt for user: {username}")
                return None
            
            # Successful login
            user.last_login = datetime.utcnow()
            user.failed_login_attempts = 0
            db.commit()
            
            self.log_user_action(db, user.id, "successful_login", {
                "ip_address": ip_address,
                "user_agent": user_agent
            })
            
            logger.info(f"Successful login for user: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            return None
    
    def log_user_action(self, db: Session, user_id: int, action: str, 
                       details: Dict = None, ip_address: str = None, 
                       user_agent: str = None):
        """Log user actions for audit trail"""
        try:
            audit_log = UserAuditLog(
                user_id=user_id,
                action=action,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            db.add(audit_log)
            db.commit()
        except Exception as e:
            logger.error(f"Error logging user action: {str(e)}")
