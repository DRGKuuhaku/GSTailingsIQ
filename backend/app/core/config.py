"""
TailingsIQ - AI-Enhanced TSF Management Platform
Configuration Module

This module contains all configuration settings, environment variables,
and production security configurations for the TailingsIQ application.
"""

from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any, Union
import os
import secrets
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden by environment variables.
    """

    # Basic Application Settings
    PROJECT_NAME: str = "TailingsIQ"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-Enhanced TSF Management Platform with Cross-Data Synthesis"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    RELOAD: bool = False

    # Database Configuration
    DATABASE_URL: str = "postgresql://tailingsiq:password@localhost:5432/tailingsiq"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_PRE_PING: bool = True
    DATABASE_ECHO: bool = False

    # Redis Configuration (for caching and background tasks)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    REDIS_SESSION_TTL: int = 86400  # 24 hours

    # Elasticsearch Configuration
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX: str = "tailingsiq_documents"
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_MAX_RETRIES: int = 3

    # ChromaDB Configuration (Vector Database)
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "tailingsiq_vectors"
    CHROMA_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # AI/ML Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_TIMEOUT: int = 60

    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    VECTOR_SEARCH_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7

    # Cross-Data Synthesis Configuration
    CDS_ENABLED: bool = True
    CDS_MAX_SOURCES: int = 50
    CDS_CONFIDENCE_THRESHOLD: float = 0.6
    CDS_CACHE_TTL: int = 1800  # 30 minutes
    CDS_MAX_QUERY_LENGTH: int = 1000
    CDS_PROCESSING_TIMEOUT: int = 300  # 5 minutes

    # Security Configuration
    SECRET_KEY: str = "your-secret-key-change-in-production-must-be-256-bit-key-for-security"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    SESSION_MAX_AGE: int = 3600  # 1 hour

    # Password Security
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SYMBOLS: bool = True
    PASSWORD_HASH_ROUNDS: int = 12

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    RATE_LIMIT_ENABLED: bool = True

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://tailingsiq-frontend.vercel.app",
        "https://*.tailingsiq.com"
    ]

    # Trusted Hosts
    ALLOWED_HOSTS: List[str] = ["*"]  # Restrict in production

    # File Upload Configuration
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_FILE_TYPES: List[str] = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",
        "text/csv",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "application/zip"
    ]

    # Monitoring and Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = "./logs/tailingsiq.log"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5

    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PATH: str = "/metrics"
    HEALTH_CHECK_PATH: str = "/health"

    # Email Configuration (for notifications)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_TLS: bool = True
    EMAIL_FROM: Optional[str] = None

    # Notification Configuration
    NOTIFICATION_ENABLED: bool = True
    NOTIFICATION_CHANNELS: List[str] = ["email", "system"]

    # Background Tasks
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    BACKGROUND_TASK_TIMEOUT: int = 300  # 5 minutes

    # External API Configuration
    WEATHER_API_KEY: Optional[str] = None
    WEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5"

    # Backup Configuration
    BACKUP_ENABLED: bool = True
    BACKUP_SCHEDULE: str = "0 2 * * *"  # Daily at 2 AM
    BACKUP_RETENTION_DAYS: int = 30
    BACKUP_DIRECTORY: str = "./backups"

    # Cache Configuration
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000

    # Feature Flags
    FEATURE_SYNTHETIC_DATA: bool = True
    FEATURE_DOCUMENT_ANALYSIS: bool = True
    FEATURE_MONITORING_ALERTS: bool = True
    FEATURE_COMPLIANCE_TRACKING: bool = True
    FEATURE_USER_ANALYTICS: bool = True

    # Testing Configuration
    TESTING: bool = False
    TEST_DATABASE_URL: Optional[str] = None

    # Development Configuration
    RELOAD_DIRS: List[str] = ["./app"]

    # Production Security Settings
    SECURE_COOKIES: bool = True
    COOKIE_SAMESITE: str = "strict"
    COOKIE_HTTPONLY: bool = True

    # API Documentation
    DOCS_ENABLED: bool = True
    REDOC_ENABLED: bool = True
    OPENAPI_ENABLED: bool = True

    # Performance Settings
    MAX_WORKERS: int = 4
    WORKER_TIMEOUT: int = 30
    KEEPALIVE: int = 2
    MAX_REQUESTS: int = 1000
    MAX_REQUESTS_JITTER: int = 100

    # Data Retention
    USER_SESSION_RETENTION_DAYS: int = 90
    AUDIT_LOG_RETENTION_DAYS: int = 365
    DOCUMENT_RETENTION_DAYS: int = 2555  # 7 years

    # Compliance Settings
    GDPR_ENABLED: bool = True
    DATA_ANONYMIZATION_ENABLED: bool = True
    AUDIT_TRAIL_ENABLED: bool = True

    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL for SQLAlchemy"""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")

    @property
    def database_url_async(self) -> str:
        """Asynchronous database URL for async SQLAlchemy"""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

    def get_secret_key(self) -> str:
        """Get or generate secret key"""
        if self.SECRET_KEY == "your-secret-key-change-in-production-must-be-256-bit-key-for-security":
            if self.DEBUG:
                return self.SECRET_KEY
            else:
                # Generate secure secret key for production
                return secrets.token_urlsafe(32)
        return self.SECRET_KEY

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins with environment override"""
        if cors_origins := os.getenv("CORS_ORIGINS"):
            return [origin.strip() for origin in cors_origins.split(",")]
        return self.BACKEND_CORS_ORIGINS

    def get_allowed_hosts(self) -> List[str]:
        """Get allowed hosts with security defaults"""
        if self.DEBUG:
            return ["*"]

        if allowed_hosts := os.getenv("ALLOWED_HOSTS"):
            return [host.strip() for host in allowed_hosts.split(",")]

        # Production defaults
        return [
            "localhost",
            "127.0.0.1",
            "tailingsiq.com",
            "*.tailingsiq.com",
            "api.tailingsiq.com"
        ]

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return not self.DEBUG and os.getenv("ENVIRONMENT", "development") == "production"

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary"""
        return {
            "url": self.database_url_sync,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_pre_ping": self.DATABASE_POOL_PRE_PING,
            "echo": self.DATABASE_ECHO and self.DEBUG
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "url": self.REDIS_URL,
            "encoding": "utf-8",
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True
        }

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration dictionary"""
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.OPENAI_MODEL,
            "embedding_model": self.OPENAI_EMBEDDING_MODEL,
            "max_tokens": self.OPENAI_MAX_TOKENS,
            "temperature": self.OPENAI_TEMPERATURE,
            "timeout": self.OPENAI_TIMEOUT
        }

    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration dictionary"""
        config = {
            "hosts": [self.ELASTICSEARCH_URL],
            "timeout": self.ELASTICSEARCH_TIMEOUT,
            "max_retries": self.ELASTICSEARCH_MAX_RETRIES,
            "retry_on_timeout": True
        }

        if self.ELASTICSEARCH_USERNAME and self.ELASTICSEARCH_PASSWORD:
            config["http_auth"] = (self.ELASTICSEARCH_USERNAME, self.ELASTICSEARCH_PASSWORD)

        return config

    def get_chroma_config(self) -> Dict[str, Any]:
        """Get ChromaDB configuration dictionary"""
        return {
            "persist_directory": self.CHROMA_PERSIST_DIRECTORY,
            "collection_name": self.CHROMA_COLLECTION_NAME,
            "embedding_model": self.CHROMA_EMBEDDING_MODEL
        }

    def get_cds_config(self) -> Dict[str, Any]:
        """Get Cross-Data Synthesis configuration dictionary"""
        return {
            "enabled": self.CDS_ENABLED,
            "max_sources": self.CDS_MAX_SOURCES,
            "confidence_threshold": self.CDS_CONFIDENCE_THRESHOLD,
            "cache_ttl": self.CDS_CACHE_TTL,
            "max_query_length": self.CDS_MAX_QUERY_LENGTH,
            "processing_timeout": self.CDS_PROCESSING_TIMEOUT
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_ECHO: bool = True
    DOCS_ENABLED: bool = True
    RELOAD: bool = True
    SECURE_COOKIES: bool = False
    RATE_LIMIT_ENABLED: bool = False

class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DATABASE_ECHO: bool = False
    DOCS_ENABLED: bool = False
    RELOAD: bool = False
    SECURE_COOKIES: bool = True
    RATE_LIMIT_ENABLED: bool = True
    WORKERS: int = 4

    # Override with secure defaults
    @property
    def ALLOWED_HOSTS(self) -> List[str]:
        return self.get_allowed_hosts()

class TestingSettings(Settings):
    """Testing environment settings"""
    TESTING: bool = True
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./test.db"
    LOG_LEVEL: str = "WARNING"
    RATE_LIMIT_ENABLED: bool = False
    NOTIFICATION_ENABLED: bool = False
    CACHE_ENABLED: bool = False

def get_settings() -> Settings:
    """
    Get settings based on environment.

    Returns:
        Settings: Environment-specific settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Create global settings instance
settings = get_settings()

# Ensure required directories exist
def create_directories():
    """Create required directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.CHROMA_PERSIST_DIRECTORY,
        settings.BACKUP_DIRECTORY,
        "./logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Create directories on import
create_directories()

# Validate critical settings
def validate_settings():
    """Validate critical settings and raise errors if invalid"""
    if settings.is_production():
        # Critical production validations
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required in production")

        if settings.SECRET_KEY == "your-secret-key-change-in-production-must-be-256-bit-key-for-security":
            raise ValueError("SECRET_KEY must be changed in production")

        if not settings.DATABASE_URL or "localhost" in settings.DATABASE_URL:
            raise ValueError("Production DATABASE_URL must not use localhost")

        if settings.DEBUG:
            raise ValueError("DEBUG must be False in production")

# Run validation
if not settings.TESTING:
    validate_settings()