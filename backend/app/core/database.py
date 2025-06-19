from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator
import logging
from .config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Create metadata instance for table operations
metadata = MetaData()

def create_tables():
    """Create all tables in the database"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Generator:
    """
    Dependency that creates a database session for each request
    and closes it when the request is complete
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialize database with any required initial data"""
    try:
        # Create tables
        create_tables()

        # Add any initial data setup here
        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

# Health check function
def check_db_health() -> bool:
    """Check if database connection is healthy"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
