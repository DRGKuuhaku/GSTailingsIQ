"""
TailingsIQ - AI-Enhanced TSF Management Platform
Main FastAPI Application Entry Point

This module sets up the complete FastAPI application with all routers, middleware,
security configurations, and startup logic for production deployment.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import logging
import time
import asyncio
import os
from contextlib import asynccontextmanager
from sqlalchemy import text
from typing import Dict, Any

from .core.config import settings
from .core.database import create_tables, get_db, init_db
from .core.security import get_current_user, verify_token
from .api import auth, documents, monitoring, query, compliance, synthetic_data, cds
#from .api import auth, synthetic_data
from .api.admin import users as admin_users
from .models.user import User, UserCreate, UserRole, UserStatus
from .services.user_service import UserService

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./logs/app.log') if os.path.exists('./logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with startup and shutdown logic"""
    # Startup
    logger.info("Starting TailingsIQ application...")
    try:
        # Initialize database
        await init_database()

        # Initialize default users
        await init_default_users()

        # Initialize CDS engine if enabled
        if settings.CDS_ENABLED:
            await init_cds_engine()

        logger.info("TailingsIQ application started successfully")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down TailingsIQ application...")
    try:
        # Clean up background tasks
        await cleanup_background_tasks()

        # Close database connections
        await cleanup_database_connections()

        logger.info("TailingsIQ application shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

async def init_database():
    """Initialize database tables and connections"""
    try:
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def init_default_users():
    """Initialize default super admin user"""
    try:
        db = next(get_db())
        user_service = UserService()

        # Check if super admin exists
        existing_admin = db.query(User).filter(User.role == UserRole.SUPER_ADMIN.value).first()
        if not existing_admin:
            logger.info("Creating default super admin user...")
            default_admin = UserCreate(
                username="superadmin",
                email="admin@tailingsiq.com",
                password="ChangeMe123!",  # CHANGE IN PRODUCTION
                first_name="Super",
                last_name="Admin",
                role=UserRole.SUPER_ADMIN,
                organization="TailingsIQ",
                position="System Administrator"
            )
            user_service.create_user(db, default_admin)
            logger.info("Default super admin created. Please change password!")

        db.close()

    except Exception as e:
        logger.error(f"Error creating default users: {e}")
        raise

async def init_cds_engine():
    """Initialize Cross-Data Synthesis engine"""
    try:
        from .services.cds_engine import CDSEngine

        # Initialize CDS engine with configuration
        cds_engine = CDSEngine()
        await cds_engine.initialize()

        logger.info("CDS engine initialized successfully")

    except Exception as e:
        logger.error(f"CDS engine initialization failed: {e}")
        # CDS is optional, don't fail startup
        logger.warning("Continuing without CDS functionality")

async def cleanup_background_tasks():
    """Clean up any running background tasks"""
    try:
        # Cancel any pending background tasks
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} pending tasks...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        logger.error(f"Error cleaning up background tasks: {e}")

async def cleanup_database_connections():
    """Clean up database connections"""
    try:
        # Close database connections gracefully
        # Implementation depends on your database setup
        logger.info("Database connections cleaned up")

    except Exception as e:
        logger.error(f"Error cleaning up database connections: {e}")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Enhanced TSF Management Platform with Cross-Data Synthesis",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    max_age=settings.SESSION_MAX_AGE
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "connect-src 'self' wss: ws:;"
    )

    return response

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracking"""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(
        f"HTTP {exc.status_code} error on {request.method} {request.url}: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper logging"""
    logger.error(
        f"Unhandled exception on {request.method} {request.url}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check(db = Depends(get_db)):
    """Health check endpoint for monitoring and load balancers"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))

        # Test CDS engine if enabled
        cds_status = "disabled"
        if settings.CDS_ENABLED:
            try:
                from .services.cds_engine import CDSEngine
                cds_engine = CDSEngine()
                if await cds_engine.health_check():
                    cds_status = "healthy"
                else:
                    cds_status = "unhealthy"
            except Exception as e:
                cds_status = f"error: {str(e)}"

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": "development" if settings.DEBUG else "production",
            "services": {
                "database": "connected",
                "api": "operational",
                "cds": cds_status
            },
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )

# Metrics endpoint for monitoring
@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring systems"""
    if not settings.DEBUG:
        # Require authentication in production
        current_user = Depends(get_current_user)
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")

    try:
        return {
            "requests_total": getattr(app.state, 'requests_total', 0),
            "requests_in_progress": getattr(app.state, 'requests_in_progress', 0),
            "memory_usage": get_memory_usage(),
            "database_connections": get_db_connection_count(),
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}

def get_db_connection_count() -> int:
    """Get current database connection count"""
    try:
        # Implementation depends on your database setup
        return 1  # Placeholder
    except Exception:
        return -1

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TailingsIQ API",
        "version": settings.VERSION,
        "description": "AI-Enhanced TSF Management Platform",
        "docs": "/docs" if settings.DEBUG else "Documentation not available in production",
        "health": "/health",
        "cds_enabled": settings.CDS_ENABLED
    }

# Include API routers
app.include_router(
    auth.router, 
    prefix=f"{settings.API_V1_STR}/auth", 
    tags=["authentication"]
)

app.include_router(
    documents.router, 
    prefix=f"{settings.API_V1_STR}/documents", 
    tags=["documents"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    monitoring.router, 
    prefix=f"{settings.API_V1_STR}/monitoring", 
    tags=["monitoring"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    query.router, 
    prefix=f"{settings.API_V1_STR}/query", 
    tags=["ai-query"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    compliance.router, 
    prefix=f"{settings.API_V1_STR}/compliance", 
    tags=["compliance"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    admin_users.router, 
    prefix=f"{settings.API_V1_STR}/admin/users", 
    tags=["admin"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    synthetic_data.router, 
    prefix=f"{settings.API_V1_STR}/synthetic-data", 
    tags=["synthetic-data"],
    dependencies=[Depends(get_current_user)]
)

# Include CDS router if enabled
if settings.CDS_ENABLED:
    app.include_router(
        cds.router,
        prefix=f"{settings.API_V1_STR}/cds",
        tags=["cross-data-synthesis"],
        dependencies=[Depends(get_current_user)]
    )

# Store startup time
@app.on_event("startup")
async def startup_event():
    """Store application startup time"""
    app.state.start_time = time.time()
    app.state.requests_total = 0
    app.state.requests_in_progress = 0
    logger.info(f"TailingsIQ API started at {app.state.start_time}")

# Request counting middleware
@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Count total requests for metrics"""
    if hasattr(app.state, 'requests_total'):
        app.state.requests_total += 1
    if hasattr(app.state, 'requests_in_progress'):
        app.state.requests_in_progress += 1

    try:
        response = await call_next(request)
        return response
    finally:
        if hasattr(app.state, 'requests_in_progress'):
            app.state.requests_in_progress -= 1

if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn for production
    uvicorn_config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "reload": settings.DEBUG,
        "workers": 1 if settings.DEBUG else settings.WORKERS,
        "log_level": "info" if settings.DEBUG else "warning",
        "access_log": settings.DEBUG,
        "server_header": False,
        "date_header": False
    }

    logger.info(f"Starting TailingsIQ with config: {uvicorn_config}")
    uvicorn.run(**uvicorn_config)