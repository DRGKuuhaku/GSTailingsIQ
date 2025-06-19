"""
TailingsIQ - AI Query Engine API

This module provides API endpoints for the AI-powered query engine that allows
users to ask natural language questions about tailings storage facilities.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query as QueryParam
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio
from uuid import uuid4

from ..core.database import get_db
from ..core.security import get_current_user, require_permission
from ..models.user import User
from ..models.document import Document
from ..models.monitoring import MonitoringReading, MonitoringStation
from ..services.ai_reasoning import AIReasoningService
from ..services.rag_processor import RAGProcessor
from ..utils.helpers import validate_query_input, sanitize_response

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for the query")
    facility_ids: Optional[List[int]] = Field(default=None, description="Specific facility IDs to query")
    data_sources: Optional[List[str]] = Field(default=["documents", "monitoring", "compliance"], description="Data sources to include")
    max_results: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of results")
    include_confidence: Optional[bool] = Field(default=True, description="Include confidence scores")

class QueryResponse(BaseModel):
    """AI query response"""
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Original query text")
    answer: str = Field(..., description="AI-generated answer")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents and data")
    processing_time: float = Field(..., description="Query processing time in seconds")
    timestamp: datetime = Field(..., description="Query timestamp")
    suggestions: Optional[List[str]] = Field(default=None, description="Related query suggestions")

class QueryHistory(BaseModel):
    """User query history"""
    queries: List[QueryResponse] = Field(..., description="List of previous queries")
    total_count: int = Field(..., description="Total number of queries")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of queries per page")

class QueryAnalytics(BaseModel):
    """Query analytics and metrics"""
    total_queries: int = Field(..., description="Total number of queries")
    avg_processing_time: float = Field(..., description="Average processing time")
    top_query_types: List[Dict[str, Any]] = Field(..., description="Most common query types")
    user_activity: Dict[str, int] = Field(..., description="Query activity by user")
    popular_topics: List[str] = Field(..., description="Most queried topics")

@router.post("/submit", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def submit_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit a natural language query to the AI engine.

    This endpoint processes natural language queries and returns AI-generated answers
    with supporting evidence from documents, monitoring data, and compliance records.
    """
    try:
        # Validate input
        if not validate_query_input(query_request.query):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query format or content"
            )

        # Check rate limiting (basic implementation)
        await check_rate_limit(current_user.id, db)

        # Generate unique query ID
        query_id = str(uuid4())
        start_time = datetime.utcnow()

        # Initialize AI reasoning service
        ai_service = AIReasoningService()

        # Process the query
        logger.info(f"Processing query {query_id} for user {current_user.username}")

        # Get query context
        context = await build_query_context(
            query_request.query,
            query_request.facility_ids,
            query_request.data_sources,
            current_user,
            db
        )

        # Generate AI response
        ai_response = await ai_service.process_query(
            query=query_request.query,
            context=context,
            user_id=current_user.id,
            max_results=query_request.max_results
        )

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Prepare response
        response = QueryResponse(
            query_id=query_id,
            query=query_request.query,
            answer=sanitize_response(ai_response.answer),
            confidence=ai_response.confidence if query_request.include_confidence else None,
            sources=ai_response.sources,
            processing_time=processing_time,
            timestamp=start_time,
            suggestions=ai_response.suggestions
        )

        # Log query for analytics (background task)
        background_tasks.add_task(
            log_query_analytics,
            query_id,
            current_user.id,
            query_request.query,
            processing_time,
            ai_response.confidence,
            db
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@router.get("/history", response_model=QueryHistory)
async def get_query_history(
    page: int = QueryParam(1, ge=1, description="Page number"),
    page_size: int = QueryParam(20, ge=1, le=100, description="Items per page"),
    start_date: Optional[datetime] = QueryParam(None, description="Start date filter"),
    end_date: Optional[datetime] = QueryParam(None, description="End date filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's query history with pagination and filtering.
    """
    try:
        # Build query filters
        filters = {"user_id": current_user.id}
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date

        # Get queries from database (assuming a QueryLog model exists)
        queries = await get_user_queries(
            filters=filters,
            page=page,
            page_size=page_size,
            db=db
        )

        # Get total count
        total_count = await count_user_queries(filters, db)

        return QueryHistory(
            queries=queries,
            total_count=total_count,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error retrieving query history for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query history"
        )

@router.get("/suggestions")
async def get_query_suggestions(
    q: Optional[str] = QueryParam(None, description="Partial query for suggestions"),
    limit: int = QueryParam(5, ge=1, le=20, description="Maximum number of suggestions"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get query suggestions based on partial input or popular queries.
    """
    try:
        suggestions = await generate_query_suggestions(
            partial_query=q,
            user_id=current_user.id,
            limit=limit,
            db=db
        )

        return {"suggestions": suggestions}

    except Exception as e:
        logger.error(f"Error generating query suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate suggestions"
        )

@router.get("/analytics", response_model=QueryAnalytics)
@require_permission("admin")
async def get_query_analytics(
    days: int = QueryParam(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get query analytics and usage metrics (admin only).
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)

        analytics = await calculate_query_analytics(
            start_date=start_date,
            db=db
        )

        return analytics

    except Exception as e:
        logger.error(f"Error calculating query analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate analytics"
        )

@router.post("/feedback")
async def submit_query_feedback(
    query_id: str,
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5"),
    feedback: Optional[str] = Field(None, max_length=1000, description="Optional feedback text"),
    helpful: bool = Field(..., description="Whether the response was helpful"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit feedback for a query response to improve AI performance.
    """
    try:
        # Validate query ID belongs to user
        query_exists = await verify_query_ownership(query_id, current_user.id, db)
        if not query_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found or access denied"
            )

        # Store feedback
        await store_query_feedback(
            query_id=query_id,
            user_id=current_user.id,
            rating=rating,
            feedback=feedback,
            helpful=helpful,
            db=db
        )

        return {"message": "Feedback submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback for query {query_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

# Helper functions

async def check_rate_limit(user_id: int, db: Session) -> None:
    """Check if user has exceeded rate limits for queries."""
    # Implement rate limiting logic
    # This is a basic example - consider using Redis for production
    recent_queries = await count_recent_queries(user_id, minutes=60, db=db)
    if recent_queries > 100:  # 100 queries per hour limit
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

async def build_query_context(
    query: str,
    facility_ids: Optional[List[int]],
    data_sources: List[str],
    user: User,
    db: Session
) -> Dict[str, Any]:
    """Build context information for the query."""
    context = {
        "user_role": user.role,
        "user_facilities": user.facilities_access or [],
        "timestamp": datetime.utcnow().isoformat(),
        "data_sources": data_sources
    }

    if facility_ids:
        # Validate user has access to requested facilities
        accessible_facilities = set(user.facilities_access or [])
        requested_facilities = set(facility_ids)

        if not requested_facilities.issubset(accessible_facilities):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to one or more requested facilities"
            )

        context["facility_ids"] = facility_ids

    return context

async def generate_query_suggestions(
    partial_query: Optional[str],
    user_id: int,
    limit: int,
    db: Session
) -> List[str]:
    """Generate query suggestions based on input and user history."""
    suggestions = []

    if partial_query:
        # AI-powered query completion
        ai_service = AIReasoningService()
        suggestions = await ai_service.suggest_completions(partial_query, limit)
    else:
        # Popular queries for this user or globally
        suggestions = await get_popular_queries(user_id, limit, db)

    return suggestions

async def get_user_queries(filters: Dict[str, Any], page: int, page_size: int, db: Session) -> List[QueryResponse]:
    """Retrieve user's query history from database."""
    # This would query the QueryLog model
    # Placeholder implementation
    return []

async def count_user_queries(filters: Dict[str, Any], db: Session) -> int:
    """Count total user queries matching filters."""
    # Placeholder implementation
    return 0

async def count_recent_queries(user_id: int, minutes: int, db: Session) -> int:
    """Count recent queries for rate limiting."""
    # Placeholder implementation
    return 0

async def calculate_query_analytics(start_date: datetime, db: Session) -> QueryAnalytics:
    """Calculate query analytics and metrics."""
    # Placeholder implementation
    return QueryAnalytics(
        total_queries=0,
        avg_processing_time=0.0,
        top_query_types=[],
        user_activity={},
        popular_topics=[]
    )

async def verify_query_ownership(query_id: str, user_id: int, db: Session) -> bool:
    """Verify that a query belongs to the specified user."""
    # Placeholder implementation
    return True

async def store_query_feedback(
    query_id: str,
    user_id: int,
    rating: int,
    feedback: Optional[str],
    helpful: bool,
    db: Session
) -> None:
    """Store user feedback for a query."""
    # Placeholder implementation
    pass

async def get_popular_queries(user_id: int, limit: int, db: Session) -> List[str]:
    """Get popular queries for suggestions."""
    # Placeholder implementation
    return [
        "Show me recent piezometer readings for Dam A",
        "What are the current compliance issues?",
        "Generate a stability report for the north embankment",
        "Show me all documents mentioning seepage",
        "What is the current freeboard level?"
    ]

async def log_query_analytics(
    query_id: str,
    user_id: int,
    query: str,
    processing_time: float,
    confidence: float,
    db: Session
) -> None:
    """Log query for analytics purposes."""
    # Background task to log query analytics
    logger.info(f"Logging analytics for query {query_id}")
    # Implementation would store in QueryLog model