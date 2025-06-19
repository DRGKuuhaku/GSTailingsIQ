"""
TailingsIQ - Cross-Data Synthesis (CDS) API

This module provides API endpoints for the Cross-Data Synthesis engine that combines
multiple data types to provide intelligent insights for tailings storage facilities.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query as QueryParam, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
import asyncio
from uuid import uuid4
import json

from ..core.database import get_db
from ..core.security import get_current_user, require_permission
from ..models.user import User
from ..models.cds_models import (
    CDSQuery, CDSResult, CDSSourceReference, CDSFeedback, CDSCorrelation,
    AnalysisType, ConfidenceLevel, DataSourceType,
    SynthesisContext, CDSResultResponse, CDSInsight, CDSRecommendation
)
from ..services.cds_engine import CDSEngine, CDSConfig
from ..services.rag_processor import RAGProcessor
from ..services.multimodal_analyzer import MultiModalAnalyzer
from ..utils.helpers import validate_query_input, sanitize_response

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
from pydantic import BaseModel, Field

class CDSQueryRequest(BaseModel):
    """Cross-Data Synthesis query request"""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    analysis_types: Optional[List[AnalysisType]] = Field(
        default=[AnalysisType.TEMPORAL, AnalysisType.CORRELATION], 
        description="Types of analysis to perform"
    )
    data_sources: Optional[List[DataSourceType]] = Field(
        default=[DataSourceType.DOCUMENTS, DataSourceType.MONITORING, DataSourceType.COMPLIANCE],
        description="Data sources to include in synthesis"
    )
    facility_ids: Optional[List[int]] = Field(default=None, description="Specific facility IDs to analyze")
    time_range: Optional[Dict[str, datetime]] = Field(default=None, description="Time range for analysis")
    confidence_threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_sources: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum number of sources to analyze")
    include_predictions: Optional[bool] = Field(default=True, description="Include predictive analysis")
    priority: Optional[str] = Field(default="normal", regex="^(low|normal|high|urgent)$", description="Query priority")

class CDSQueryResponse(BaseModel):
    """Cross-Data Synthesis query response"""
    query_id: str = Field(..., description="Unique query identifier")
    status: str = Field(..., description="Query processing status")
    query: str = Field(..., description="Original query text")
    synthesis_summary: Optional[str] = Field(None, description="High-level synthesis summary")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    insights: List[CDSInsight] = Field(default_factory=list, description="Generated insights")
    recommendations: List[CDSRecommendation] = Field(default_factory=list, description="Action recommendations")
    correlations: List[Dict[str, Any]] = Field(default_factory=list, description="Discovered correlations")
    source_summary: Dict[str, int] = Field(default_factory=dict, description="Summary of sources used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(..., description="Query timestamp")
    human_review_required: bool = Field(default=False, description="Whether human review is recommended")

class CDSProcessingStatus(BaseModel):
    """CDS query processing status"""
    query_id: str = Field(..., description="Query identifier")
    status: str = Field(..., description="Current processing status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Processing progress (0-1)")
    current_step: str = Field(..., description="Current processing step")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated seconds remaining")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class CDSFeedbackRequest(BaseModel):
    """Feedback on CDS results"""
    query_id: str = Field(..., description="Query identifier")
    overall_rating: int = Field(..., ge=1, le=5, description="Overall rating (1-5)")
    accuracy_rating: int = Field(..., ge=1, le=5, description="Accuracy rating (1-5)")
    usefulness_rating: int = Field(..., ge=1, le=5, description="Usefulness rating (1-5)")
    insights_helpful: List[str] = Field(default_factory=list, description="IDs of helpful insights")
    insights_incorrect: List[str] = Field(default_factory=list, description="IDs of incorrect insights")
    comments: Optional[str] = Field(None, max_length=2000, description="Additional feedback comments")
    suggested_improvements: Optional[str] = Field(None, max_length=1000, description="Suggested improvements")

class CDSAnalytics(BaseModel):
    """CDS system analytics"""
    total_queries: int = Field(..., description="Total number of queries processed")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    avg_confidence_score: float = Field(..., description="Average confidence score")
    success_rate: float = Field(..., description="Query success rate")
    top_query_types: List[Dict[str, Any]] = Field(..., description="Most common query types")
    data_source_usage: Dict[str, int] = Field(..., description="Usage statistics by data source")
    user_satisfaction: Dict[str, float] = Field(..., description="User satisfaction metrics")
    recent_performance: List[Dict[str, Any]] = Field(..., description="Recent performance metrics")

class HumanReviewRequest(BaseModel):
    """Human review submission for CDS results"""
    query_id: str = Field(..., description="Query identifier")
    review_status: str = Field(..., regex="^(approved|rejected|needs_revision)$", description="Review decision")
    reviewer_comments: Optional[str] = Field(None, max_length=2000, description="Reviewer comments")
    confidence_adjustment: Optional[float] = Field(None, ge=0.0, le=1.0, description="Adjusted confidence score")
    approved_insights: List[str] = Field(default_factory=list, description="IDs of approved insights")
    rejected_insights: List[str] = Field(default_factory=list, description="IDs of rejected insights")

@router.post("/query", response_model=CDSQueryResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_cds_query(
    query_request: CDSQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit a Cross-Data Synthesis query for processing.

    This endpoint initiates CDS processing that combines multiple data types
    to provide comprehensive insights about tailings storage facilities.
    """
    try:
        # Validate input
        if not validate_query_input(query_request.query):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid query format or content"
            )

        # Check user permissions for requested facilities
        if query_request.facility_ids:
            await validate_facility_access(current_user, query_request.facility_ids)

        # Generate unique query ID
        query_id = str(uuid4())

        # Create query record
        db_query = CDSQuery(
            id=query_id,
            user_id=current_user.id,
            query_text=query_request.query,
            analysis_types=query_request.analysis_types,
            data_sources=query_request.data_sources,
            facility_ids=query_request.facility_ids,
            time_range=query_request.time_range,
            confidence_threshold=query_request.confidence_threshold,
            max_sources=query_request.max_sources,
            priority=query_request.priority,
            status="queued",
            created_at=datetime.utcnow()
        )

        db.add(db_query)
        db.commit()
        db.refresh(db_query)

        # Start background processing
        background_tasks.add_task(
            process_cds_query,
            query_id,
            query_request,
            current_user.id,
            db
        )

        logger.info(f"CDS query {query_id} submitted by user {current_user.username}")

        return CDSQueryResponse(
            query_id=query_id,
            status="processing",
            query=query_request.query,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting CDS query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit CDS query: {str(e)}"
        )

@router.get("/query/{query_id}/status", response_model=CDSProcessingStatus)
async def get_query_status(
    query_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the processing status of a CDS query.
    """
    try:
        # Get query from database
        query = db.query(CDSQuery).filter(
            CDSQuery.id == query_id,
            CDSQuery.user_id == current_user.id
        ).first()

        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found or access denied"
            )

        # Calculate progress based on status
        progress_map = {
            "queued": 0.0,
            "processing": 0.5,
            "completed": 1.0,
            "failed": 1.0
        }

        return CDSProcessingStatus(
            query_id=query_id,
            status=query.status,
            progress=progress_map.get(query.status, 0.0),
            current_step=query.current_step or "Initializing",
            estimated_time_remaining=query.estimated_completion_time,
            error_message=query.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query status {query_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query status"
        )

@router.get("/query/{query_id}/result", response_model=CDSQueryResponse)
async def get_query_result(
    query_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the results of a completed CDS query.
    """
    try:
        # Get query and result from database
        query = db.query(CDSQuery).filter(
            CDSQuery.id == query_id,
            CDSQuery.user_id == current_user.id
        ).first()

        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found or access denied"
            )

        if query.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query not completed. Current status: {query.status}"
            )

        # Get the result
        result = db.query(CDSResult).filter(CDSResult.query_id == query_id).first()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query result not found"
            )

        # Get sources
        sources = db.query(CDSSourceReference).filter(
            CDSSourceReference.result_id == result.id
        ).all()

        # Get correlations
        correlations = db.query(CDSCorrelation).filter(
            CDSCorrelation.result_id == result.id
        ).all()

        return CDSQueryResponse(
            query_id=query_id,
            status=query.status,
            query=query.query_text,
            synthesis_summary=result.synthesis_summary,
            confidence_score=result.confidence_score,
            insights=result.insights or [],
            recommendations=result.recommendations or [],
            correlations=[c.to_dict() for c in correlations],
            source_summary=result.source_summary or {},
            processing_time=result.processing_time,
            timestamp=query.created_at,
            human_review_required=result.human_review_required
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query result {query_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query result"
        )

@router.get("/queries", response_model=List[CDSQueryResponse])
async def get_user_queries(
    status: Optional[str] = QueryParam(None, description="Filter by status"),
    limit: int = QueryParam(20, ge=1, le=100, description="Maximum number of queries"),
    offset: int = QueryParam(0, ge=0, description="Number of queries to skip"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's CDS query history.
    """
    try:
        query = db.query(CDSQuery).filter(CDSQuery.user_id == current_user.id)

        if status:
            query = query.filter(CDSQuery.status == status)

        queries = query.order_by(CDSQuery.created_at.desc()).offset(offset).limit(limit).all()

        # Convert to response format
        responses = []
        for q in queries:
            response = CDSQueryResponse(
                query_id=q.id,
                status=q.status,
                query=q.query_text,
                timestamp=q.created_at
            )

            # Add result data if completed
            if q.status == "completed":
                result = db.query(CDSResult).filter(CDSResult.query_id == q.id).first()
                if result:
                    response.synthesis_summary = result.synthesis_summary
                    response.confidence_score = result.confidence_score
                    response.insights = result.insights or []
                    response.recommendations = result.recommendations or []
                    response.processing_time = result.processing_time

            responses.append(response)

        return responses

    except Exception as e:
        logger.error(f"Error getting user queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query history"
        )

@router.post("/feedback")
async def submit_feedback(
    feedback: CDSFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit feedback on CDS query results to improve future performance.
    """
    try:
        # Verify query ownership
        query = db.query(CDSQuery).filter(
            CDSQuery.id == feedback.query_id,
            CDSQuery.user_id == current_user.id
        ).first()

        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found or access denied"
            )

        # Create feedback record
        db_feedback = CDSFeedback(
            query_id=feedback.query_id,
            user_id=current_user.id,
            overall_rating=feedback.overall_rating,
            accuracy_rating=feedback.accuracy_rating,
            usefulness_rating=feedback.usefulness_rating,
            insights_helpful=feedback.insights_helpful,
            insights_incorrect=feedback.insights_incorrect,
            comments=feedback.comments,
            suggested_improvements=feedback.suggested_improvements,
            created_at=datetime.utcnow()
        )

        db.add(db_feedback)
        db.commit()

        logger.info(f"Feedback submitted for query {feedback.query_id} by user {current_user.username}")

        return {"message": "Feedback submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

@router.get("/analytics", response_model=CDSAnalytics)
@require_permission("admin")
async def get_cds_analytics(
    days: int = QueryParam(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get CDS system analytics and performance metrics (admin only).
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)

        # Calculate analytics
        analytics = await calculate_cds_analytics(start_date, db)

        return analytics

    except Exception as e:
        logger.error(f"Error calculating CDS analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate analytics"
        )

@router.get("/correlations", response_model=List[Dict[str, Any]])
async def get_discovered_correlations(
    facility_id: Optional[int] = QueryParam(None, description="Filter by facility"),
    significance_threshold: float = QueryParam(0.7, ge=0.0, le=1.0, description="Minimum significance"),
    limit: int = QueryParam(50, ge=1, le=200, description="Maximum number of correlations"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get discovered correlations from CDS analysis.
    """
    try:
        query = db.query(CDSCorrelation).filter(
            CDSCorrelation.significance >= significance_threshold
        )

        if facility_id:
            await validate_facility_access(current_user, [facility_id])
            query = query.filter(CDSCorrelation.facility_id == facility_id)

        correlations = query.order_by(
            CDSCorrelation.significance.desc()
        ).limit(limit).all()

        return [c.to_dict() for c in correlations]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting correlations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve correlations"
        )

@router.get("/health")
async def cds_health_check(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Check the health status of the CDS system.
    """
    try:
        # Check CDS engine health
        cds_engine = CDSEngine()
        engine_health = await cds_engine.health_check()

        # Check recent query success rate
        recent_queries = db.query(CDSQuery).filter(
            CDSQuery.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).all()

        success_count = len([q for q in recent_queries if q.status == "completed"])
        total_count = len(recent_queries)
        success_rate = success_count / total_count if total_count > 0 else 1.0

        # Check processing queue
        queued_count = db.query(CDSQuery).filter(CDSQuery.status == "queued").count()
        processing_count = db.query(CDSQuery).filter(CDSQuery.status == "processing").count()

        health_status = {
            "status": "healthy" if engine_health and success_rate > 0.8 else "degraded",
            "cds_engine": "operational" if engine_health else "error",
            "recent_success_rate": success_rate,
            "queries_in_queue": queued_count,
            "queries_processing": processing_count,
            "timestamp": datetime.utcnow().isoformat()
        }

        return health_status

    except Exception as e:
        logger.error(f"CDS health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"CDS system health check failed: {str(e)}"
        )

@router.post("/human-review")
@require_permission("compliance_review")
async def submit_human_review(
    review: HumanReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit human review for CDS results that require expert validation.
    """
    try:
        # Get the result that needs review
        result = db.query(CDSResult).filter(CDSResult.query_id == review.query_id).first()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query result not found"
            )

        if not result.human_review_required:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This query does not require human review"
            )

        # Update result with review
        result.human_review_status = review.review_status
        result.human_review_comments = review.reviewer_comments
        result.reviewed_by = current_user.id
        result.reviewed_at = datetime.utcnow()

        if review.confidence_adjustment is not None:
            result.confidence_score = review.confidence_adjustment

        # Update insight approval status
        if result.insights:
            for insight in result.insights:
                if insight.get("id") in review.approved_insights:
                    insight["human_approved"] = True
                elif insight.get("id") in review.rejected_insights:
                    insight["human_approved"] = False

        db.commit()

        logger.info(f"Human review submitted for query {review.query_id} by {current_user.username}")

        return {"message": "Human review submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting human review: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit human review"
        )

# Helper functions

async def validate_facility_access(user: User, facility_ids: List[int]) -> None:
    """Validate user has access to all requested facilities."""
    user_facilities = set(user.facilities_access or [])
    requested_facilities = set(facility_ids)

    if not requested_facilities.issubset(user_facilities):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to one or more requested facilities"
        )

async def process_cds_query(
    query_id: str,
    query_request: CDSQueryRequest,
    user_id: int,
    db: Session
) -> None:
    """Background task to process CDS query."""
    try:
        logger.info(f"Starting CDS processing for query {query_id}")

        # Update query status
        query = db.query(CDSQuery).filter(CDSQuery.id == query_id).first()
        query.status = "processing"
        query.current_step = "Initializing CDS engine"
        db.commit()

        # Initialize CDS engine
        config = CDSConfig(
            max_sources=query_request.max_sources,
            confidence_threshold=query_request.confidence_threshold,
            processing_timeout=300  # 5 minutes
        )
        cds_engine = CDSEngine(config)

        # Build synthesis context
        context = SynthesisContext(
            query_text=query_request.query,
            analysis_types=query_request.analysis_types,
            data_sources=query_request.data_sources,
            facility_ids=query_request.facility_ids,
            time_range=query_request.time_range,
            user_id=user_id
        )

        # Process the query
        start_time = datetime.utcnow()

        query.current_step = "Processing synthesis"
        db.commit()

        result = await cds_engine.synthesize(context, db)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Create result record
        db_result = CDSResult(
            query_id=query_id,
            synthesis_summary=result.synthesis_summary,
            confidence_score=result.confidence_score,
            insights=result.insights,
            recommendations=result.recommendations,
            source_summary=result.source_summary,
            processing_time=processing_time,
            human_review_required=result.confidence_score < 0.7,  # Require review for low confidence
            created_at=datetime.utcnow()
        )

        db.add(db_result)

        # Update query status
        query.status = "completed"
        query.current_step = "Complete"
        query.completed_at = datetime.utcnow()

        db.commit()

        logger.info(f"CDS processing completed for query {query_id} in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Error processing CDS query {query_id}: {str(e)}")

        # Update query with error status
        query = db.query(CDSQuery).filter(CDSQuery.id == query_id).first()
        if query:
            query.status = "failed"
            query.error_message = str(e)
            query.current_step = "Failed"
            db.commit()

async def calculate_cds_analytics(start_date: datetime, db: Session) -> CDSAnalytics:
    """Calculate CDS system analytics."""
    try:
        # Get all queries in date range
        queries = db.query(CDSQuery).filter(CDSQuery.created_at >= start_date).all()

        total_queries = len(queries)
        completed_queries = [q for q in queries if q.status == "completed"]

        # Calculate metrics
        success_rate = len(completed_queries) / total_queries if total_queries > 0 else 0.0

        # Get processing times
        results = db.query(CDSResult).join(CDSQuery).filter(
            CDSQuery.created_at >= start_date
        ).all()

        avg_processing_time = sum(r.processing_time for r in results) / len(results) if results else 0.0
        avg_confidence_score = sum(r.confidence_score for r in results) / len(results) if results else 0.0

        # Get feedback for satisfaction
        feedbacks = db.query(CDSFeedback).join(CDSQuery).filter(
            CDSQuery.created_at >= start_date
        ).all()

        user_satisfaction = {
            "overall": sum(f.overall_rating for f in feedbacks) / len(feedbacks) / 5.0 if feedbacks else 0.0,
            "accuracy": sum(f.accuracy_rating for f in feedbacks) / len(feedbacks) / 5.0 if feedbacks else 0.0,
            "usefulness": sum(f.usefulness_rating for f in feedbacks) / len(feedbacks) / 5.0 if feedbacks else 0.0
        }

        return CDSAnalytics(
            total_queries=total_queries,
            avg_processing_time=avg_processing_time,
            avg_confidence_score=avg_confidence_score,
            success_rate=success_rate,
            top_query_types=[],  # Would implement based on query analysis
            data_source_usage={},  # Would implement based on usage tracking
            user_satisfaction=user_satisfaction,
            recent_performance=[]  # Would implement based on recent metrics
        )

    except Exception as e:
        logger.error(f"Error calculating CDS analytics: {str(e)}")
        raise