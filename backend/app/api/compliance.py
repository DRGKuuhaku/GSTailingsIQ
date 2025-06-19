"""
TailingsIQ - Compliance Tracking API

This module provides API endpoints for managing compliance with tailings storage
facility regulations including GISTM, ANCOLD, CDA, and other standards.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query as QueryParam
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from enum import Enum

from ..core.database import get_db
from ..core.security import get_current_user, require_permission
from ..models.user import User
from ..models.compliance import (
    ComplianceRequirement, ComplianceAssessment, ComplianceAction, 
    ComplianceReport, ComplianceStandard, ComplianceStatus
)
from ..services.compliance_management import ComplianceManagementService
from ..utils.helpers import validate_compliance_data, generate_compliance_report

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
from pydantic import BaseModel, Field

class ComplianceRequirementCreate(BaseModel):
    """Create a new compliance requirement"""
    title: str = Field(..., min_length=1, max_length=200, description="Requirement title")
    description: str = Field(..., min_length=1, max_length=2000, description="Detailed description")
    standard: ComplianceStandard = Field(..., description="Compliance standard (GISTM, ANCOLD, etc.)")
    section_reference: str = Field(..., max_length=50, description="Section reference in standard")
    priority: str = Field("medium", regex="^(low|medium|high|critical)$", description="Priority level")
    facility_id: Optional[int] = Field(None, description="Specific facility ID if applicable")
    deadline: Optional[datetime] = Field(None, description="Compliance deadline")
    responsible_party: Optional[str] = Field(None, max_length=100, description="Responsible party")

class ComplianceRequirementUpdate(BaseModel):
    """Update a compliance requirement"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1, max_length=2000)
    priority: Optional[str] = Field(None, regex="^(low|medium|high|critical)$")
    deadline: Optional[datetime] = None
    responsible_party: Optional[str] = Field(None, max_length=100)
    status: Optional[ComplianceStatus] = None

class ComplianceRequirementResponse(BaseModel):
    """Compliance requirement response"""
    id: int
    title: str
    description: str
    standard: ComplianceStandard
    section_reference: str
    priority: str
    facility_id: Optional[int]
    deadline: Optional[datetime]
    responsible_party: Optional[str]
    status: ComplianceStatus
    created_at: datetime
    updated_at: Optional[datetime]
    last_assessment_date: Optional[datetime]
    next_review_date: Optional[datetime]

class ComplianceAssessmentCreate(BaseModel):
    """Create a compliance assessment"""
    requirement_id: int = Field(..., description="Compliance requirement ID")
    status: ComplianceStatus = Field(..., description="Assessment status")
    assessment_date: datetime = Field(..., description="Assessment date")
    findings: str = Field(..., min_length=1, max_length=5000, description="Assessment findings")
    evidence_documents: Optional[List[int]] = Field(default=[], description="Supporting document IDs")
    assessor_name: str = Field(..., min_length=1, max_length=100, description="Assessor name")
    next_review_date: Optional[datetime] = Field(None, description="Next review date")
    actions_required: Optional[List[str]] = Field(default=[], description="Required actions")

class ComplianceAssessmentResponse(BaseModel):
    """Compliance assessment response"""
    id: int
    requirement_id: int
    status: ComplianceStatus
    assessment_date: datetime
    findings: str
    evidence_documents: List[int]
    assessor_name: str
    next_review_date: Optional[datetime]
    actions_required: List[str]
    created_at: datetime
    created_by: int

class ComplianceActionCreate(BaseModel):
    """Create a compliance action"""
    assessment_id: int = Field(..., description="Related assessment ID")
    title: str = Field(..., min_length=1, max_length=200, description="Action title")
    description: str = Field(..., min_length=1, max_length=2000, description="Action description")
    priority: str = Field("medium", regex="^(low|medium|high|critical)$", description="Priority level")
    assigned_to: str = Field(..., min_length=1, max_length=100, description="Person assigned")
    due_date: datetime = Field(..., description="Due date for completion")
    estimated_cost: Optional[float] = Field(None, ge=0, description="Estimated cost")

class ComplianceActionResponse(BaseModel):
    """Compliance action response"""
    id: int
    assessment_id: int
    title: str
    description: str
    priority: str
    assigned_to: str
    due_date: datetime
    estimated_cost: Optional[float]
    status: str
    completion_date: Optional[datetime]
    actual_cost: Optional[float]
    created_at: datetime

class ComplianceDashboard(BaseModel):
    """Compliance dashboard data"""
    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    pending_count: int
    overdue_actions: int
    upcoming_reviews: int
    compliance_rate: float
    standards_breakdown: Dict[str, Dict[str, int]]
    recent_assessments: List[ComplianceAssessmentResponse]
    priority_actions: List[ComplianceActionResponse]

class ComplianceReport(BaseModel):
    """Compliance report"""
    report_id: str
    generated_date: datetime
    period_start: datetime
    period_end: datetime
    facility_ids: Optional[List[int]]
    standards: List[ComplianceStandard]
    summary: Dict[str, Any]
    requirements: List[ComplianceRequirementResponse]
    assessments: List[ComplianceAssessmentResponse]
    actions: List[ComplianceActionResponse]

@router.get("/dashboard", response_model=ComplianceDashboard)
async def get_compliance_dashboard(
    facility_id: Optional[int] = QueryParam(None, description="Filter by facility"),
    standard: Optional[ComplianceStandard] = QueryParam(None, description="Filter by standard"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get compliance dashboard with key metrics and summaries.
    """
    try:
        compliance_service = ComplianceManagementService()

        # Build filters based on user permissions
        filters = await build_compliance_filters(current_user, facility_id, standard)

        dashboard_data = await compliance_service.get_dashboard_data(filters, db)

        return dashboard_data

    except Exception as e:
        logger.error(f"Error getting compliance dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load compliance dashboard"
        )

@router.get("/requirements", response_model=List[ComplianceRequirementResponse])
async def get_compliance_requirements(
    standard: Optional[ComplianceStandard] = QueryParam(None, description="Filter by standard"),
    status: Optional[ComplianceStatus] = QueryParam(None, description="Filter by status"),
    facility_id: Optional[int] = QueryParam(None, description="Filter by facility"),
    priority: Optional[str] = QueryParam(None, description="Filter by priority"),
    page: int = QueryParam(1, ge=1, description="Page number"),
    page_size: int = QueryParam(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get compliance requirements with filtering and pagination.
    """
    try:
        # Build query filters
        filters = await build_compliance_filters(current_user, facility_id, standard)
        if status:
            filters["status"] = status
        if priority:
            filters["priority"] = priority

        # Get requirements
        requirements = db.query(ComplianceRequirement).filter_by(**filters)

        # Apply pagination
        total = requirements.count()
        requirements = requirements.offset((page - 1) * page_size).limit(page_size).all()

        return requirements

    except Exception as e:
        logger.error(f"Error getting compliance requirements: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance requirements"
        )

@router.post("/requirements", response_model=ComplianceRequirementResponse, status_code=status.HTTP_201_CREATED)
@require_permission("compliance_manage")
async def create_compliance_requirement(
    requirement: ComplianceRequirementCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new compliance requirement.
    """
    try:
        # Validate facility access if specified
        if requirement.facility_id:
            await validate_facility_access(current_user, requirement.facility_id)

        # Validate compliance data
        if not validate_compliance_data(requirement.dict()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid compliance requirement data"
            )

        # Create requirement
        db_requirement = ComplianceRequirement(
            **requirement.dict(),
            created_by=current_user.id,
            status=ComplianceStatus.PENDING
        )

        db.add(db_requirement)
        db.commit()
        db.refresh(db_requirement)

        logger.info(f"Created compliance requirement {db_requirement.id} by user {current_user.username}")

        return db_requirement

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating compliance requirement: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create compliance requirement"
        )

@router.get("/requirements/{requirement_id}", response_model=ComplianceRequirementResponse)
async def get_compliance_requirement(
    requirement_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific compliance requirement by ID.
    """
    try:
        requirement = db.query(ComplianceRequirement).filter(
            ComplianceRequirement.id == requirement_id
        ).first()

        if not requirement:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Compliance requirement not found"
            )

        # Check facility access
        if requirement.facility_id:
            await validate_facility_access(current_user, requirement.facility_id)

        return requirement

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compliance requirement {requirement_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance requirement"
        )

@router.put("/requirements/{requirement_id}", response_model=ComplianceRequirementResponse)
@require_permission("compliance_manage")
async def update_compliance_requirement(
    requirement_id: int,
    requirement_update: ComplianceRequirementUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a compliance requirement.
    """
    try:
        requirement = db.query(ComplianceRequirement).filter(
            ComplianceRequirement.id == requirement_id
        ).first()

        if not requirement:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Compliance requirement not found"
            )

        # Check facility access
        if requirement.facility_id:
            await validate_facility_access(current_user, requirement.facility_id)

        # Update fields
        update_data = requirement_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(requirement, field, value)

        requirement.updated_at = datetime.utcnow()
        requirement.updated_by = current_user.id

        db.commit()
        db.refresh(requirement)

        logger.info(f"Updated compliance requirement {requirement_id} by user {current_user.username}")

        return requirement

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating compliance requirement {requirement_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update compliance requirement"
        )

@router.post("/assessments", response_model=ComplianceAssessmentResponse, status_code=status.HTTP_201_CREATED)
@require_permission("compliance_assess")
async def create_compliance_assessment(
    assessment: ComplianceAssessmentCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new compliance assessment.
    """
    try:
        # Validate requirement exists
        requirement = db.query(ComplianceRequirement).filter(
            ComplianceRequirement.id == assessment.requirement_id
        ).first()

        if not requirement:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Compliance requirement not found"
            )

        # Check facility access
        if requirement.facility_id:
            await validate_facility_access(current_user, requirement.facility_id)

        # Create assessment
        db_assessment = ComplianceAssessment(
            **assessment.dict(),
            created_by=current_user.id
        )

        db.add(db_assessment)

        # Update requirement status and review date
        requirement.status = assessment.status
        requirement.last_assessment_date = assessment.assessment_date
        if assessment.next_review_date:
            requirement.next_review_date = assessment.next_review_date

        db.commit()
        db.refresh(db_assessment)

        # Create required actions if any
        if assessment.actions_required:
            background_tasks.add_task(
                create_compliance_actions,
                db_assessment.id,
                assessment.actions_required,
                current_user.id,
                db
            )

        logger.info(f"Created compliance assessment {db_assessment.id} by user {current_user.username}")

        return db_assessment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating compliance assessment: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create compliance assessment"
        )

@router.get("/assessments", response_model=List[ComplianceAssessmentResponse])
async def get_compliance_assessments(
    requirement_id: Optional[int] = QueryParam(None, description="Filter by requirement"),
    status: Optional[ComplianceStatus] = QueryParam(None, description="Filter by status"),
    start_date: Optional[datetime] = QueryParam(None, description="Start date filter"),
    end_date: Optional[datetime] = QueryParam(None, description="End date filter"),
    page: int = QueryParam(1, ge=1, description="Page number"),
    page_size: int = QueryParam(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get compliance assessments with filtering and pagination.
    """
    try:
        query = db.query(ComplianceAssessment)

        # Apply filters
        if requirement_id:
            query = query.filter(ComplianceAssessment.requirement_id == requirement_id)
        if status:
            query = query.filter(ComplianceAssessment.status == status)
        if start_date:
            query = query.filter(ComplianceAssessment.assessment_date >= start_date)
        if end_date:
            query = query.filter(ComplianceAssessment.assessment_date <= end_date)

        # Apply pagination
        assessments = query.offset((page - 1) * page_size).limit(page_size).all()

        return assessments

    except Exception as e:
        logger.error(f"Error getting compliance assessments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance assessments"
        )

@router.post("/actions", response_model=ComplianceActionResponse, status_code=status.HTTP_201_CREATED)
@require_permission("compliance_manage")
async def create_compliance_action(
    action: ComplianceActionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new compliance action.
    """
    try:
        # Validate assessment exists
        assessment = db.query(ComplianceAssessment).filter(
            ComplianceAssessment.id == action.assessment_id
        ).first()

        if not assessment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Compliance assessment not found"
            )

        # Create action
        db_action = ComplianceAction(
            **action.dict(),
            status="open",
            created_by=current_user.id
        )

        db.add(db_action)
        db.commit()
        db.refresh(db_action)

        logger.info(f"Created compliance action {db_action.id} by user {current_user.username}")

        return db_action

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating compliance action: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create compliance action"
        )

@router.get("/actions", response_model=List[ComplianceActionResponse])
async def get_compliance_actions(
    assessment_id: Optional[int] = QueryParam(None, description="Filter by assessment"),
    status: Optional[str] = QueryParam(None, description="Filter by status"),
    assigned_to: Optional[str] = QueryParam(None, description="Filter by assignee"),
    overdue_only: bool = QueryParam(False, description="Show only overdue actions"),
    page: int = QueryParam(1, ge=1, description="Page number"),
    page_size: int = QueryParam(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get compliance actions with filtering and pagination.
    """
    try:
        query = db.query(ComplianceAction)

        # Apply filters
        if assessment_id:
            query = query.filter(ComplianceAction.assessment_id == assessment_id)
        if status:
            query = query.filter(ComplianceAction.status == status)
        if assigned_to:
            query = query.filter(ComplianceAction.assigned_to.ilike(f"%{assigned_to}%"))
        if overdue_only:
            query = query.filter(
                ComplianceAction.due_date < datetime.utcnow(),
                ComplianceAction.status != "completed"
            )

        # Apply pagination
        actions = query.offset((page - 1) * page_size).limit(page_size).all()

        return actions

    except Exception as e:
        logger.error(f"Error getting compliance actions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance actions"
        )

@router.post("/reports/generate")
@require_permission("compliance_report")
async def generate_compliance_report(
    facility_ids: Optional[List[int]] = QueryParam(None, description="Facility IDs to include"),
    standards: Optional[List[ComplianceStandard]] = QueryParam(None, description="Standards to include"),
    period_start: datetime = QueryParam(..., description="Report period start"),
    period_end: datetime = QueryParam(..., description="Report period end"),
    format: str = QueryParam("pdf", regex="^(pdf|excel|json)$", description="Report format"),
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a comprehensive compliance report.
    """
    try:
        # Validate date range
        if period_end <= period_start:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )

        # Generate report ID
        report_id = f"compliance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Start background report generation
        background_tasks.add_task(
            generate_compliance_report_task,
            report_id,
            facility_ids,
            standards,
            period_start,
            period_end,
            format,
            current_user.id,
            db
        )

        return {
            "message": "Report generation started",
            "report_id": report_id,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting compliance report generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start report generation"
        )

# Helper functions

async def build_compliance_filters(user: User, facility_id: Optional[int], standard: Optional[ComplianceStandard]) -> Dict[str, Any]:
    """Build compliance query filters based on user permissions."""
    filters = {}

    # Apply facility access restrictions
    user_facilities = user.facilities_access or []
    if facility_id:
        if facility_id not in user_facilities:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to requested facility"
            )
        filters["facility_id"] = facility_id
    elif user_facilities:
        filters["facility_id__in"] = user_facilities

    if standard:
        filters["standard"] = standard

    return filters

async def validate_facility_access(user: User, facility_id: int) -> None:
    """Validate user has access to specified facility."""
    user_facilities = user.facilities_access or []
    if facility_id not in user_facilities:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to requested facility"
        )

async def create_compliance_actions(
    assessment_id: int,
    actions: List[str],
    created_by: int,
    db: Session
) -> None:
    """Create compliance actions from assessment requirements."""
    try:
        for action_desc in actions:
            action = ComplianceAction(
                assessment_id=assessment_id,
                title=f"Action required: {action_desc[:50]}...",
                description=action_desc,
                priority="medium",
                assigned_to="TBD",
                due_date=datetime.utcnow() + timedelta(days=30),
                status="open",
                created_by=created_by
            )
            db.add(action)

        db.commit()
        logger.info(f"Created {len(actions)} compliance actions for assessment {assessment_id}")

    except Exception as e:
        logger.error(f"Error creating compliance actions: {str(e)}")
        db.rollback()

async def generate_compliance_report_task(
    report_id: str,
    facility_ids: Optional[List[int]],
    standards: Optional[List[ComplianceStandard]],
    period_start: datetime,
    period_end: datetime,
    format: str,
    user_id: int,
    db: Session
) -> None:
    """Background task to generate compliance report."""
    try:
        logger.info(f"Starting compliance report generation: {report_id}")

        # Generate report using compliance service
        compliance_service = ComplianceManagementService()
        report_data = await compliance_service.generate_report(
            facility_ids=facility_ids,
            standards=standards,
            period_start=period_start,
            period_end=period_end,
            format=format,
            db=db
        )

        # Save report to storage and update status
        # Implementation would depend on your file storage setup

        logger.info(f"Completed compliance report generation: {report_id}")

    except Exception as e:
        logger.error(f"Error generating compliance report {report_id}: {str(e)}")