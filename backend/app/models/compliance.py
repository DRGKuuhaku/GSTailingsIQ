from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from enum import Enum

class ComplianceStandard(str, Enum):
    GISTM = "gistm"
    ANCOLD = "ancold"
    CDA = "cda"
    ICOLD = "icold"
    MAC = "mac"
    LOCAL_REGULATION = "local_regulation"
    COMPANY_STANDARD = "company_standard"
    OTHER = "other"

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"

class ComplianceRequirement(Base):
    __tablename__ = "compliance_requirements"

    id = Column(Integer, primary_key=True, index=True)
    requirement_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)

    # Standard details
    standard = Column(String(50), nullable=False)
    section = Column(String(100))
    subsection = Column(String(100))
    version = Column(String(20))

    # Requirement classification
    category = Column(String(100))  # Design, Operation, Monitoring, etc.
    subcategory = Column(String(100))
    risk_level = Column(String(20))  # High, Medium, Low

    # Compliance details
    is_mandatory = Column(Boolean, default=True)
    frequency = Column(String(50))  # Annual, Quarterly, Monthly, Continuous
    due_date_rule = Column(String(200))  # How to calculate due dates

    # Documentation
    guidance_notes = Column(Text)
    references = Column(JSON, default=[])
    related_requirements = Column(JSON, default=[])

    # Status
    is_active = Column(Boolean, default=True)
    effective_date = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ComplianceAssessment(Base):
    __tablename__ = "compliance_assessments"

    id = Column(Integer, primary_key=True, index=True)
    requirement_id = Column(String(100), ForeignKey("compliance_requirements.requirement_id"), nullable=False)
    facility_id = Column(String(100), nullable=False)

    # Assessment details
    assessment_date = Column(DateTime(timezone=True), nullable=False)
    assessor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(30), nullable=False, default=ComplianceStatus.UNDER_REVIEW.value)

    # Evidence and findings
    evidence_provided = Column(Text)
    evidence_documents = Column(JSON, default=[])  # Document IDs
    findings = Column(Text)
    recommendations = Column(Text)

    # Scoring
    compliance_score = Column(Float)  # 0-100
    risk_score = Column(Float)  # 0-100
    confidence_level = Column(Float)  # 0-100

    # Actions required
    actions_required = Column(JSON, default=[])
    due_date = Column(DateTime(timezone=True))

    # Review and approval
    is_reviewed = Column(Boolean, default=False)
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    reviewed_at = Column(DateTime(timezone=True))
    review_comments = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    requirement = relationship("ComplianceRequirement")

class ComplianceAction(Base):
    __tablename__ = "compliance_actions"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("compliance_assessments.id"), nullable=False)

    # Action details
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    action_type = Column(String(50))  # Corrective, Preventive, Improvement
    priority = Column(String(20))  # High, Medium, Low

    # Assignment
    assigned_to = Column(Integer, ForeignKey("users.id"))
    assigned_by = Column(Integer, ForeignKey("users.id"))
    assigned_date = Column(DateTime(timezone=True))
    due_date = Column(DateTime(timezone=True))

    # Status tracking
    status = Column(String(30), default="open")  # open, in_progress, completed, cancelled
    progress_percentage = Column(Integer, default=0)

    # Completion
    completed_date = Column(DateTime(timezone=True))
    completion_notes = Column(Text)
    verification_required = Column(Boolean, default=False)
    verified_by = Column(Integer, ForeignKey("users.id"))
    verified_date = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    assessment = relationship("ComplianceAssessment")

class ComplianceReport(Base):
    __tablename__ = "compliance_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)

    # Report scope
    facility_id = Column(String(100), nullable=False)
    standard = Column(String(50), nullable=False)
    reporting_period_start = Column(DateTime(timezone=True))
    reporting_period_end = Column(DateTime(timezone=True))

    # Report content
    executive_summary = Column(Text)
    overall_status = Column(String(30))
    compliance_percentage = Column(Float)
    risk_rating = Column(String(20))

    # Findings
    key_findings = Column(JSON, default=[])
    recommendations = Column(JSON, default=[])
    action_items = Column(JSON, default=[])

    # Generation details
    generated_by = Column(Integer, ForeignKey("users.id"))
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_by = Column(Integer, ForeignKey("users.id"))
    approved_at = Column(DateTime(timezone=True))

    # File details
    file_path = Column(String(500))
    file_size = Column(Integer)

    # Status
    is_final = Column(Boolean, default=False)
    is_published = Column(Boolean, default=False)

# Pydantic Models
class ComplianceRequirementCreate(BaseModel):
    requirement_id: str
    title: str
    description: str
    standard: ComplianceStandard
    section: Optional[str] = None
    category: Optional[str] = None
    risk_level: Optional[str] = "Medium"
    is_mandatory: bool = True
    frequency: Optional[str] = None

class ComplianceRequirementResponse(BaseModel):
    id: int
    requirement_id: str
    title: str
    description: str
    standard: str
    section: Optional[str]
    category: Optional[str]
    risk_level: Optional[str]
    is_mandatory: bool
    frequency: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ComplianceAssessmentCreate(BaseModel):
    requirement_id: str
    facility_id: str
    assessment_date: datetime
    status: ComplianceStatus
    evidence_provided: Optional[str] = None
    findings: Optional[str] = None
    recommendations: Optional[str] = None
    compliance_score: Optional[float] = None
    due_date: Optional[datetime] = None

class ComplianceAssessmentResponse(BaseModel):
    id: int
    requirement_id: str
    facility_id: str
    assessment_date: datetime
    status: str
    compliance_score: Optional[float]
    risk_score: Optional[float]
    confidence_level: Optional[float]
    due_date: Optional[datetime]
    is_reviewed: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ComplianceActionCreate(BaseModel):
    assessment_id: int
    title: str
    description: str
    action_type: Optional[str] = "Corrective"
    priority: Optional[str] = "Medium"
    assigned_to: Optional[int] = None
    due_date: Optional[datetime] = None

class ComplianceActionResponse(BaseModel):
    id: int
    assessment_id: int
    title: str
    description: str
    action_type: Optional[str]
    priority: Optional[str]
    assigned_to: Optional[int]
    due_date: Optional[datetime]
    status: str
    progress_percentage: int
    created_at: datetime

    class Config:
        from_attributes = True

class ComplianceDashboard(BaseModel):
    facility_id: str
    overall_compliance_percentage: float
    total_requirements: int
    compliant_requirements: int
    non_compliant_requirements: int
    overdue_actions: int
    upcoming_assessments: int
    recent_assessments: List[ComplianceAssessmentResponse]
    compliance_by_standard: Dict[str, float]
    risk_distribution: Dict[str, int]
