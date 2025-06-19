from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    TECHNICAL_REPORT = "technical_report"
    DESIGN_DOCUMENT = "design_document"
    MONITORING_REPORT = "monitoring_report"
    COMPLIANCE_DOCUMENT = "compliance_document"
    RISK_ASSESSMENT = "risk_assessment"
    EMERGENCY_PLAN = "emergency_plan"
    INSPECTION_REPORT = "inspection_report"
    PERMIT = "permit"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"

class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)

    # Document metadata
    title = Column(String(500))
    description = Column(Text)
    document_type = Column(String(50), nullable=False, default=DocumentType.OTHER.value)
    status = Column(String(20), nullable=False, default=DocumentStatus.PROCESSING.value)

    # Processing results
    extracted_text = Column(Text)
    extracted_metadata = Column(JSON, default={})
    ai_analysis = Column(JSON, default={})

    # Search and indexing
    search_vector = Column(Text)  # For full-text search
    embedding_vector = Column(JSON)  # For semantic search

    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True))
    last_accessed = Column(DateTime(timezone=True))

    # User associations
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    facility_id = Column(String(100))  # Associated TSF

    # Document relationships
    tags = Column(JSON, default=[])
    related_documents = Column(JSON, default=[])

    # Security and access
    access_level = Column(String(20), default="standard")
    is_confidential = Column(Boolean, default=False)

class DocumentChunk(Base):
    """Chunks of documents for vector storage"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_embedding = Column(JSON)
    chunk_metadata = Column(JSON, default={})

    # Relationship
    document = relationship("Document")

# Pydantic Models
class DocumentCreate(BaseModel):
    filename: str
    document_type: DocumentType
    description: Optional[str] = None
    facility_id: Optional[str] = None
    tags: List[str] = []
    access_level: str = "standard"
    is_confidential: bool = False

class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: Optional[DocumentType] = None
    tags: Optional[List[str]] = None
    facility_id: Optional[str] = None
    access_level: Optional[str] = None
    is_confidential: Optional[bool] = None

class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    title: Optional[str]
    description: Optional[str]
    document_type: DocumentType
    status: DocumentStatus
    file_size: int
    content_type: str
    uploaded_at: datetime
    processed_at: Optional[datetime]
    uploaded_by: Optional[int]
    facility_id: Optional[str]
    tags: List[str]
    access_level: str
    is_confidential: bool

    class Config:
        from_attributes = True

class DocumentSearchQuery(BaseModel):
    query: str
    document_type: Optional[DocumentType] = None
    facility_id: Optional[str] = None
    tags: List[str] = []
    limit: int = Field(default=10, le=100)
    semantic_search: bool = True

class DocumentAnalysisResult(BaseModel):
    document_id: int
    summary: str
    key_findings: List[str]
    compliance_status: Dict[str, Any]
    risk_indicators: List[str]
    recommendations: List[str]
    confidence_score: float
