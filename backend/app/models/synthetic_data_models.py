from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum
import uuid

Base = declarative_base()

class SyntheticDataType(str, Enum):
    """Types of synthetic data that can be generated"""
    MONITORING = "monitoring"
    DOCUMENT = "document"
    COMPLIANCE = "compliance"
    GEOTECHNICAL = "geotechnical"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"

class SyntheticDataSet(Base):
    """Model for synthetic data sets"""
    __tablename__ = "synthetic_datasets"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    data_type = Column(String, nullable=False)  # SyntheticDataType
    record_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)

    # Relationships
    records = relationship("SyntheticDataRecord", back_populates="dataset")
    creator = relationship("User", back_populates="synthetic_datasets")

class SyntheticDataRecord(Base):
    """Individual synthetic data records"""
    __tablename__ = "synthetic_data_records"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("synthetic_datasets.id"))
    record_data = Column(Text)  # JSON data as string
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("SyntheticDataSet", back_populates="records")

class SyntheticMonitoringData(Base):
    """Synthetic monitoring data for TSF"""
    __tablename__ = "synthetic_monitoring_data"

    id = Column(Integer, primary_key=True, index=True)
    facility_id = Column(String, nullable=False)
    facility_name = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # Monitoring parameters
    water_level = Column(Float)  # meters
    pore_pressure = Column(Float)  # kPa
    settlement = Column(Float)  # mm
    seepage_rate = Column(Float)  # L/s
    dam_height = Column(Float)  # meters
    freeboard = Column(Float)  # meters

    # Environmental parameters
    ph_level = Column(Float)
    conductivity = Column(Float)  # μS/cm
    turbidity = Column(Float)  # NTU
    temperature = Column(Float)  # Celsius

    # Stability parameters
    factor_of_safety = Column(Float)
    slope_angle = Column(Float)  # degrees

    # Status and alerts
    status = Column(String, default="normal")  # normal, warning, critical
    alert_level = Column(Integer, default=0)  # 0=none, 1=low, 2=medium, 3=high

    created_at = Column(DateTime, default=datetime.utcnow)

class SyntheticDocumentData(Base):
    """Synthetic document metadata"""
    __tablename__ = "synthetic_documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    author = Column(String)
    organization = Column(String)
    creation_date = Column(DateTime)
    file_size = Column(Integer)  # bytes
    page_count = Column(Integer)

    # Content categories
    contains_monitoring_data = Column(Boolean, default=False)
    contains_compliance_info = Column(Boolean, default=False)
    contains_geotechnical_data = Column(Boolean, default=False)
    contains_environmental_data = Column(Boolean, default=False)

    # Synthetic metadata
    facility_name = Column(String)
    facility_location = Column(String)
    report_period = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

class SyntheticComplianceData(Base):
    """Synthetic compliance and regulatory data"""
    __tablename__ = "synthetic_compliance"

    id = Column(Integer, primary_key=True, index=True)
    facility_id = Column(String, nullable=False)
    regulation_type = Column(String, nullable=False)  # GISTM, Local, etc.
    requirement_id = Column(String, nullable=False)
    requirement_description = Column(Text)

    # Compliance status
    compliance_status = Column(String, default="compliant")  # compliant, non-compliant, pending
    assessment_date = Column(DateTime)
    next_review_date = Column(DateTime)

    # Risk assessment
    risk_level = Column(String, default="low")  # low, medium, high, critical
    mitigation_measures = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models for API responses
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class SyntheticDataSetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: SyntheticDataType
    record_count: int = Field(default=100, ge=1, le=10000)

class SyntheticDataSetResponse(BaseModel):
    id: int
    uuid: str
    name: str
    description: Optional[str]
    data_type: str
    record_count: int
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class SyntheticDataGenerationRequest(BaseModel):
    data_type: SyntheticDataType
    record_count: int = Field(default=100, ge=1, le=10000)
    parameters: Optional[dict] = None

class SyntheticDataGenerationResponse(BaseModel):
    success: bool
    dataset_id: int
    record_count: int
    data_type: str
    message: str
