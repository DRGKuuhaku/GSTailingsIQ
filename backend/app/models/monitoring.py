from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MonitoringType(str, Enum):
    PORE_PRESSURE = "pore_pressure"
    WATER_LEVEL = "water_level"
    SETTLEMENT = "settlement"
    DISPLACEMENT = "displacement"
    SEEPAGE = "seepage"
    VIBRATION = "vibration"
    WEATHER = "weather"
    CHEMISTRY = "chemistry"
    FLOW_RATE = "flow_rate"
    OTHER = "other"

class AlertLevel(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"

class MonitoringStation(Base):
    __tablename__ = "monitoring_stations"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Location
    facility_id = Column(String(100), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    elevation = Column(Float)

    # Station details
    monitoring_type = Column(String(50), nullable=False)
    manufacturer = Column(String(100))
    model = Column(String(100))
    serial_number = Column(String(100))
    installation_date = Column(DateTime(timezone=True))

    # Status and configuration
    is_active = Column(Boolean, default=True)
    sampling_interval = Column(Integer)  # in minutes
    alert_thresholds = Column(JSON, default={})
    calibration_data = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_reading_at = Column(DateTime(timezone=True))

    # Relationships
    readings = relationship("MonitoringReading", back_populates="station")

class MonitoringReading(Base):
    __tablename__ = "monitoring_readings"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(50), ForeignKey("monitoring_stations.station_id"), nullable=False)

    # Reading data
    timestamp = Column(DateTime(timezone=True), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)
    quality_code = Column(String(10))  # Good, Fair, Poor, Bad

    # Additional data
    raw_value = Column(Float)
    processed_value = Column(Float)
    correction_factor = Column(Float)

    # Flags and status
    is_validated = Column(Boolean, default=False)
    is_anomaly = Column(Boolean, default=False)
    alert_level = Column(String(20), default=AlertLevel.NORMAL.value)

    # Metadata
    metadata = Column(JSON, default={})
    notes = Column(Text)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    validated_by = Column(Integer, ForeignKey("users.id"))

    # Relationships
    station = relationship("MonitoringStation", back_populates="readings")

class MonitoringAlert(Base):
    __tablename__ = "monitoring_alerts"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(50), ForeignKey("monitoring_stations.station_id"), nullable=False)
    reading_id = Column(Integer, ForeignKey("monitoring_readings.id"))

    # Alert details
    alert_type = Column(String(50), nullable=False)  # threshold_exceeded, data_missing, etc.
    alert_level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)

    # Values
    trigger_value = Column(Float)
    threshold_value = Column(Float)

    # Status
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"))
    acknowledged_at = Column(DateTime(timezone=True))

    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(Integer, ForeignKey("users.id"))
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Pydantic Models
class MonitoringStationCreate(BaseModel):
    station_id: str
    name: str
    description: Optional[str] = None
    facility_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    monitoring_type: MonitoringType
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    sampling_interval: Optional[int] = 60
    alert_thresholds: Dict[str, Any] = {}

class MonitoringStationResponse(BaseModel):
    id: int
    station_id: str
    name: str
    description: Optional[str]
    facility_id: str
    latitude: Optional[float]
    longitude: Optional[float]
    elevation: Optional[float]
    monitoring_type: str
    manufacturer: Optional[str]
    model: Optional[str]
    is_active: bool
    sampling_interval: Optional[int]
    alert_thresholds: Dict[str, Any]
    created_at: datetime
    last_reading_at: Optional[datetime]

    class Config:
        from_attributes = True

class MonitoringReadingCreate(BaseModel):
    station_id: str
    timestamp: datetime
    value: float
    unit: str
    quality_code: Optional[str] = "Good"
    raw_value: Optional[float] = None
    metadata: Dict[str, Any] = {}
    notes: Optional[str] = None

class MonitoringReadingResponse(BaseModel):
    id: int
    station_id: str
    timestamp: datetime
    value: float
    unit: str
    quality_code: Optional[str]
    is_validated: bool
    is_anomaly: bool
    alert_level: str
    metadata: Dict[str, Any]
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class MonitoringAlertResponse(BaseModel):
    id: int
    station_id: str
    alert_type: str
    alert_level: str
    message: str
    trigger_value: Optional[float]
    threshold_value: Optional[float]
    is_active: bool
    is_acknowledged: bool
    is_resolved: bool
    created_at: datetime

    class Config:
        from_attributes = True

class MonitoringDashboard(BaseModel):
    total_stations: int
    active_stations: int
    active_alerts: int
    critical_alerts: int
    recent_readings: List[MonitoringReadingResponse]
    alert_summary: Dict[str, int]
    station_status: List[Dict[str, Any]]
