"""
TailingsIQ - TSF Monitoring Data API Endpoints

This module provides FastAPI endpoints for managing and retrieving
tailings storage facility monitoring data including sensor readings,
alerts, and dashboard information.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncio

from ..core.database import get_db
from ..core.security import get_current_user, require_permission
from ..models.user import User
from ..models.monitoring import (
    MonitoringStation, MonitoringReading, MonitoringAlert,
    MonitoringStationCreate, MonitoringStationUpdate, MonitoringStationResponse,
    MonitoringReadingCreate, MonitoringReadingResponse,
    MonitoringAlertResponse, MonitoringDashboard,
    MonitoringType, AlertLevel
)
from ..services.data_integration import DataIntegrationService
from ..services.multimodal_analyzer import MultiModalAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
data_integration = DataIntegrationService()
analyzer = MultiModalAnalyzer()

@router.get("/stations", response_model=List[MonitoringStationResponse])
async def list_monitoring_stations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    facility_id: Optional[str] = None,
    monitoring_type: Optional[MonitoringType] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List monitoring stations with filtering and pagination.

    Args:
        skip: Number of stations to skip
        limit: Maximum number of stations to return
        facility_id: Filter by facility ID
        monitoring_type: Filter by monitoring type
        status: Filter by station status
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[MonitoringStationResponse]: List of monitoring stations
    """
    try:
        query = db.query(MonitoringStation)

        # Apply filters
        if facility_id:
            query = query.filter(MonitoringStation.facility_id == facility_id)

        if monitoring_type:
            query = query.filter(MonitoringStation.monitoring_type == monitoring_type.value)

        if status:
            query = query.filter(MonitoringStation.status == status)

        # Apply pagination and ordering
        stations = query.order_by(MonitoringStation.station_name).offset(skip).limit(limit).all()

        return [MonitoringStationResponse.from_orm(station) for station in stations]

    except Exception as e:
        logger.error(f"Error listing monitoring stations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve monitoring stations: {str(e)}"
        )

@router.post("/stations", response_model=MonitoringStationResponse, status_code=status.HTTP_201_CREATED)
async def create_monitoring_station(
    station_data: MonitoringStationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new monitoring station.

    Args:
        station_data: Station creation data
        current_user: Current authenticated user
        db: Database session

    Returns:
        MonitoringStationResponse: Created station information
    """
    try:
        # Check permissions
        if not require_permission(current_user, "can_manage_monitoring"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to create monitoring stations"
            )

        # Check for duplicate station names
        existing_station = db.query(MonitoringStation).filter(
            and_(
                MonitoringStation.station_name == station_data.station_name,
                MonitoringStation.facility_id == station_data.facility_id
            )
        ).first()

        if existing_station:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Station with this name already exists for this facility"
            )

        # Create station
        db_station = MonitoringStation(**station_data.dict(), created_by=current_user.id)
        db.add(db_station)
        db.commit()
        db.refresh(db_station)

        logger.info(f"Monitoring station created: {db_station.id}")
        return MonitoringStationResponse.from_orm(db_station)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating monitoring station: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create monitoring station: {str(e)}"
        )

@router.get("/stations/{station_id}", response_model=MonitoringStationResponse)
async def get_monitoring_station(
    station_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific monitoring station.

    Args:
        station_id: Station ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        MonitoringStationResponse: Station information
    """
    try:
        station = db.query(MonitoringStation).filter(MonitoringStation.id == station_id).first()
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Monitoring station not found"
            )

        return MonitoringStationResponse.from_orm(station)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving monitoring station: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve monitoring station: {str(e)}"
        )

@router.put("/stations/{station_id}", response_model=MonitoringStationResponse)
async def update_monitoring_station(
    station_id: str,
    station_update: MonitoringStationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a monitoring station.

    Args:
        station_id: Station ID
        station_update: Updated station information
        current_user: Current authenticated user
        db: Database session

    Returns:
        MonitoringStationResponse: Updated station information
    """
    try:
        station = db.query(MonitoringStation).filter(MonitoringStation.id == station_id).first()
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Monitoring station not found"
            )

        # Check permissions
        if not require_permission(current_user, "can_manage_monitoring"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update monitoring stations"
            )

        # Update fields
        update_data = station_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(station, field, value)

        station.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(station)

        logger.info(f"Monitoring station updated: {station_id}")
        return MonitoringStationResponse.from_orm(station)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating monitoring station: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update monitoring station: {str(e)}"
        )

@router.delete("/stations/{station_id}")
async def delete_monitoring_station(
    station_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a monitoring station.

    Args:
        station_id: Station ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        dict: Success message
    """
    try:
        station = db.query(MonitoringStation).filter(MonitoringStation.id == station_id).first()
        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Monitoring station not found"
            )

        # Check permissions
        if not require_permission(current_user, "can_manage_monitoring"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete monitoring stations"
            )

        # Check if station has readings (soft delete if yes)
        reading_count = db.query(MonitoringReading).filter(
            MonitoringReading.station_id == station_id
        ).count()

        if reading_count > 0:
            # Soft delete - mark as inactive
            station.status = "inactive"
            station.updated_at = datetime.utcnow()
            db.commit()
            message = "Monitoring station deactivated (has historical readings)"
        else:
            # Hard delete
            db.delete(station)
            db.commit()
            message = "Monitoring station deleted successfully"

        logger.info(f"Monitoring station deleted: {station_id}")
        return {"message": message}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting monitoring station: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete monitoring station: {str(e)}"
        )

@router.post("/readings", response_model=MonitoringReadingResponse, status_code=status.HTTP_201_CREATED)
async def create_monitoring_reading(
    reading_data: MonitoringReadingCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new monitoring reading.

    Args:
        reading_data: Reading data
        background_tasks: Background task queue
        current_user: Current authenticated user
        db: Database session

    Returns:
        MonitoringReadingResponse: Created reading information
    """
    try:
        # Verify station exists
        station = db.query(MonitoringStation).filter(
            MonitoringStation.id == reading_data.station_id
        ).first()

        if not station:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Monitoring station not found"
            )

        # Create reading
        db_reading = MonitoringReading(**reading_data.dict())
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)

        # Schedule background analysis for anomaly detection
        background_tasks.add_task(
            analyze_reading_background,
            reading_id=db_reading.id,
            station_id=reading_data.station_id
        )

        logger.info(f"Monitoring reading created: {db_reading.id}")
        return MonitoringReadingResponse.from_orm(db_reading)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating monitoring reading: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create monitoring reading: {str(e)}"
        )

@router.get("/readings", response_model=List[MonitoringReadingResponse])
async def list_monitoring_readings(
    station_id: Optional[str] = None,
    facility_id: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List monitoring readings with filtering and pagination.

    Args:
        station_id: Filter by station ID
        facility_id: Filter by facility ID
        date_from: Filter readings from this date
        date_to: Filter readings to this date
        skip: Number of readings to skip
        limit: Maximum number of readings to return
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[MonitoringReadingResponse]: List of readings
    """
    try:
        query = db.query(MonitoringReading)

        # Apply filters
        if station_id:
            query = query.filter(MonitoringReading.station_id == station_id)

        if facility_id:
            # Join with stations to filter by facility
            query = query.join(MonitoringStation).filter(
                MonitoringStation.facility_id == facility_id
            )

        if date_from:
            query = query.filter(MonitoringReading.timestamp >= date_from)

        if date_to:
            query = query.filter(MonitoringReading.timestamp <= date_to)

        # Apply pagination and ordering
        readings = query.order_by(MonitoringReading.timestamp.desc()).offset(skip).limit(limit).all()

        return [MonitoringReadingResponse.from_orm(reading) for reading in readings]

    except Exception as e:
        logger.error(f"Error listing monitoring readings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve monitoring readings: {str(e)}"
        )

@router.get("/readings/latest")
async def get_latest_readings(
    facility_id: Optional[str] = None,
    station_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get latest readings for stations.

    Args:
        facility_id: Filter by facility ID
        station_id: Filter by specific station
        current_user: Current authenticated user
        db: Database session

    Returns:
        Dict: Latest readings by station
    """
    try:
        # Build base query for latest readings
        subquery = db.query(
            MonitoringReading.station_id,
            func.max(MonitoringReading.timestamp).label("latest_timestamp")
        ).group_by(MonitoringReading.station_id).subquery()

        query = db.query(MonitoringReading).join(
            subquery,
            and_(
                MonitoringReading.station_id == subquery.c.station_id,
                MonitoringReading.timestamp == subquery.c.latest_timestamp
            )
        )

        # Apply filters
        if facility_id:
            query = query.join(MonitoringStation).filter(
                MonitoringStation.facility_id == facility_id
            )

        if station_id:
            query = query.filter(MonitoringReading.station_id == station_id)

        latest_readings = query.all()

        # Group by station
        result = {}
        for reading in latest_readings:
            station = db.query(MonitoringStation).filter(
                MonitoringStation.id == reading.station_id
            ).first()

            result[reading.station_id] = {
                "station_name": station.station_name if station else "Unknown",
                "monitoring_type": station.monitoring_type if station else "Unknown",
                "reading": MonitoringReadingResponse.from_orm(reading).dict(),
                "age_minutes": (datetime.utcnow() - reading.timestamp).total_seconds() / 60
            }

        return result

    except Exception as e:
        logger.error(f"Error getting latest readings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest readings: {str(e)}"
        )

@router.get("/alerts", response_model=List[MonitoringAlertResponse])
async def list_monitoring_alerts(
    station_id: Optional[str] = None,
    facility_id: Optional[str] = None,
    alert_level: Optional[AlertLevel] = None,
    status: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List monitoring alerts with filtering and pagination.

    Args:
        station_id: Filter by station ID
        facility_id: Filter by facility ID
        alert_level: Filter by alert level
        status: Filter by alert status
        date_from: Filter alerts from this date
        date_to: Filter alerts to this date
        skip: Number of alerts to skip
        limit: Maximum number of alerts to return
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[MonitoringAlertResponse]: List of alerts
    """
    try:
        query = db.query(MonitoringAlert)

        # Apply filters
        if station_id:
            query = query.filter(MonitoringAlert.station_id == station_id)

        if facility_id:
            # Join with stations to filter by facility
            query = query.join(MonitoringStation).filter(
                MonitoringStation.facility_id == facility_id
            )

        if alert_level:
            query = query.filter(MonitoringAlert.alert_level == alert_level.value)

        if status:
            query = query.filter(MonitoringAlert.status == status)

        if date_from:
            query = query.filter(MonitoringAlert.triggered_at >= date_from)

        if date_to:
            query = query.filter(MonitoringAlert.triggered_at <= date_to)

        # Apply pagination and ordering
        alerts = query.order_by(MonitoringAlert.triggered_at.desc()).offset(skip).limit(limit).all()

        return [MonitoringAlertResponse.from_orm(alert) for alert in alerts]

    except Exception as e:
        logger.error(f"Error listing monitoring alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve monitoring alerts: {str(e)}"
        )

@router.put("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Acknowledge a monitoring alert.

    Args:
        alert_id: Alert ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        dict: Success message
    """
    try:
        alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )

        # Update alert status
        alert.status = "acknowledged"
        alert.acknowledged_by = current_user.id
        alert.acknowledged_at = datetime.utcnow()
        db.commit()

        logger.info(f"Alert acknowledged: {alert_id} by user {current_user.id}")
        return {"message": "Alert acknowledged successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )

@router.get("/dashboard", response_model=MonitoringDashboard)
async def get_monitoring_dashboard(
    facility_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get monitoring dashboard data.

    Args:
        facility_id: Filter by facility ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        MonitoringDashboard: Dashboard data
    """
    try:
        # Build base queries
        station_query = db.query(MonitoringStation)
        alert_query = db.query(MonitoringAlert)

        if facility_id:
            station_query = station_query.filter(MonitoringStation.facility_id == facility_id)
            alert_query = alert_query.join(MonitoringStation).filter(
                MonitoringStation.facility_id == facility_id
            )

        # Get station counts by status
        station_counts = {}
        for status_value in ["active", "inactive", "maintenance", "error"]:
            count = station_query.filter(MonitoringStation.status == status_value).count()
            station_counts[status_value] = count

        # Get active alerts count by level
        active_alerts = alert_query.filter(MonitoringAlert.status == "active")
        alert_counts = {}
        for level in AlertLevel:
            count = active_alerts.filter(MonitoringAlert.alert_level == level.value).count()
            alert_counts[level.value] = count

        # Get recent readings count (last 24 hours)
        recent_time = datetime.utcnow() - timedelta(hours=24)
        recent_readings_count = db.query(MonitoringReading).filter(
            MonitoringReading.timestamp >= recent_time
        ).count()

        # Get data freshness (time since last reading)
        latest_reading = db.query(MonitoringReading).order_by(
            MonitoringReading.timestamp.desc()
        ).first()

        data_freshness = None
        if latest_reading:
            data_freshness = (datetime.utcnow() - latest_reading.timestamp).total_seconds()

        return MonitoringDashboard(
            total_stations=sum(station_counts.values()),
            active_stations=station_counts.get("active", 0),
            station_status_breakdown=station_counts,
            active_alerts=sum(alert_counts.values()),
            alert_level_breakdown=alert_counts,
            recent_readings_count=recent_readings_count,
            data_freshness_seconds=data_freshness,
            last_updated=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard data: {str(e)}"
        )

@router.post("/readings/bulk", status_code=status.HTTP_201_CREATED)
async def create_bulk_readings(
    readings: List[MonitoringReadingCreate],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create multiple monitoring readings in bulk.

    Args:
        readings: List of reading data
        background_tasks: Background task queue
        current_user: Current authenticated user
        db: Database session

    Returns:
        dict: Success message with count
    """
    try:
        # Validate all station IDs exist
        station_ids = list(set(reading.station_id for reading in readings))
        existing_stations = db.query(MonitoringStation.id).filter(
            MonitoringStation.id.in_(station_ids)
        ).all()
        existing_station_ids = {station.id for station in existing_stations}

        # Filter out readings for non-existent stations
        valid_readings = [
            reading for reading in readings 
            if reading.station_id in existing_station_ids
        ]

        if not valid_readings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid station IDs found"
            )

        # Create readings in batches
        batch_size = 1000
        created_count = 0

        for i in range(0, len(valid_readings), batch_size):
            batch = valid_readings[i:i + batch_size]
            db_readings = [MonitoringReading(**reading.dict()) for reading in batch]
            db.add_all(db_readings)
            created_count += len(db_readings)

        db.commit()

        # Schedule background analysis for anomaly detection
        for reading in valid_readings:
            background_tasks.add_task(
                analyze_reading_background,
                reading_id=None,  # Will be determined in background task
                station_id=reading.station_id
            )

        logger.info(f"Bulk readings created: {created_count}")
        return {
            "message": f"Successfully created {created_count} readings",
            "created_count": created_count,
            "skipped_count": len(readings) - len(valid_readings)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating bulk readings: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create bulk readings: {str(e)}"
        )

# Background tasks

async def analyze_reading_background(reading_id: Optional[str], station_id: str):
    """Background task to analyze readings for anomalies."""
    try:
        from ..core.database import SessionLocal
        db = SessionLocal()

        try:
            # Get recent readings for this station
            recent_time = datetime.utcnow() - timedelta(hours=24)
            recent_readings = db.query(MonitoringReading).filter(
                and_(
                    MonitoringReading.station_id == station_id,
                    MonitoringReading.timestamp >= recent_time
                )
            ).order_by(MonitoringReading.timestamp.desc()).limit(100).all()

            if len(recent_readings) < 10:
                return  # Not enough data for analysis

            # Prepare data for analysis
            reading_data = []
            for reading in recent_readings:
                reading_data.append({
                    "timestamp": reading.timestamp,
                    "value": reading.value,
                    "metadata": reading.metadata or {}
                })

            # Perform anomaly analysis
            analysis_result = await analyzer.analyze_temporal_patterns(
                data=reading_data,
                pattern_type="anomaly"
            )

            # Check if anomalies detected
            if analysis_result.get("anomalies"):
                await create_anomaly_alerts(station_id, analysis_result, db)

            logger.info(f"Reading analysis completed for station: {station_id}")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error analyzing readings for station {station_id}: {str(e)}")

async def create_anomaly_alerts(station_id: str, analysis_result: Dict, db: Session):
    """Create alerts based on anomaly analysis."""
    try:
        for anomaly in analysis_result.get("anomalies", []):
            # Determine alert level based on anomaly severity
            severity = anomaly.get("severity", 0.5)
            if severity > 0.8:
                alert_level = AlertLevel.CRITICAL
            elif severity > 0.6:
                alert_level = AlertLevel.HIGH
            elif severity > 0.4:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW

            # Check if similar alert already exists (avoid spam)
            existing_alert = db.query(MonitoringAlert).filter(
                and_(
                    MonitoringAlert.station_id == station_id,
                    MonitoringAlert.alert_type == "anomaly_detected",
                    MonitoringAlert.status == "active",
                    MonitoringAlert.triggered_at >= datetime.utcnow() - timedelta(hours=1)
                )
            ).first()

            if existing_alert:
                continue  # Skip if similar alert already exists

            # Create new alert
            alert = MonitoringAlert(
                station_id=station_id,
                alert_type="anomaly_detected",
                alert_level=alert_level.value,
                message=f"Anomaly detected: {anomaly.get('description', 'Unknown anomaly')}",
                details=anomaly,
                triggered_at=datetime.utcnow(),
                status="active"
            )

            db.add(alert)

        db.commit()

    except Exception as e:
        logger.error(f"Error creating anomaly alerts: {str(e)}")
        db.rollback()