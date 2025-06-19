from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import logging

from ..core.database import get_db
from ..core.security import get_current_user, check_permissions
from ..models.synthetic_data_models import (
    SyntheticDataSet, SyntheticDataRecord, SyntheticMonitoringData,
    SyntheticDataSetCreate, SyntheticDataSetResponse,
    SyntheticDataGenerationRequest, SyntheticDataGenerationResponse,
    SyntheticDataType
)
from ..models.user import User, UserRole
from ..services.synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize synthetic data generator
data_generator = SyntheticDataGenerator()

@router.get("/datasets", response_model=List[SyntheticDataSetResponse])
async def list_synthetic_datasets(
    skip: int = 0,
    limit: int = 100,
    data_type: Optional[SyntheticDataType] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all synthetic data sets"""

    # Check permissions - Admin and Super Admin can view all, others can view their own
    query = db.query(SyntheticDataSet)

    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        query = query.filter(SyntheticDataSet.created_by == current_user.id)

    if data_type:
        query = query.filter(SyntheticDataSet.data_type == data_type.value)

    datasets = query.filter(SyntheticDataSet.is_active == True).offset(skip).limit(limit).all()

    return datasets

@router.post("/datasets", response_model=SyntheticDataSetResponse)
async def create_synthetic_dataset(
    dataset: SyntheticDataSetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new synthetic data set"""

    # Check permissions - only Admin and Super Admin can create datasets
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create synthetic datasets"
        )

    try:
        db_dataset = SyntheticDataSet(
            name=dataset.name,
            description=dataset.description,
            data_type=dataset.data_type.value,
            record_count=0,  # Will be updated when data is generated
            created_by=current_user.id
        )

        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)

        logger.info(f"Created synthetic dataset: {dataset.name} by user {current_user.id}")

        return db_dataset

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating synthetic dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating synthetic dataset"
        )

@router.post("/generate", response_model=SyntheticDataGenerationResponse)
async def generate_synthetic_data(
    request: SyntheticDataGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate synthetic data based on request parameters"""

    # Check permissions
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to generate synthetic data"
        )

    try:
        # Create dataset record
        dataset_name = f"Generated {request.data_type.value.title()} Data - {current_user.username}"

        db_dataset = SyntheticDataSet(
            name=dataset_name,
            description=f"Auto-generated {request.data_type.value} data with {request.record_count} records",
            data_type=request.data_type.value,
            record_count=request.record_count,
            created_by=current_user.id
        )

        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)

        # Generate data in background
        background_tasks.add_task(
            _generate_data_background,
            db_dataset.id,
            request.data_type,
            request.record_count,
            request.parameters or {}
        )

        logger.info(f"Started generating {request.record_count} {request.data_type.value} records for dataset {db_dataset.id}")

        return SyntheticDataGenerationResponse(
            success=True,
            dataset_id=db_dataset.id,
            record_count=request.record_count,
            data_type=request.data_type.value,
            message=f"Started generating {request.record_count} {request.data_type.value} records"
        )

    except Exception as e:
        logger.error(f"Error initiating synthetic data generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error initiating synthetic data generation"
        )

@router.get("/datasets/{dataset_id}")
async def get_synthetic_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get specific synthetic dataset with records"""

    dataset = db.query(SyntheticDataSet).filter(SyntheticDataSet.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synthetic dataset not found"
        )

    # Check permissions
    if (current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value] and 
        dataset.created_by != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access this dataset"
        )

    # Get sample records (first 10)
    sample_records = db.query(SyntheticDataRecord).filter(
        SyntheticDataRecord.dataset_id == dataset_id
    ).limit(10).all()

    # Parse JSON data for sample records
    sample_data = []
    for record in sample_records:
        try:
            sample_data.append(json.loads(record.record_data))
        except json.JSONDecodeError:
            continue

    return {
        "dataset": dataset,
        "sample_records": sample_data,
        "total_records": len(dataset.records) if dataset.records else 0
    }

@router.delete("/datasets/{dataset_id}")
async def delete_synthetic_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a synthetic dataset"""

    dataset = db.query(SyntheticDataSet).filter(SyntheticDataSet.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synthetic dataset not found"
        )

    # Check permissions - only creator or admin can delete
    if (current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value] and 
        dataset.created_by != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this dataset"
        )

    try:
        # Soft delete
        dataset.is_active = False
        db.commit()

        logger.info(f"Deleted synthetic dataset {dataset_id} by user {current_user.id}")

        return {"message": "Dataset deleted successfully"}

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting synthetic dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting synthetic dataset"
        )

@router.get("/datasets/{dataset_id}/export/{format}")
async def export_synthetic_dataset(
    dataset_id: int,
    format: str,  # json, csv
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export synthetic dataset in specified format"""

    if format not in ["json", "csv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format. Supported formats: json, csv"
        )

    dataset = db.query(SyntheticDataSet).filter(SyntheticDataSet.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synthetic dataset not found"
        )

    # Check permissions
    if (current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value] and 
        dataset.created_by != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to export this dataset"
        )

    try:
        # Get all records
        records = db.query(SyntheticDataRecord).filter(
            SyntheticDataRecord.dataset_id == dataset_id
        ).all()

        # Parse JSON data
        data = []
        for record in records:
            try:
                data.append(json.loads(record.record_data))
            except json.JSONDecodeError:
                continue

        if format == "json":
            return {
                "dataset_info": {
                    "id": dataset.id,
                    "name": dataset.name,
                    "data_type": dataset.data_type,
                    "record_count": len(data)
                },
                "data": data
            }
        elif format == "csv":
            # For CSV, return the data in a format that can be converted
            return {
                "dataset_info": {
                    "id": dataset.id,
                    "name": dataset.name,
                    "data_type": dataset.data_type,
                    "record_count": len(data)
                },
                "csv_data": data,
                "headers": list(data[0].keys()) if data else []
            }

    except Exception as e:
        logger.error(f"Error exporting synthetic dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error exporting synthetic dataset"
        )

@router.get("/preview/{data_type}")
async def preview_synthetic_data(
    data_type: SyntheticDataType,
    count: int = 5,
    current_user: User = Depends(get_current_user)
):
    """Preview synthetic data without saving to database"""

    if count > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Preview count cannot exceed 20 records"
        )

    try:
        # Generate preview data
        if data_type == SyntheticDataType.MONITORING:
            data = data_generator.generate_monitoring_data(facility_count=1, records_per_facility=count)
        elif data_type == SyntheticDataType.DOCUMENT:
            data = data_generator.generate_document_data(count=count)
        elif data_type == SyntheticDataType.COMPLIANCE:
            data = data_generator.generate_compliance_data(count=count)
        elif data_type == SyntheticDataType.GEOTECHNICAL:
            data = data_generator.generate_geotechnical_data(count=count)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Preview not available for data type: {data_type.value}"
            )

        return {
            "data_type": data_type.value,
            "count": len(data),
            "preview_data": data
        }

    except Exception as e:
        logger.error(f"Error generating preview data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating preview data"
        )

# Background task function
def _generate_data_background(dataset_id: int, data_type: SyntheticDataType, 
                            record_count: int, parameters: Dict[str, Any]):
    """Background task to generate synthetic data"""

    try:
        from ..core.database import SessionLocal
        db = SessionLocal()

        # Generate data based on type
        if data_type == SyntheticDataType.MONITORING:
            facility_count = parameters.get('facility_count', 5)
            records_per_facility = record_count // facility_count
            data = data_generator.generate_monitoring_data(
                facility_count=facility_count,
                records_per_facility=records_per_facility
            )
        elif data_type == SyntheticDataType.DOCUMENT:
            data = data_generator.generate_document_data(count=record_count)
        elif data_type == SyntheticDataType.COMPLIANCE:
            data = data_generator.generate_compliance_data(count=record_count)
        elif data_type == SyntheticDataType.GEOTECHNICAL:
            data = data_generator.generate_geotechnical_data(count=record_count)
        else:
            logger.error(f"Unsupported data type for generation: {data_type.value}")
            return

        # Save records to database
        for record_data in data:
            db_record = SyntheticDataRecord(
                dataset_id=dataset_id,
                record_data=json.dumps(record_data)
            )
            db.add(db_record)

        # Update dataset record count
        dataset = db.query(SyntheticDataSet).filter(SyntheticDataSet.id == dataset_id).first()
        if dataset:
            dataset.record_count = len(data)

        db.commit()
        logger.info(f"Successfully generated {len(data)} records for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"Error in background data generation: {e}")
        if 'db' in locals():
            db.rollback()
    finally:
        if 'db' in locals():
            db.close()
