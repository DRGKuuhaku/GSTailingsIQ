"""
Synthetic Data Service for TailingsIQ

This service provides methods for integrating synthetic data generation
with the main TailingsIQ application, including database operations
and background task management.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from ..models.synthetic_data_models import (
    SyntheticDataSet, SyntheticDataRecord, SyntheticMonitoringData,
    SyntheticDocumentData, SyntheticComplianceData, SyntheticDataType
)
from ..models.user import User
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

class SyntheticDataService:
    """Service for managing synthetic data operations"""

    def __init__(self):
        self.generator = SyntheticDataGenerator()

    async def create_dataset(self, db: Session, dataset_name: str, 
                           data_type: SyntheticDataType, description: str,
                           created_by: int) -> SyntheticDataSet:
        """Create a new synthetic dataset"""

        try:
            dataset = SyntheticDataSet(
                name=dataset_name,
                description=description,
                data_type=data_type.value,
                created_by=created_by
            )

            db.add(dataset)
            db.commit()
            db.refresh(dataset)

            logger.info(f"Created synthetic dataset: {dataset_name}")
            return dataset

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating synthetic dataset: {e}")
            raise

    async def generate_and_store_data(self, db: Session, dataset: SyntheticDataSet,
                                    record_count: int, parameters: Dict[str, Any] = None) -> int:
        """Generate synthetic data and store in database"""

        try:
            parameters = parameters or {}

            # Generate data based on type
            if dataset.data_type == SyntheticDataType.MONITORING.value:
                data = await self._generate_monitoring_data(record_count, parameters)
            elif dataset.data_type == SyntheticDataType.DOCUMENT.value:
                data = await self._generate_document_data(record_count, parameters)
            elif dataset.data_type == SyntheticDataType.COMPLIANCE.value:
                data = await self._generate_compliance_data(record_count, parameters)
            elif dataset.data_type == SyntheticDataType.GEOTECHNICAL.value:
                data = await self._generate_geotechnical_data(record_count, parameters)
            else:
                raise ValueError(f"Unsupported data type: {dataset.data_type}")

            # Store data in database
            stored_count = 0

            # Store as generic records
            for record_data in data:
                db_record = SyntheticDataRecord(
                    dataset_id=dataset.id,
                    record_data=json.dumps(record_data)
                )
                db.add(db_record)
                stored_count += 1

            # Also store in specific tables for better querying
            if dataset.data_type == SyntheticDataType.MONITORING.value:
                await self._store_monitoring_data(db, data)
            elif dataset.data_type == SyntheticDataType.DOCUMENT.value:
                await self._store_document_data(db, data)
            elif dataset.data_type == SyntheticDataType.COMPLIANCE.value:
                await self._store_compliance_data(db, data)

            # Update dataset record count
            dataset.record_count = stored_count

            db.commit()
            logger.info(f"Generated and stored {stored_count} records for dataset {dataset.id}")

            return stored_count

        except Exception as e:
            db.rollback()
            logger.error(f"Error generating and storing synthetic data: {e}")
            raise

    async def _generate_monitoring_data(self, count: int, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate monitoring data with parameters"""

        facility_count = parameters.get('facility_count', 5)
        days_back = parameters.get('days_back', 365)
        records_per_facility = max(1, count // facility_count)

        return self.generator.generate_monitoring_data(
            facility_count=facility_count,
            records_per_facility=records_per_facility,
            days_back=days_back
        )

    async def _generate_document_data(self, count: int, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate document data with parameters"""
        return self.generator.generate_document_data(count=count)

    async def _generate_compliance_data(self, count: int, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compliance data with parameters"""
        return self.generator.generate_compliance_data(count=count)

    async def _generate_geotechnical_data(self, count: int, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate geotechnical data with parameters"""
        return self.generator.generate_geotechnical_data(count=count)

    async def _store_monitoring_data(self, db: Session, data: List[Dict[str, Any]]) -> None:
        """Store monitoring data in specific table"""

        for record in data:
            monitoring_record = SyntheticMonitoringData(
                facility_id=record['facility_id'],
                facility_name=record['facility_name'],
                timestamp=datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')),
                water_level=record['water_level'],
                pore_pressure=record['pore_pressure'],
                settlement=record['settlement'],
                seepage_rate=record['seepage_rate'],
                dam_height=record['dam_height'],
                freeboard=record['freeboard'],
                ph_level=record['ph_level'],
                conductivity=record['conductivity'],
                turbidity=record['turbidity'],
                temperature=record['temperature'],
                factor_of_safety=record['factor_of_safety'],
                slope_angle=record['slope_angle'],
                status=record['status'],
                alert_level=record['alert_level']
            )
            db.add(monitoring_record)

    async def _store_document_data(self, db: Session, data: List[Dict[str, Any]]) -> None:
        """Store document data in specific table"""

        for record in data:
            document_record = SyntheticDocumentData(
                title=record['title'],
                document_type=record['document_type'],
                author=record['author'],
                organization=record['organization'],
                creation_date=datetime.fromisoformat(record['creation_date']),
                file_size=record['file_size'],
                page_count=record['page_count'],
                contains_monitoring_data=record['contains_monitoring_data'],
                contains_compliance_info=record['contains_compliance_info'],
                contains_geotechnical_data=record['contains_geotechnical_data'],
                contains_environmental_data=record['contains_environmental_data'],
                facility_name=record['facility_name'],
                facility_location=record['facility_location'],
                report_period=record['report_period']
            )
            db.add(document_record)

    async def _store_compliance_data(self, db: Session, data: List[Dict[str, Any]]) -> None:
        """Store compliance data in specific table"""

        for record in data:
            compliance_record = SyntheticComplianceData(
                facility_id=record['facility_id'],
                regulation_type=record['regulation_type'],
                requirement_id=record['requirement_id'],
                requirement_description=record['requirement_description'],
                compliance_status=record['compliance_status'],
                assessment_date=datetime.fromisoformat(record['assessment_date']),
                next_review_date=datetime.fromisoformat(record['next_review_date']),
                risk_level=record['risk_level'],
                mitigation_measures=record['mitigation_measures']
            )
            db.add(compliance_record)

    async def get_dataset_statistics(self, db: Session, dataset_id: int) -> Dict[str, Any]:
        """Get statistics for a synthetic dataset"""

        dataset = db.query(SyntheticDataSet).filter(SyntheticDataSet.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Get record count
        record_count = db.query(SyntheticDataRecord).filter(
            SyntheticDataRecord.dataset_id == dataset_id
        ).count()

        # Get sample data for analysis
        sample_records = db.query(SyntheticDataRecord).filter(
            SyntheticDataRecord.dataset_id == dataset_id
        ).limit(100).all()

        # Parse sample data
        sample_data = []
        for record in sample_records:
            try:
                sample_data.append(json.loads(record.record_data))
            except json.JSONDecodeError:
                continue

        # Generate statistics based on data type
        stats = {
            "dataset_id": dataset_id,
            "name": dataset.name,
            "data_type": dataset.data_type,
            "total_records": record_count,
            "created_at": dataset.created_at,
            "sample_size": len(sample_data)
        }

        if dataset.data_type == SyntheticDataType.MONITORING.value and sample_data:
            stats.update(self._calculate_monitoring_stats(sample_data))
        elif dataset.data_type == SyntheticDataType.DOCUMENT.value and sample_data:
            stats.update(self._calculate_document_stats(sample_data))
        elif dataset.data_type == SyntheticDataType.COMPLIANCE.value and sample_data:
            stats.update(self._calculate_compliance_stats(sample_data))

        return stats

    def _calculate_monitoring_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for monitoring data"""

        if not data:
            return {}

        # Calculate averages and ranges
        stats = {}

        # Numeric fields to analyze
        numeric_fields = ['water_level', 'pore_pressure', 'settlement', 'seepage_rate',
                         'dam_height', 'freeboard', 'ph_level', 'conductivity', 
                         'turbidity', 'temperature', 'factor_of_safety', 'slope_angle']

        for field in numeric_fields:
            values = [record.get(field, 0) for record in data if record.get(field) is not None]
            if values:
                stats[f"{field}_avg"] = round(sum(values) / len(values), 2)
                stats[f"{field}_min"] = round(min(values), 2)
                stats[f"{field}_max"] = round(max(values), 2)

        # Status distribution
        statuses = [record.get('status') for record in data]
        status_counts = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        stats['status_distribution'] = status_counts

        # Facility count
        facilities = set(record.get('facility_id') for record in data)
        stats['unique_facilities'] = len(facilities)

        return stats

    def _calculate_document_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for document data"""

        if not data:
            return {}

        stats = {}

        # Document type distribution
        doc_types = [record.get('document_type') for record in data]
        type_counts = {}
        for doc_type in doc_types:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        stats['document_type_distribution'] = type_counts

        # Average file size and page count
        file_sizes = [record.get('file_size', 0) for record in data]
        page_counts = [record.get('page_count', 0) for record in data]

        if file_sizes:
            stats['avg_file_size_mb'] = round(sum(file_sizes) / len(file_sizes) / 1024 / 1024, 2)
        if page_counts:
            stats['avg_page_count'] = round(sum(page_counts) / len(page_counts), 1)

        # Content type analysis
        content_types = ['contains_monitoring_data', 'contains_compliance_info',
                        'contains_geotechnical_data', 'contains_environmental_data']

        for content_type in content_types:
            count = sum(1 for record in data if record.get(content_type, False))
            stats[f"{content_type}_count"] = count

        return stats

    def _calculate_compliance_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for compliance data"""

        if not data:
            return {}

        stats = {}

        # Compliance status distribution
        statuses = [record.get('compliance_status') for record in data]
        status_counts = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        stats['compliance_status_distribution'] = status_counts

        # Risk level distribution
        risk_levels = [record.get('risk_level') for record in data]
        risk_counts = {}
        for risk in risk_levels:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        stats['risk_level_distribution'] = risk_counts

        # Regulation type distribution
        reg_types = [record.get('regulation_type') for record in data]
        reg_counts = {}
        for reg_type in reg_types:
            reg_counts[reg_type] = reg_counts.get(reg_type, 0) + 1
        stats['regulation_type_distribution'] = reg_counts

        return stats

    async def cleanup_old_datasets(self, db: Session, days_old: int = 30) -> int:
        """Clean up old synthetic datasets"""

        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        # Find old datasets
        old_datasets = db.query(SyntheticDataSet).filter(
            SyntheticDataSet.created_at < cutoff_date,
            SyntheticDataSet.is_active == True
        ).all()

        cleaned_count = 0
        for dataset in old_datasets:
            # Soft delete
            dataset.is_active = False
            cleaned_count += 1

        db.commit()
        logger.info(f"Cleaned up {cleaned_count} old synthetic datasets")

        return cleaned_count
