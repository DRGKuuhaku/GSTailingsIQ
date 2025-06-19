# TailingsIQ Synthetic Data Components - Complete Package

## Overview

This package contains all the synthetic data generation components for the TailingsIQ application. Based on the analysis of the provided documents, the following files were referenced but not fully implemented, so complete implementations have been created.

## Components Created

### 1. Database Models (`synthetic_data_models.py`)
**Purpose:** SQLAlchemy models and Pydantic schemas for synthetic data storage and API responses.

**Key Features:**
- `SyntheticDataType` enum for data categories (monitoring, document, compliance, geotechnical, environmental, financial)
- `SyntheticDataSet` model for organizing generated datasets
- `SyntheticDataRecord` model for storing individual data records as JSON
- Specialized models for different data types:
  - `SyntheticMonitoringData` - TSF monitoring parameters
  - `SyntheticDocumentData` - Document metadata
  - `SyntheticComplianceData` - Regulatory compliance tracking
- Pydantic models for API validation and responses

### 2. Data Generator (`synthetic_data_generator.py`)
**Purpose:** Core data generation engine using Faker and custom algorithms.

**Key Features:**
- Realistic TSF monitoring data with proper parameter relationships
- Synthetic document metadata with facility associations
- Compliance data with realistic status distributions
- Geotechnical test data with proper soil parameters
- Facility name generation with mining industry terminology
- Status determination based on safety thresholds
- Export capabilities (JSON, CSV)

**Generated Data Types:**
- **Monitoring Data:** Water levels, pore pressure, settlement, seepage rates, environmental parameters
- **Document Data:** Technical reports, compliance documents, safety protocols
- **Compliance Data:** GISTM requirements, regulatory tracking, risk assessments
- **Geotechnical Data:** Soil tests, material properties, laboratory results

### 3. API Routes (`synthetic_data.py`)
**Purpose:** FastAPI endpoints for synthetic data management.

**Endpoints:**
- `GET /datasets` - List synthetic datasets with filtering
- `POST /datasets` - Create new dataset
- `POST /generate` - Generate synthetic data (background task)
- `GET /datasets/{id}` - Get dataset with sample records
- `DELETE /datasets/{id}` - Soft delete dataset
- `GET /datasets/{id}/export/{format}` - Export dataset (JSON/CSV)
- `GET /preview/{data_type}` - Preview data without saving

**Security Features:**
- Role-based access control (Admin/Super Admin for creation)
- User ownership validation
- Background task processing for large datasets

### 4. Standalone Script (`generate_synthetic_data.py`)
**Purpose:** Command-line tool for data generation outside the web application.

**Usage Examples:**
```bash
# Generate monitoring data
python generate_synthetic_data.py --type monitoring --count 1000 --facilities 5

# Generate all data types
python generate_synthetic_data.py --all --count 500 --format both

# Preview data
python generate_synthetic_data.py --type document --count 10 --preview

# Generate with reproducible results
python generate_synthetic_data.py --type compliance --count 100 --seed 42
```

**Features:**
- Command-line argument parsing
- Multiple output formats (JSON, CSV, both)
- Preview mode for testing
- Batch generation of all data types
- Configurable parameters

### 5. Service Layer (`synthetic_data_service.py`)
**Purpose:** Business logic layer for integrating synthetic data with the main application.

**Key Features:**
- Database integration with transaction management
- Data generation with parameter customization
- Storage in both generic and specialized tables
- Statistical analysis of generated datasets
- Dataset cleanup and maintenance
- Error handling and logging

## Integration with TailingsIQ

### Database Integration
The synthetic data models integrate with the existing TailingsIQ database:
- Foreign key relationships with the User model
- Proper indexing for performance
- Soft delete functionality
- Timestamp tracking

### API Integration
The synthetic data API routes are included in the main FastAPI application:
```python
# In main.py
app.include_router(synthetic_data.router, prefix=f"{settings.API_V1_STR}/synthetic-data", tags=["synthetic-data"])
```

### Dependencies
Required packages are already included in the main requirements.txt:
- `faker==21.0.0` - Fake data generation
- `names==0.3.0` - Name generation

## Data Quality and Realism

### Monitoring Data
- Realistic parameter ranges based on actual TSF monitoring
- Proper correlations between safety indicators
- Time-series data with realistic progression
- Status determination based on industry thresholds

### Document Data
- Industry-appropriate document types and organizations
- Realistic file sizes and page counts
- Proper facility associations
- Content categorization for search

### Compliance Data
- Real regulatory frameworks (GISTM, local codes)
- Realistic compliance status distributions
- Risk level assignments
- Temporal tracking (assessment and review dates)

## Usage in Development and Testing

### Development Environment
```python
from app.services.synthetic_data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
monitoring_data = generator.generate_monitoring_data(facility_count=3, records_per_facility=50)
```

### API Testing
```python
# Create dataset
response = client.post("/api/v1/synthetic-data/datasets", json={
    "name": "Test Monitoring Data",
    "data_type": "monitoring",
    "record_count": 100
})

# Generate data
response = client.post("/api/v1/synthetic-data/generate", json={
    "data_type": "monitoring",
    "record_count": 500,
    "parameters": {"facility_count": 5}
})
```

### Command Line Usage
```bash
# Generate test data for development
python generate_synthetic_data.py --all --count 100 --output-dir ./test_data/

# Create sample data for demos
python generate_synthetic_data.py --type monitoring --count 50 --format json --seed 123
```

## Performance Considerations

### Large Dataset Generation
- Background task processing prevents API timeouts
- Batch processing for database operations
- Memory-efficient data handling
- Progress tracking and logging

### Database Optimization
- Indexed fields for common queries
- JSON storage for flexible data structures
- Separate tables for different data types
- Pagination support for large datasets

## Security and Permissions

### Access Control
- Admin and Super Admin roles can create/generate data
- Users can only access their own datasets
- Role-based filtering of dataset lists
- Secure export functionality

### Data Privacy
- No real facility or personal information
- Configurable data generation parameters
- Audit trail for dataset creation
- Soft delete for data retention

## Deployment Considerations

### Environment Variables
```bash
# Optional: Set custom parameters in .env
SYNTHETIC_DATA_DEFAULT_FACILITY_COUNT=5
SYNTHETIC_DATA_MAX_RECORDS_PER_REQUEST=10000
SYNTHETIC_DATA_CLEANUP_DAYS=30
```

### Database Migrations
The synthetic data models require database migrations:
```bash
alembic revision --autogenerate -m "Add synthetic data models"
alembic upgrade head
```

### Background Tasks
Ensure proper configuration for background task processing in production environments.

## File Organization in Project

```
tailingsiq/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   └── synthetic_data_models.py
│   │   ├── services/
│   │   │   ├── synthetic_data_generator.py
│   │   │   └── synthetic_data_service.py
│   │   └── api/
│   │       └── synthetic_data.py
│   └── generate_synthetic_data.py
```

## Conclusion

This complete synthetic data package provides TailingsIQ with comprehensive data generation capabilities for:
- Development and testing
- Demo environments
- Performance testing
- User training
- Feature validation

All components are production-ready with proper error handling, security controls, and documentation.
