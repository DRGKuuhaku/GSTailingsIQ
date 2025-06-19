import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from faker import Faker
import names
import uuid

class SyntheticDataGenerator:
    """Generate synthetic data for TailingsIQ testing and development"""

    def __init__(self, seed: Optional[int] = None):
        self.faker = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)

    def generate_monitoring_data(self, 
                               facility_count: int = 5, 
                               records_per_facility: int = 100,
                               days_back: int = 365) -> List[Dict[str, Any]]:
        """Generate synthetic monitoring data for TSF facilities"""

        facilities = self._generate_facility_names(facility_count)
        monitoring_data = []

        for facility in facilities:
            base_time = datetime.utcnow() - timedelta(days=days_back)

            for i in range(records_per_facility):
                # Generate realistic time progression
                timestamp = base_time + timedelta(hours=i * 6)  # Every 6 hours

                # Generate realistic monitoring parameters with some variation
                water_level = random.uniform(5.0, 25.0)
                pore_pressure = random.uniform(10.0, 150.0)
                settlement = random.uniform(0.0, 50.0)
                seepage_rate = random.uniform(0.1, 10.0)
                dam_height = random.uniform(20.0, 100.0)
                freeboard = random.uniform(1.0, 10.0)

                # Environmental parameters
                ph_level = random.uniform(6.5, 8.5)
                conductivity = random.uniform(100.0, 2000.0)
                turbidity = random.uniform(0.1, 50.0)
                temperature = random.uniform(5.0, 35.0)

                # Stability parameters
                factor_of_safety = random.uniform(1.2, 3.0)
                slope_angle = random.uniform(20.0, 45.0)

                # Determine status based on parameters
                status, alert_level = self._determine_status(
                    water_level, pore_pressure, freeboard, factor_of_safety
                )

                record = {
                    'facility_id': f"TSF_{facility['id']}",
                    'facility_name': facility['name'],
                    'timestamp': timestamp.isoformat(),
                    'water_level': round(water_level, 2),
                    'pore_pressure': round(pore_pressure, 2),
                    'settlement': round(settlement, 2),
                    'seepage_rate': round(seepage_rate, 2),
                    'dam_height': round(dam_height, 2),
                    'freeboard': round(freeboard, 2),
                    'ph_level': round(ph_level, 2),
                    'conductivity': round(conductivity, 2),
                    'turbidity': round(turbidity, 2),
                    'temperature': round(temperature, 2),
                    'factor_of_safety': round(factor_of_safety, 2),
                    'slope_angle': round(slope_angle, 2),
                    'status': status,
                    'alert_level': alert_level,
                    'created_at': datetime.utcnow().isoformat()
                }

                monitoring_data.append(record)

        return monitoring_data

    def generate_document_data(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate synthetic document metadata"""

        document_types = [
            'Technical Report', 'Monitoring Report', 'Compliance Report',
            'Geotechnical Assessment', 'Environmental Impact Study',
            'Safety Protocol', 'Operating Manual', 'Emergency Response Plan',
            'Dam Safety Review', 'Stability Analysis', 'Risk Assessment'
        ]

        organizations = [
            'Mining Corp Ltd', 'GeoTech Consultants', 'Environmental Solutions Inc',
            'Safety First Engineering', 'Tailings Management Co', 'Regulatory Affairs Ltd',
            'Compliance Solutions', 'Risk Management Group', 'Engineering Dynamics'
        ]

        documents = []
        facilities = self._generate_facility_names(10)

        for i in range(count):
            doc_type = random.choice(document_types)
            facility = random.choice(facilities)

            document = {
                'id': i + 1,
                'title': f"{doc_type} - {facility['name']}",
                'document_type': doc_type,
                'author': names.get_full_name(),
                'organization': random.choice(organizations),
                'creation_date': self.faker.date_between(start_date='-2y', end_date='today').isoformat(),
                'file_size': random.randint(100000, 50000000),  # 100KB to 50MB
                'page_count': random.randint(5, 200),
                'contains_monitoring_data': random.choice([True, False]),
                'contains_compliance_info': random.choice([True, False]),
                'contains_geotechnical_data': random.choice([True, False]),
                'contains_environmental_data': random.choice([True, False]),
                'facility_name': facility['name'],
                'facility_location': facility['location'],
                'report_period': f"{random.randint(2020, 2024)}-Q{random.randint(1, 4)}",
                'created_at': datetime.utcnow().isoformat()
            }

            documents.append(document)

        return documents

    def generate_compliance_data(self, count: int = 30) -> List[Dict[str, Any]]:
        """Generate synthetic compliance data"""

        regulation_types = ['GISTM', 'Local Mining Code', 'Environmental Protection Act',
                           'Dam Safety Regulations', 'Workplace Safety Standards']

        requirements = [
            'Regular monitoring and reporting',
            'Emergency response procedures',
            'Environmental impact assessment',
            'Structural stability analysis',
            'Community consultation',
            'Water quality monitoring',
            'Waste management protocols',
            'Personnel training requirements',
            'Equipment maintenance schedules',
            'Documentation standards'
        ]

        compliance_data = []
        facilities = self._generate_facility_names(8)

        for i in range(count):
            facility = random.choice(facilities)

            compliance = {
                'id': i + 1,
                'facility_id': f"TSF_{facility['id']}",
                'regulation_type': random.choice(regulation_types),
                'requirement_id': f"REQ-{random.randint(1000, 9999)}",
                'requirement_description': random.choice(requirements),
                'compliance_status': random.choices(
                    ['compliant', 'non-compliant', 'pending'],
                    weights=[70, 20, 10]
                )[0],
                'assessment_date': self.faker.date_between(start_date='-1y', end_date='today').isoformat(),
                'next_review_date': self.faker.date_between(start_date='today', end_date='+1y').isoformat(),
                'risk_level': random.choices(
                    ['low', 'medium', 'high', 'critical'],
                    weights=[50, 30, 15, 5]
                )[0],
                'mitigation_measures': self.faker.text(max_nb_chars=200),
                'created_at': datetime.utcnow().isoformat()
            }

            compliance_data.append(compliance)

        return compliance_data

    def generate_geotechnical_data(self, count: int = 40) -> List[Dict[str, Any]]:
        """Generate synthetic geotechnical data"""

        test_types = [
            'Triaxial Compression Test', 'Direct Shear Test', 'Consolidation Test',
            'Permeability Test', 'Proctor Compaction Test', 'Atterberg Limits Test',
            'Particle Size Distribution', 'Specific Gravity Test'
        ]

        soil_types = [
            'Silty Clay', 'Sandy Clay', 'Clay', 'Silt', 'Fine Sand',
            'Coarse Sand', 'Gravel', 'Rock Fill', 'Tailings Material'
        ]

        geotechnical_data = []
        facilities = self._generate_facility_names(6)

        for i in range(count):
            facility = random.choice(facilities)

            data = {
                'id': i + 1,
                'facility_id': f"TSF_{facility['id']}",
                'test_type': random.choice(test_types),
                'soil_type': random.choice(soil_types),
                'sample_depth': random.uniform(0.5, 50.0),
                'moisture_content': random.uniform(5.0, 40.0),
                'dry_density': random.uniform(1.2, 2.2),
                'cohesion': random.uniform(0.0, 100.0),
                'friction_angle': random.uniform(15.0, 45.0),
                'permeability': random.uniform(1e-9, 1e-4),
                'plasticity_index': random.uniform(0.0, 50.0),
                'liquid_limit': random.uniform(20.0, 80.0),
                'specific_gravity': random.uniform(2.4, 2.8),
                'test_date': self.faker.date_between(start_date='-2y', end_date='today').isoformat(),
                'laboratory': random.choice(['GeoLab Inc', 'Soil Testing Co', 'Materials Lab']),
                'created_at': datetime.utcnow().isoformat()
            }

            geotechnical_data.append(data)

        return geotechnical_data

    def _generate_facility_names(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic facility names and locations"""

        mine_names = [
            'Copper Ridge', 'Gold Valley', 'Silver Creek', 'Iron Mountain',
            'Diamond Peak', 'Platinum Hills', 'Zinc Harbor', 'Lead Canyon',
            'Nickel Point', 'Cobalt Bay', 'Titanium Ridge', 'Molybdenum Valley'
        ]

        locations = [
            'Western Australia', 'British Columbia', 'Nevada', 'Chile',
            'Peru', 'Queensland', 'Ontario', 'Arizona', 'Colorado', 'Utah'
        ]

        facilities = []
        for i in range(count):
            facility = {
                'id': f"{i+1:03d}",
                'name': f"{random.choice(mine_names)} TSF",
                'location': random.choice(locations),
                'operator': f"{random.choice(mine_names)} Mining Corp"
            }
            facilities.append(facility)

        return facilities

    def _determine_status(self, water_level: float, pore_pressure: float, 
                         freeboard: float, factor_of_safety: float) -> tuple:
        """Determine facility status based on monitoring parameters"""

        # Critical conditions
        if (water_level > 20.0 or pore_pressure > 120.0 or 
            freeboard < 2.0 or factor_of_safety < 1.3):
            return 'critical', 3

        # Warning conditions
        if (water_level > 15.0 or pore_pressure > 100.0 or 
            freeboard < 3.0 or factor_of_safety < 1.5):
            return 'warning', 2

        # Caution conditions
        if (water_level > 10.0 or pore_pressure > 80.0 or 
            freeboard < 5.0 or factor_of_safety < 2.0):
            return 'caution', 1

        # Normal conditions
        return 'normal', 0

    def generate_all_data(self, 
                         monitoring_records: int = 500,
                         document_records: int = 50,
                         compliance_records: int = 30,
                         geotechnical_records: int = 40) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all types of synthetic data"""

        return {
            'monitoring_data': self.generate_monitoring_data(records_per_facility=monitoring_records//5),
            'document_data': self.generate_document_data(document_records),
            'compliance_data': self.generate_compliance_data(compliance_records),
            'geotechnical_data': self.generate_geotechnical_data(geotechnical_records)
        }

    def export_to_json(self, data: Dict[str, Any], filename: str) -> None:
        """Export data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> None:
        """Export data to CSV file"""
        if not data:
            return

        import csv

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
