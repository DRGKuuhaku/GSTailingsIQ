#!/usr/bin/env python3
"""
TailingsIQ Synthetic Data Generation Script

This script generates synthetic data for testing and development purposes.
Can be run standalone or as part of the deployment process.

Usage:
    python generate_synthetic_data.py --type monitoring --count 1000
    python generate_synthetic_data.py --type document --count 50 --output documents.json
    python generate_synthetic_data.py --all --output-dir ./synthetic_data/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the app directory to Python path if running as standalone
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.synthetic_data_generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for TailingsIQ")

    # Data type options
    parser.add_argument(
        "--type", 
        choices=["monitoring", "document", "compliance", "geotechnical", "all"],
        help="Type of synthetic data to generate"
    )

    # Count options
    parser.add_argument(
        "--count", 
        type=int, 
        default=100,
        help="Number of records to generate (default: 100)"
    )

    # Output options
    parser.add_argument(
        "--output", 
        help="Output file path (default: auto-generated based on type and timestamp)"
    )

    parser.add_argument(
        "--output-dir", 
        default="./synthetic_data",
        help="Output directory for generated files (default: ./synthetic_data)"
    )

    parser.add_argument(
        "--format", 
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)"
    )

    # Generation options
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducible results"
    )

    parser.add_argument(
        "--facilities", 
        type=int, 
        default=5,
        help="Number of facilities for monitoring data (default: 5)"
    )

    # Special options
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Generate all types of synthetic data"
    )

    parser.add_argument(
        "--preview", 
        action="store_true",
        help="Generate a small preview of data without saving"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.type and not args.all:
        parser.error("Must specify either --type or --all")

    if args.preview and args.count > 20:
        args.count = 20
        print("Preview mode: limiting count to 20 records")

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.preview:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}")

    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    print(f"Initialized synthetic data generator" + (f" with seed {args.seed}" if args.seed else ""))

    # Generate data based on type
    if args.all:
        generate_all_data(generator, args, output_dir)
    else:
        generate_single_type(generator, args, output_dir)

def generate_single_type(generator, args, output_dir):
    """Generate a single type of synthetic data"""

    print(f"Generating {args.count} {args.type} records...")

    # Generate data
    if args.type == "monitoring":
        records_per_facility = max(1, args.count // args.facilities)
        data = generator.generate_monitoring_data(
            facility_count=args.facilities,
            records_per_facility=records_per_facility
        )
    elif args.type == "document":
        data = generator.generate_document_data(count=args.count)
    elif args.type == "compliance":
        data = generator.generate_compliance_data(count=args.count)
    elif args.type == "geotechnical":
        data = generator.generate_geotechnical_data(count=args.count)
    else:
        print(f"Unsupported data type: {args.type}")
        return

    print(f"Generated {len(data)} records")

    # Preview mode - just display sample
    if args.preview:
        print("\n=== PREVIEW DATA ===")
        for i, record in enumerate(data[:5]):
            print(f"\nRecord {i+1}:")
            for key, value in record.items():
                print(f"  {key}: {value}")

        if len(data) > 5:
            print(f"\n... and {len(data) - 5} more records")
        return

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_path = Path(args.output)
    else:
        filename = f"synthetic_{args.type}_{timestamp}"
        output_path = output_dir / filename

    # Save in requested format(s)
    if args.format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        generator.export_to_json(data, str(json_path))
        print(f"Saved JSON data to: {json_path}")

    if args.format in ["csv", "both"]:
        csv_path = output_path.with_suffix(".csv")
        generator.export_to_csv(data, str(csv_path))
        print(f"Saved CSV data to: {csv_path}")

def generate_all_data(generator, args, output_dir):
    """Generate all types of synthetic data"""

    print("Generating all types of synthetic data...")

    # Default counts for each type
    counts = {
        "monitoring": args.count,
        "document": max(10, args.count // 10),
        "compliance": max(5, args.count // 20),
        "geotechnical": max(10, args.count // 10)
    }

    # Generate all data
    all_data = generator.generate_all_data(
        monitoring_records=counts["monitoring"],
        document_records=counts["document"],
        compliance_records=counts["compliance"],
        geotechnical_records=counts["geotechnical"]
    )

    print(f"Generated data summary:")
    for data_type, data in all_data.items():
        print(f"  {data_type}: {len(data)} records")

    # Preview mode
    if args.preview:
        print("\n=== PREVIEW DATA ===")
        for data_type, data in all_data.items():
            print(f"\n{data_type.upper()} (first 2 records):")
            for i, record in enumerate(data[:2]):
                print(f"  Record {i+1}: {list(record.keys())}")
        return

    # Save all data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save combined data
    if args.format in ["json", "both"]:
        combined_path = output_dir / f"synthetic_all_data_{timestamp}.json"
        generator.export_to_json(all_data, str(combined_path))
        print(f"Saved combined JSON data to: {combined_path}")

    # Save individual files
    for data_type, data in all_data.items():
        filename = f"synthetic_{data_type}_{timestamp}"

        if args.format in ["json", "both"]:
            json_path = output_dir / f"{filename}.json"
            generator.export_to_json(data, str(json_path))
            print(f"Saved {data_type} JSON to: {json_path}")

        if args.format in ["csv", "both"]:
            csv_path = output_dir / f"{filename}.csv"
            generator.export_to_csv(data, str(csv_path))
            print(f"Saved {data_type} CSV to: {csv_path}")

def generate_sample_data():
    """Generate sample data for development/testing"""

    print("Generating sample data for development...")

    generator = SyntheticDataGenerator(seed=42)  # Fixed seed for consistency

    # Generate small amounts of each type
    sample_data = {
        "monitoring_data": generator.generate_monitoring_data(facility_count=2, records_per_facility=10),
        "document_data": generator.generate_document_data(count=5),
        "compliance_data": generator.generate_compliance_data(count=3),
        "geotechnical_data": generator.generate_geotechnical_data(count=5)
    }

    # Save to a single file
    output_path = "./sample_data.json"
    generator.export_to_json(sample_data, output_path)
    print(f"Sample data saved to: {output_path}")

    # Print summary
    print("\nSample data summary:")
    for data_type, data in sample_data.items():
        print(f"  {data_type}: {len(data)} records")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
