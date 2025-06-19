from typing import Any, Dict, List, Optional, Union
import json
import logging
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import mimetypes
import re
from decimal import Decimal

logger = logging.getLogger(__name__)

def generate_uuid() -> str:
    """Generate a UUID string"""
    return str(uuid.uuid4())

def generate_hash(data: str) -> str:
    """Generate SHA-256 hash of data"""
    return hashlib.sha256(data.encode()).hexdigest()

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: Any = None) -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return json.dumps(default) if default is not None else "{}"

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove path separators and dangerous characters
    filename = re.sub(r"[/\\:*?"<>|]", "_", filename)
    # Remove any non-ASCII characters
    filename = re.sub(r"[^\x00-\x7F]", "_", filename)
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:250-len(ext)] + ext
    return filename

def get_file_mimetype(filename: str) -> str:
    """Get MIME type from filename"""
    mimetype, _ = mimetypes.guess_type(filename)
    return mimetype or "application/octet-stream"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"

def paginate_query(query, page: int = 1, page_size: int = 20):
    """Add pagination to SQLAlchemy query"""
    offset = (page - 1) * page_size
    return query.offset(offset).limit(page_size)

def calculate_pagination_info(total_count: int, page: int, page_size: int) -> Dict[str, Any]:
    """Calculate pagination metadata"""
    total_pages = (total_count + page_size - 1) // page_size
    has_next = page < total_pages
    has_prev = page > 1

    return {
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages,
        "has_next": has_next,
        "has_prev": has_prev,
        "next_page": page + 1 if has_next else None,
        "prev_page": page - 1 if has_prev else None
    }

def extract_coordinates(text: str) -> Optional[tuple]:
    """Extract latitude/longitude coordinates from text"""
    # Pattern for decimal degrees
    pattern = r"(-?\d+\.\d+),\s*(-?\d+\.\d+)"
    match = re.search(pattern, text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
    return None

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string"""
    if dt is None:
        return ""
    return dt.strftime(format_str)

def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """Parse datetime from string"""
    try:
        return datetime.strptime(dt_str, format_str)
    except (ValueError, TypeError):
        return None

def calculate_age_in_days(date: datetime) -> int:
    """Calculate age in days from a date"""
    if date is None:
        return 0
    return (datetime.utcnow() - date).days

def is_business_day(date: datetime) -> bool:
    """Check if date is a business day (Monday-Friday)"""
    return date.weekday() < 5

def next_business_day(date: datetime) -> datetime:
    """Get next business day"""
    next_day = date + timedelta(days=1)
    while not is_business_day(next_day):
        next_day += timedelta(days=1)
    return next_day

def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive data showing only first few characters"""
    if not data or len(data) <= visible_chars:
        return mask_char * len(data) if data else ""

    return data[:visible_chars] + mask_char * (len(data) - visible_chars)

def validate_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digit characters
    digits_only = re.sub(r"\D", "", phone)
    # Check if it has 10-15 digits
    return 10 <= len(digits_only) <= 15

def normalize_facility_id(facility_id: str) -> str:
    """Normalize facility ID to standard format"""
    if not facility_id:
        return ""

    # Remove spaces and convert to uppercase
    normalized = re.sub(r"\s+", "", facility_id.upper())
    # Remove any non-alphanumeric characters except hyphens
    normalized = re.sub(r"[^A-Z0-9-]", "", normalized)

    return normalized

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in kilometers using Haversine formula"""
    import math

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def retry_operation(func, max_attempts: int = 3, delay: float = 1.0):
    """Retry an operation with exponential backoff"""
    import time

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e

            wait_time = delay * (2 ** attempt)
            logger.warning(f"Operation failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s: {e}")
            time.sleep(wait_time)

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper
