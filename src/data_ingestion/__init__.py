"""
Data Ingestion Package for Claims Risk Classification Pipeline

This package provides comprehensive data ingestion capabilities for collecting
claims data from various sources including databases, S3, and streaming services.
"""

from .database_connector import DatabaseConnector
from .s3_connector import S3DataConnector
from .data_loader import DataLoader
from .stream_processor import StreamProcessor

__all__ = [
    'DatabaseConnector',
    'S3DataConnector', 
    'DataLoader',
    'StreamProcessor'
]

__version__ = "1.0.0"