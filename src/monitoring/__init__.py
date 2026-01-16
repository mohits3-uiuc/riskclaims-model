"""
Monitoring Package for Claims Risk Classification Pipeline

This package provides comprehensive monitoring capabilities including data drift detection,
model performance monitoring, and integrated alerting for production ML models.
"""

from .drift_detector import DataDriftDetector, ModelPerformanceMonitor, IntegratedMonitoringSystem

__all__ = [
    'DataDriftDetector',
    'ModelPerformanceMonitor', 
    'IntegratedMonitoringSystem'
]

__version__ = "1.0.0"