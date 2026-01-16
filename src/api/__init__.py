"""
FastAPI REST API for Claims Risk Classification Pipeline

This package provides a production-ready REST API for claims risk classification
with comprehensive features including authentication, monitoring, and validation.

Main Components:
- main: FastAPI application with all endpoints
- schemas: Pydantic models for request/response validation
- config: Configuration management for different environments
- auth: Authentication and authorization system
- middleware: Custom middleware for logging, validation, and security
- model_manager: ML model loading and management

Features:
- Single and batch prediction endpoints
- Model management and health monitoring
- Data drift and performance monitoring integration
- Comprehensive error handling and logging
- API key and JWT authentication
- Rate limiting and security middleware
- Business impact analysis
- Feature importance and model explainability
"""

from .main import app
from .config import get_config, get_settings
from .schemas import APISchemas
from .auth import api_key_manager, jwt_manager
from .model_manager import ModelManager

__all__ = [
    'app',
    'get_config', 
    'get_settings',
    'APISchemas',
    'api_key_manager',
    'jwt_manager', 
    'ModelManager'
]

__version__ = "1.0.0"
__author__ = "Claims Risk Classification Team"
__description__ = "Production API for ML-powered claims risk classification"
