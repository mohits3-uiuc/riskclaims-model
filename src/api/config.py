"""
API Configuration Management

This module handles all API configuration settings including security,
performance parameters, and environment-specific settings.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, validator
import secrets


class APIConfig(BaseSettings):
    """
    API Configuration using Pydantic BaseSettings for environment variable management
    """
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug_mode: bool = False
    
    # API Security
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = []
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS and Security
    allowed_origins: List[str] = ["*"]
    allowed_hosts: List[str] = ["*"]
    enable_https: bool = False
    
    # Model Configuration
    default_model: str = "Random Forest"
    available_models: List[str] = ["Random Forest", "XGBoost", "Neural Network"]
    model_cache_size: int = 3
    model_timeout_seconds: int = 30
    
    # Prediction Limits
    max_batch_size: int = 100
    request_timeout_seconds: int = 60
    max_requests_per_minute: int = 1000
    
    # Monitoring Configuration
    enable_monitoring: bool = True
    monitoring_interval_minutes: int = 15
    drift_check_sample_size: int = 1000
    performance_history_days: int = 30
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "api.log"
    log_rotation_size: str = "10MB"
    log_retention_days: int = 30
    enable_request_logging: bool = True
    
    # Database Configuration (for monitoring and logging)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    enable_caching: bool = False
    cache_ttl_seconds: int = 300
    
    # Performance Configuration
    max_request_size_mb: int = 10
    response_timeout_seconds: int = 30
    enable_compression: bool = True
    
    # Feature Flags
    enable_feature_importance: bool = True
    enable_business_impact: bool = True
    enable_batch_processing: bool = True
    enable_async_processing: bool = False
    
    # Model-specific Configuration
    model_config: Dict[str, Any] = {
        "Random Forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        },
        "XGBoost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1
        },
        "Neural Network": {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.3,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10
        }
    }
    
    # Monitoring Configuration
    monitoring_config: Dict[str, Any] = {
        "drift_config": {
            "ks_test_threshold": 0.05,
            "psi_threshold": 0.2,
            "warning_threshold": 0.1,
            "critical_threshold": 0.3,
            "enable_alerts": True
        },
        "performance_config": {
            "degradation_threshold": 0.05,
            "critical_threshold": 0.10,
            "retrain_threshold": 0.08,
            "enable_alerts": True
        }
    }
    
    # Business Rules Configuration
    business_config: Dict[str, Any] = {
        "cost_thresholds": {
            "low_risk_claim_cost": 5000,
            "high_risk_claim_cost": 25000,
            "investigation_cost": 500
        },
        "confidence_thresholds": {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        },
        "risk_factors": {
            "age_risk_min": 25,
            "age_risk_max": 65,
            "amount_risk_threshold": 20000,
            "previous_claims_risk_threshold": 2
        }
    }
    
    # Development/Testing Configuration
    test_mode: bool = False
    mock_model_predictions: bool = False
    enable_detailed_errors: bool = False
    
    @validator('api_keys', pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string from environment variable
            return [key.strip() for key in v.split(',') if key.strip()]
        return v or []
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v or ["*"]
    
    @validator('allowed_hosts', pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',') if host.strip()]
        return v or ["*"]
    
    @validator('available_models', pre=True)
    def parse_available_models(cls, v):
        if isinstance(v, str):
            return [model.strip() for model in v.split(',') if model.strip()]
        return v or ["Random Forest", "XGBoost", "Neural Network"]
    
    @validator('port')
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('workers')
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError('Workers must be at least 1')
        return v
    
    @validator('max_batch_size')
    def validate_max_batch_size(cls, v):
        if v < 1 or v > 10000:
            raise ValueError('Max batch size must be between 1 and 10000')
        return v
    
    @validator('default_model')
    def validate_default_model(cls, v, values):
        available_models = values.get('available_models', [])
        if available_models and v not in available_models:
            raise ValueError(f'Default model must be one of {available_models}')
        return v
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
        # Environment variable mapping
        fields = {
            'api_keys': {'env': 'API_KEYS'},
            'jwt_secret_key': {'env': 'JWT_SECRET_KEY'},
            'database_url': {'env': 'DATABASE_URL'},
            'redis_url': {'env': 'REDIS_URL'},
            'allowed_origins': {'env': 'ALLOWED_ORIGINS'},
            'allowed_hosts': {'env': 'ALLOWED_HOSTS'},
            'available_models': {'env': 'AVAILABLE_MODELS'}
        }


class DevelopmentConfig(APIConfig):
    """Development environment configuration"""
    debug_mode: bool = True
    log_level: str = "DEBUG"
    enable_detailed_errors: bool = True
    workers: int = 1
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]


class ProductionConfig(APIConfig):
    """Production environment configuration"""
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_detailed_errors: bool = False
    workers: int = 4
    enable_https: bool = True
    max_requests_per_minute: int = 10000
    enable_monitoring: bool = True
    
    @validator('api_keys')
    def validate_production_api_keys(cls, v):
        if not v:
            raise ValueError('API keys are required in production')
        return v


class TestingConfig(APIConfig):
    """Testing environment configuration"""
    test_mode: bool = True
    debug_mode: bool = True
    mock_model_predictions: bool = True
    log_level: str = "DEBUG"
    database_url: str = "sqlite:///test.db"
    enable_monitoring: bool = False
    
    # Reduced limits for testing
    max_batch_size: int = 10
    request_timeout_seconds: int = 10


def get_config(environment: Optional[str] = None) -> APIConfig:
    """
    Get configuration based on environment
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        APIConfig instance for the specified environment
    """
    environment = environment or os.getenv('ENVIRONMENT', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'dev': DevelopmentConfig,
        'production': ProductionConfig,
        'prod': ProductionConfig,
        'testing': TestingConfig,
        'test': TestingConfig
    }
    
    config_class = config_map.get(environment, DevelopmentConfig)
    return config_class()


def validate_config(config: APIConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors
    
    Args:
        config: Configuration instance to validate
        
    Returns:
        List of validation messages
    """
    warnings = []
    
    # Security validations
    if not config.api_keys and not config.test_mode:
        warnings.append("No API keys configured - API will be unprotected")
    
    if config.debug_mode and not config.test_mode:
        warnings.append("Debug mode enabled - not recommended for production")
    
    if "*" in config.allowed_origins and not config.debug_mode:
        warnings.append("CORS allows all origins - potential security risk")
    
    # Performance validations
    if config.workers > 8:
        warnings.append("High number of workers may cause resource contention")
    
    if config.max_batch_size > 1000:
        warnings.append("Large batch size may cause memory issues")
    
    # Feature validations
    if config.enable_monitoring and not config.database_url:
        warnings.append("Monitoring enabled but no database URL configured")
    
    if config.enable_caching and not config.redis_url:
        warnings.append("Caching enabled but no Redis URL configured")
    
    return warnings


# Global configuration instance
_config_instance = None

def get_settings() -> APIConfig:
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_config()
        
        # Validate configuration and log warnings
        warnings = validate_config(_config_instance)
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    return _config_instance


# Example usage and testing
if __name__ == "__main__":
    # Test different environment configurations
    dev_config = get_config('development')
    prod_config = get_config('production')  
    test_config = get_config('testing')
    
    print("Development Config:")
    print(f"  Debug Mode: {dev_config.debug_mode}")
    print(f"  Workers: {dev_config.workers}")
    print(f"  Log Level: {dev_config.log_level}")
    
    print("\nProduction Config:")
    print(f"  Debug Mode: {prod_config.debug_mode}")
    print(f"  Workers: {prod_config.workers}")
    print(f"  HTTPS Enabled: {prod_config.enable_https}")
    
    print("\nTesting Config:")
    print(f"  Test Mode: {test_config.test_mode}")
    print(f"  Mock Predictions: {test_config.mock_model_predictions}")
    print(f"  Max Batch Size: {test_config.max_batch_size}")
    
    # Validate configurations
    print("\nConfiguration Warnings:")
    for config_name, config in [("dev", dev_config), ("prod", prod_config), ("test", test_config)]:
        warnings = validate_config(config)
        if warnings:
            print(f"{config_name}: {warnings}")
        else:
            print(f"{config_name}: No warnings")