"""
Pydantic schemas for API request and response models

This module defines all the data models for API requests and responses
with comprehensive validation and documentation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd


class ClaimData(BaseModel):
    """Base claim data model"""
    
    # Structured claim fields
    claim_amount: float = Field(..., gt=0, description="Claim amount in dollars")
    claimant_age: int = Field(..., ge=0, le=120, description="Age of claimant")
    policy_duration_months: int = Field(..., ge=0, description="Policy duration in months")
    previous_claims: int = Field(..., ge=0, description="Number of previous claims")
    region: str = Field(..., description="Geographic region")
    claim_type: str = Field(..., description="Type of claim (Auto, Home, Health, Life)")
    day_of_week: str = Field(..., description="Day of week when claim was filed")
    weather_condition: Optional[str] = Field(None, description="Weather condition at time of incident")
    
    # Unstructured claim fields
    claim_description: str = Field(..., min_length=10, description="Detailed description of the claim")
    
    # Optional additional fields
    policy_premium: Optional[float] = Field(None, gt=0, description="Annual policy premium")
    deductible_amount: Optional[float] = Field(None, ge=0, description="Deductible amount")
    claim_adjuster_notes: Optional[str] = Field(None, description="Adjuster's notes")
    
    @validator('region')
    def validate_region(cls, v):
        valid_regions = ['North', 'South', 'East', 'West', 'Central']
        if v not in valid_regions:
            raise ValueError(f'Region must be one of {valid_regions}')
        return v
    
    @validator('claim_type')
    def validate_claim_type(cls, v):
        valid_types = ['Auto', 'Home', 'Health', 'Life', 'Commercial']
        if v not in valid_types:
            raise ValueError(f'Claim type must be one of {valid_types}')
        return v
    
    @validator('day_of_week')
    def validate_day_of_week(cls, v):
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                     'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        if v not in valid_days:
            raise ValueError(f'Day of week must be one of {valid_days}')
        return v
    
    @validator('claim_description')
    def validate_description(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Claim description must be at least 10 characters')
        return v.strip()


class ClaimsPredictionRequest(BaseModel):
    """Request model for single claim prediction"""
    
    claim_id: str = Field(..., description="Unique identifier for the claim")
    claim_data: ClaimData = Field(..., description="Claim data for prediction")
    
    # Optional parameters
    return_probabilities: bool = Field(True, description="Whether to return class probabilities")
    return_feature_importance: bool = Field(False, description="Whether to return feature importance")
    return_business_impact: bool = Field(True, description="Whether to return business impact analysis")
    
    @validator('claim_id')
    def validate_claim_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Claim ID cannot be empty')
        return v.strip()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert request to pandas DataFrame for model prediction"""
        data_dict = self.claim_data.dict()
        return pd.DataFrame([data_dict])


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    claims: List[ClaimsPredictionRequest] = Field(..., description="List of claims to process")
    
    # Batch processing options
    fail_on_error: bool = Field(False, description="Whether to fail entire batch if one claim fails")
    return_aggregated_stats: bool = Field(True, description="Whether to return batch statistics")
    
    @validator('claims')
    def validate_claims_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Claims list cannot be empty')
        if len(v) > 100:  # Configurable batch size limit
            raise ValueError('Batch size cannot exceed 100 claims')
        return v
    
    @validator('batch_id')
    def validate_batch_id(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Batch ID cannot be empty string')
        return v.strip() if v else None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert batch request to pandas DataFrame"""
        data_list = []
        for claim_request in self.claims:
            claim_dict = claim_request.claim_data.dict()
            claim_dict['claim_id'] = claim_request.claim_id
            data_list.append(claim_dict)
        
        return pd.DataFrame(data_list)


class BusinessImpact(BaseModel):
    """Business impact analysis for a prediction"""
    
    estimated_cost: float = Field(..., description="Estimated total cost of claim")
    investigation_cost: float = Field(..., description="Cost of investigation if flagged as high-risk")
    potential_savings: float = Field(..., description="Potential savings compared to baseline")
    confidence_multiplier: float = Field(..., description="Confidence-based cost adjustment")
    recommendation: str = Field(..., description="Recommended action based on prediction")


class ClaimsPredictionResponse(BaseModel):
    """Response model for single claim prediction"""
    
    # Basic prediction results
    claim_id: str = Field(..., description="Original claim identifier")
    predicted_risk: str = Field(..., description="Predicted risk level (high/low)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    
    # Probability scores
    probability_high_risk: float = Field(..., ge=0, le=1, description="Probability of high risk")
    probability_low_risk: float = Field(..., ge=0, le=1, description="Probability of low risk")
    
    # Model information
    model_used: str = Field(..., description="Name of model used for prediction")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Optional detailed results
    feature_importance: Optional[List[Dict[str, Any]]] = Field(None, description="Feature importance scores")
    business_impact: Optional[BusinessImpact] = Field(None, description="Business impact analysis")
    
    # Validation and quality indicators
    input_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality score of input data")
    prediction_reliability: Optional[str] = Field(None, description="Reliability assessment")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    batch_id: str = Field(..., description="Batch identifier")
    predictions: List[ClaimsPredictionResponse] = Field(..., description="Individual predictions")
    
    # Batch-level statistics
    batch_statistics: Dict[str, Any] = Field(..., description="Aggregated batch statistics")
    processing_summary: Dict[str, Any] = Field(..., description="Processing performance summary")
    
    # Error handling
    failed_predictions: Optional[List[Dict[str, Any]]] = Field(None, description="Claims that failed to process")
    success_rate: Optional[float] = Field(None, ge=0, le=1, description="Percentage of successful predictions")


class ModelStatusResponse(BaseModel):
    """Response model for model status information"""
    
    model_name: str = Field(..., description="Name of the model")
    status: str = Field(..., description="Current status (loaded, not_loaded, error)")
    version: Optional[str] = Field(None, description="Model version")
    last_used: Optional[str] = Field(None, description="ISO timestamp of last use")
    prediction_count: int = Field(..., description="Total number of predictions made")
    
    # Performance metrics (if available)
    recent_performance: Optional[Dict[str, float]] = Field(None, description="Recent performance metrics")
    load_time_seconds: Optional[float] = Field(None, description="Time taken to load model")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    
    # Component status
    available_models: Optional[List[str]] = Field(None, description="List of available models")
    dependencies: Optional[Dict[str, str]] = Field(None, description="Status of dependencies")
    
    # System metrics
    system_metrics: Optional[Dict[str, Any]] = Field(None, description="System performance metrics")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    
    error_code: int = Field(..., description="HTTP error code")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    path: str = Field(..., description="Request path that caused error")
    
    # Optional detailed error information
    details: Optional[str] = Field(None, description="Detailed error information")
    validation_errors: Optional[List[Dict[str, Any]]] = Field(None, description="Validation error details")
    trace_id: Optional[str] = Field(None, description="Request trace identifier")


class DriftStatusResponse(BaseModel):
    """Response model for data drift status"""
    
    model_name: str = Field(..., description="Model name")
    drift_detected: bool = Field(..., description="Whether drift is detected")
    drift_score: float = Field(..., description="Overall drift score")
    severity: str = Field(..., description="Drift severity level")
    last_check: str = Field(..., description="Last drift check timestamp")
    
    # Detailed drift information
    feature_drifts: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Per-feature drift details")
    drift_history: Optional[List[Dict[str, Any]]] = Field(None, description="Historical drift data")


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics"""
    
    model_name: str = Field(..., description="Model name")
    period_days: int = Field(..., description="Metrics calculation period in days")
    
    # Performance metrics
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    trend: str = Field(..., description="Performance trend (improving/stable/declining)")
    last_updated: str = Field(..., description="Last metrics update timestamp")
    
    # Detailed performance data
    daily_metrics: Optional[List[Dict[str, Any]]] = Field(None, description="Daily performance breakdown")
    benchmark_comparison: Optional[Dict[str, float]] = Field(None, description="Comparison with baseline")


class ValidationRequest(BaseModel):
    """Request model for data validation"""
    
    data: Dict[str, Any] = Field(..., description="Data to validate")
    validation_rules: Optional[List[str]] = Field(None, description="Specific validation rules to apply")
    strict_mode: bool = Field(False, description="Whether to use strict validation")


class ValidationResponse(BaseModel):
    """Response model for data validation"""
    
    is_valid: bool = Field(..., description="Whether data is valid")
    validation_errors: List[Dict[str, Any]] = Field(..., description="List of validation errors")
    warnings: Optional[List[str]] = Field(None, description="Validation warnings")
    quality_score: float = Field(..., ge=0, le=1, description="Overall data quality score")


# Request/Response model mappings for different endpoints
class APISchemas:
    """Container for all API schemas"""
    
    # Request models
    ClaimsPredictionRequest = ClaimsPredictionRequest
    BatchPredictionRequest = BatchPredictionRequest
    ValidationRequest = ValidationRequest
    
    # Response models  
    ClaimsPredictionResponse = ClaimsPredictionResponse
    BatchPredictionResponse = BatchPredictionResponse
    ModelStatusResponse = ModelStatusResponse
    HealthCheckResponse = HealthCheckResponse
    ErrorResponse = ErrorResponse
    DriftStatusResponse = DriftStatusResponse
    PerformanceMetricsResponse = PerformanceMetricsResponse
    ValidationResponse = ValidationResponse
    
    # Data models
    ClaimData = ClaimData
    BusinessImpact = BusinessImpact


# Example usage and validation
if __name__ == "__main__":
    # Example claim data for testing
    sample_claim = ClaimData(
        claim_amount=15000.0,
        claimant_age=35,
        policy_duration_months=24,
        previous_claims=1,
        region="North",
        claim_type="Auto",
        day_of_week="Monday",
        weather_condition="Clear",
        claim_description="Vehicle collision on interstate highway resulting in front-end damage",
        policy_premium=1200.0,
        deductible_amount=500.0
    )
    
    # Example prediction request
    prediction_request = ClaimsPredictionRequest(
        claim_id="CLAIM_2026_001",
        claim_data=sample_claim,
        return_probabilities=True,
        return_feature_importance=True,
        return_business_impact=True
    )
    
    print("Sample prediction request:")
    print(prediction_request.json(indent=2))
    
    # Convert to DataFrame for testing
    df = prediction_request.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")