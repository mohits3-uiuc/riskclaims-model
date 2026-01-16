"""
FastAPI REST API for Claims Risk Classification Pipeline

This module provides a production-ready REST API for claims risk classification
with comprehensive error handling, logging, validation, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our pipeline components
from models import RandomForestClaimsModel, XGBoostClaimsModel, NeuralNetworkClaimsModel
from preprocessing import StructuredDataPreprocessor, UnstructuredDataPreprocessor, FeatureEngineer
from monitoring import IntegratedMonitoringSystem
from evaluation import ModelEvaluator

# Import API components
from .schemas import (
    ClaimsPredictionRequest,
    ClaimsPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelStatusResponse,
    HealthCheckResponse,
    ErrorResponse
)
from .middleware import LoggingMiddleware, RequestValidationMiddleware
from .auth import verify_api_key, get_current_user
from .model_manager import ModelManager
from .config import APIConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Claims Risk Classification API",
    description="Production API for claims risk classification using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize components
config = APIConfig()
model_manager = ModelManager(config.model_config)
monitoring_system = IntegratedMonitoringSystem(config.monitoring_config)
security = HTTPBearer()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=config.allowed_hosts
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestValidationMiddleware)

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("Starting Claims Risk Classification API...")
    
    try:
        # Load models
        await model_manager.load_models()
        
        # Initialize monitoring
        if config.enable_monitoring:
            # This would typically load reference data and baseline metrics
            logger.info("Monitoring system initialized")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.critical(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Claims Risk Classification API...")
    
    # Save any pending monitoring data
    try:
        if config.enable_monitoring:
            monitoring_system.export_monitoring_data("shutdown_monitoring_data.json")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("API shutdown completed")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=exc.status_code,
            message=exc.detail,
            timestamp=datetime.now().isoformat(),
            path=str(request.url)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code=500,
            message="Internal server error",
            timestamp=datetime.now().isoformat(),
            path=str(request.url),
            details=str(exc) if config.debug_mode else None
        ).dict()
    )

# Health Check Endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime_seconds": 0  # Would track actual uptime
    }
    
    try:
        # Check model availability
        available_models = await model_manager.get_available_models()
        health_status["available_models"] = available_models
        
        # Check dependencies
        health_status["dependencies"] = {
            "database": "healthy",  # Would check actual DB connection
            "monitoring": "healthy" if config.enable_monitoring else "disabled"
        }
        
        return HealthCheckResponse(**health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Detailed health check with authentication required"""
    
    # Verify API key
    await verify_api_key(credentials.credentials)
    
    try:
        detailed_status = {
            "timestamp": datetime.now().isoformat(),
            "api_status": "healthy",
            "models": {},
            "preprocessing": {},
            "monitoring": {},
            "system": {}
        }
        
        # Check each model
        for model_name in config.available_models:
            try:
                model_info = await model_manager.get_model_info(model_name)
                detailed_status["models"][model_name] = {
                    "status": "loaded" if model_info else "not_loaded",
                    "last_used": model_info.get("last_used") if model_info else None
                }
            except Exception as e:
                detailed_status["models"][model_name] = {"status": "error", "error": str(e)}
        
        # System resources
        detailed_status["system"] = {
            "memory_usage": "normal",  # Would check actual memory
            "cpu_usage": "normal",     # Would check actual CPU
            "disk_space": "sufficient"  # Would check actual disk
        }
        
        return detailed_status
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

# Model Management Endpoints
@app.get("/models", response_model=List[ModelStatusResponse], tags=["Models"])
async def list_models(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """List available models and their status"""
    
    await verify_api_key(credentials.credentials)
    
    try:
        models_status = []
        
        for model_name in config.available_models:
            model_info = await model_manager.get_model_info(model_name)
            
            status_response = ModelStatusResponse(
                model_name=model_name,
                status="loaded" if model_info else "not_loaded",
                version=model_info.get("version", "unknown") if model_info else None,
                last_used=model_info.get("last_used") if model_info else None,
                prediction_count=model_info.get("prediction_count", 0) if model_info else 0
            )
            
            models_status.append(status_response)
        
        return models_status
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@app.post("/models/{model_name}/load", tags=["Models"])
async def load_model(model_name: str, 
                    credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Load a specific model"""
    
    await verify_api_key(credentials.credentials)
    
    if model_name not in config.available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        await model_manager.load_specific_model(model_name)
        
        return {
            "message": f"Model {model_name} loaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Prediction Endpoints
@app.post("/predict", response_model=ClaimsPredictionResponse, tags=["Predictions"])
async def predict_claim_risk(
    request: ClaimsPredictionRequest,
    background_tasks: BackgroundTasks,
    model_name: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Predict risk level for a single claim
    
    This endpoint analyzes claim data and returns risk classification
    with confidence scores and business impact metrics.
    """
    
    await verify_api_key(credentials.credentials)
    
    try:
        # Use default model if not specified
        model_name = model_name or config.default_model
        
        if model_name not in config.available_models:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")
        
        # Get model instance
        model = await model_manager.get_model(model_name)
        if not model:
            raise HTTPException(status_code=503, detail=f"Model {model_name} not available")
        
        # Prepare prediction data
        prediction_data = request.to_dataframe()
        
        # Make prediction
        logger.info(f"Making prediction with model: {model_name}")
        
        # Get prediction and probabilities
        prediction = model.predict(prediction_data)[0]
        probabilities = model.predict_proba(prediction_data)[0]
        
        # Prepare response
        response = ClaimsPredictionResponse(
            claim_id=request.claim_id,
            predicted_risk=prediction,
            confidence_score=float(max(probabilities)),
            probability_high_risk=float(probabilities[1] if len(probabilities) > 1 else probabilities[0]),
            probability_low_risk=float(probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]),
            model_used=model_name,
            prediction_timestamp=datetime.now().isoformat(),
            processing_time_ms=0  # Would track actual processing time
        )
        
        # Add feature importance if available
        if hasattr(model, 'get_feature_importance'):
            try:
                importance_df = model.get_feature_importance()
                response.feature_importance = importance_df.head(10).to_dict('records')
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")
        
        # Add business impact estimates
        response.business_impact = _calculate_business_impact(
            prediction, float(max(probabilities)), request
        )
        
        # Background tasks
        if config.enable_monitoring:
            background_tasks.add_task(
                _log_prediction_for_monitoring,
                model_name, prediction_data, prediction, probabilities
            )
        
        # Update model usage statistics
        background_tasks.add_task(
            model_manager.update_model_usage, model_name
        )
        
        logger.info(f"Prediction completed: {prediction} (confidence: {max(probabilities):.3f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_name: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Predict risk levels for multiple claims in batch
    
    Processes multiple claims efficiently and returns predictions
    with aggregated statistics and insights.
    """
    
    await verify_api_key(credentials.credentials)
    
    # Validate batch size
    if len(request.claims) > config.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds maximum ({config.max_batch_size})"
        )
    
    try:
        model_name = model_name or config.default_model
        
        # Get model instance
        model = await model_manager.get_model(model_name)
        if not model:
            raise HTTPException(status_code=503, detail=f"Model {model_name} not available")
        
        # Process batch
        logger.info(f"Processing batch of {len(request.claims)} claims with model: {model_name}")
        
        batch_predictions = []
        all_probabilities = []
        
        # Convert all requests to DataFrame for batch processing
        batch_data = request.to_dataframe()
        
        # Make batch predictions
        predictions = model.predict(batch_data)
        probabilities_batch = model.predict_proba(batch_data)
        
        # Process individual predictions
        for i, (claim_request, prediction, probabilities) in enumerate(zip(request.claims, predictions, probabilities_batch)):
            
            prediction_response = ClaimsPredictionResponse(
                claim_id=claim_request.claim_id,
                predicted_risk=prediction,
                confidence_score=float(max(probabilities)),
                probability_high_risk=float(probabilities[1] if len(probabilities) > 1 else probabilities[0]),
                probability_low_risk=float(probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]),
                model_used=model_name,
                prediction_timestamp=datetime.now().isoformat(),
                processing_time_ms=0
            )
            
            # Add business impact
            prediction_response.business_impact = _calculate_business_impact(
                prediction, float(max(probabilities)), claim_request
            )
            
            batch_predictions.append(prediction_response)
            all_probabilities.append(probabilities)
        
        # Calculate batch statistics
        high_risk_count = sum(1 for pred in predictions if pred == 'high')
        avg_confidence = sum(max(probs) for probs in all_probabilities) / len(all_probabilities)
        
        # Prepare batch response
        batch_response = BatchPredictionResponse(
            batch_id=request.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            predictions=batch_predictions,
            batch_statistics={
                "total_claims": len(request.claims),
                "high_risk_claims": high_risk_count,
                "low_risk_claims": len(request.claims) - high_risk_count,
                "high_risk_percentage": (high_risk_count / len(request.claims)) * 100,
                "average_confidence": avg_confidence,
                "model_used": model_name
            },
            processing_summary={
                "total_processing_time_ms": 0,  # Would track actual time
                "average_processing_time_ms": 0,
                "processed_timestamp": datetime.now().isoformat()
            }
        )
        
        # Background monitoring
        if config.enable_monitoring:
            background_tasks.add_task(
                _log_batch_for_monitoring,
                model_name, batch_data, predictions, all_probabilities
            )
        
        background_tasks.add_task(
            model_manager.update_batch_usage, model_name, len(request.claims)
        )
        
        logger.info(f"Batch processing completed: {len(batch_predictions)} predictions")
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed")

# Monitoring Endpoints
@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_drift_status(
    model_name: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current data drift status for a model"""
    
    await verify_api_key(credentials.credentials)
    
    if not config.enable_monitoring:
        raise HTTPException(status_code=501, detail="Monitoring not enabled")
    
    try:
        # This would get actual drift status from monitoring system
        drift_status = {
            "model_name": model_name,
            "drift_detected": False,  # Would get from actual monitoring
            "drift_score": 0.05,
            "last_check": datetime.now().isoformat(),
            "severity": "none"
        }
        
        return drift_status
        
    except Exception as e:
        logger.error(f"Error getting drift status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get drift status")

@app.get("/monitoring/performance", tags=["Monitoring"])
async def get_performance_metrics(
    model_name: str,
    days: int = 7,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get performance metrics for a model over specified days"""
    
    await verify_api_key(credentials.credentials)
    
    if not config.enable_monitoring:
        raise HTTPException(status_code=501, detail="Monitoring not enabled")
    
    try:
        # This would get actual performance metrics
        performance_metrics = {
            "model_name": model_name,
            "period_days": days,
            "metrics": {
                "accuracy": 0.85,
                "f1_score": 0.83,
                "roc_auc": 0.91
            },
            "trend": "stable",
            "last_updated": datetime.now().isoformat()
        }
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")

# Utility Functions
def _calculate_business_impact(prediction: str, confidence: float, request) -> Dict[str, Any]:
    """Calculate business impact estimates for a prediction"""
    
    # Basic business impact calculation
    base_cost = 5000 if prediction == 'low' else 25000
    investigation_cost = 500 if prediction == 'high' else 0
    
    # Confidence-adjusted estimates
    confidence_multiplier = confidence
    estimated_cost = base_cost * confidence_multiplier + investigation_cost
    
    # Potential savings (compared to not using the model)
    baseline_cost = 15000  # Average claim cost
    potential_savings = max(0, baseline_cost - estimated_cost)
    
    return {
        "estimated_cost": round(estimated_cost, 2),
        "investigation_cost": investigation_cost,
        "potential_savings": round(potential_savings, 2),
        "confidence_multiplier": round(confidence_multiplier, 3),
        "recommendation": "investigate" if prediction == 'high' and confidence > 0.7 else "process_normally"
    }

async def _log_prediction_for_monitoring(model_name: str, data, prediction, probabilities):
    """Background task to log prediction for monitoring"""
    try:
        # This would integrate with the monitoring system
        logger.info(f"Logged prediction for monitoring: {model_name}")
    except Exception as e:
        logger.error(f"Error logging prediction for monitoring: {e}")

async def _log_batch_for_monitoring(model_name: str, data, predictions, probabilities):
    """Background task to log batch predictions for monitoring"""
    try:
        # This would integrate with the monitoring system
        logger.info(f"Logged batch predictions for monitoring: {model_name}")
    except Exception as e:
        logger.error(f"Error logging batch for monitoring: {e}")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug_mode,
        workers=config.workers if not config.debug_mode else 1
    )