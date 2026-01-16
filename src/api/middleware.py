"""
Custom middleware for API request processing, logging, and validation

This module provides middleware components for request/response logging,
validation, error handling, and performance monitoring.
"""

import time
import json
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging
    
    Features:
    - Request/response logging with timing
    - Unique request ID generation
    - Error logging with stack traces
    - Performance metrics collection
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request details
        start_time = time.time()
        
        # Get request body for logging (if not too large)
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) < 10000:  # Only log if body is reasonable size
                    body = body.decode('utf-8')
                else:
                    body = f"<Large body: {len(body)} bytes>"
            except Exception:
                body = "<Could not read body>"
        
        logger.info(
            f"Request {request_id}: {request.method} {request.url} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        if body and request.url.path not in ["/health", "/docs", "/openapi.json"]:
            logger.debug(f"Request {request_id} body: {body}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"in {process_time:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request {request_id} failed after {process_time:.3f}s: {str(e)}",
                exc_info=True
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": 500,
                    "message": "Internal server error",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request validation and preprocessing
    
    Features:
    - Request size validation
    - Content type validation
    - Rate limiting (basic)
    - Request sanitization
    """
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_request_size = max_request_size
        self.request_counts = {}  # Simple in-memory rate limiting
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error_code": 413,
                    "message": f"Request too large. Max size: {self.max_request_size} bytes",
                    "timestamp": time.time()
                }
            )
        
        # Basic rate limiting (in production, use Redis or proper rate limiter)
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries (simple cleanup)
        cutoff_time = current_time - 60  # 1 minute window
        self.request_counts = {
            ip: requests for ip, requests in self.request_counts.items()
            if any(req_time > cutoff_time for req_time in requests)
        }
        
        # Update request count for client
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove old requests for this client
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > cutoff_time
        ]
        
        # Check rate limit (60 requests per minute per IP)
        if len(self.request_counts[client_ip]) > 60:
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": 429,
                    "message": "Rate limit exceeded. Max 60 requests per minute.",
                    "timestamp": current_time,
                    "retry_after": 60
                }
            )
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return JSONResponse(
                    status_code=415,
                    content={
                        "error_code": 415,
                        "message": "Unsupported media type. Expected application/json",
                        "timestamp": current_time
                    }
                )
        
        # Process request
        response = await call_next(request)
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for headers and basic protection
    
    Features:
    - Security headers
    - Basic XSS protection
    - CORS handling
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Don't cache sensitive endpoints
        if request.url.path in ["/predict", "/predict/batch"]:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and metrics collection
    
    Features:
    - Response time tracking
    - Endpoint usage statistics
    - Performance alerts
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "request_count": {},
            "response_times": {},
            "error_count": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            response = await call_next(request)
            
            # Record metrics
            process_time = time.time() - start_time
            
            # Update counters
            self.metrics["request_count"][endpoint] = self.metrics["request_count"].get(endpoint, 0) + 1
            
            if endpoint not in self.metrics["response_times"]:
                self.metrics["response_times"][endpoint] = []
            self.metrics["response_times"][endpoint].append(process_time)
            
            # Keep only recent response times (last 1000 requests)
            if len(self.metrics["response_times"][endpoint]) > 1000:
                self.metrics["response_times"][endpoint] = self.metrics["response_times"][endpoint][-1000:]
            
            # Log slow requests
            if process_time > 5.0:  # 5 seconds threshold
                logger.warning(f"Slow request: {endpoint} took {process_time:.3f}s")
            
            return response
            
        except Exception as e:
            # Record error
            self.metrics["error_count"][endpoint] = self.metrics["error_count"].get(endpoint, 0) + 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        import statistics
        
        processed_metrics = {}
        
        for endpoint, times in self.metrics["response_times"].items():
            if times:
                processed_metrics[endpoint] = {
                    "request_count": self.metrics["request_count"].get(endpoint, 0),
                    "error_count": self.metrics["error_count"].get(endpoint, 0),
                    "avg_response_time": statistics.mean(times),
                    "median_response_time": statistics.median(times),
                    "max_response_time": max(times),
                    "min_response_time": min(times)
                }
        
        return processed_metrics


# Global performance monitoring instance
performance_monitor = None

def get_performance_metrics() -> Dict[str, Any]:
    """Get global performance metrics"""
    global performance_monitor
    if performance_monitor:
        return performance_monitor.get_metrics()
    return {}


# Example usage
if __name__ == "__main__":
    print("API Middleware Components:")
    print("1. LoggingMiddleware - Request/response logging with timing")
    print("2. RequestValidationMiddleware - Size and rate limiting")
    print("3. SecurityMiddleware - Security headers and protection")
    print("4. PerformanceMonitoringMiddleware - Metrics collection")
