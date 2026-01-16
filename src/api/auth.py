"""
Authentication and authorization for Claims Risk Classification API

This module provides API key validation, JWT token handling,
and user authentication for secure API access.
"""

import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import logging

# Import configuration
from .config import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer()
config = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationError(Exception):
    """Custom authentication error"""
    pass


class APIKeyManager:
    """
    Manages API keys for authentication
    
    Features:
    - API key validation
    - Key rotation and management
    - Usage tracking
    - Rate limiting per key
    """
    
    def __init__(self):
        self.valid_keys = set(config.api_keys)
        self.key_usage = {}  # Track API key usage
        self.key_metadata = {}  # Store metadata for each key
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        is_valid = api_key in self.valid_keys
        
        if is_valid:
            # Track usage
            if api_key not in self.key_usage:
                self.key_usage[api_key] = {
                    "request_count": 0,
                    "first_used": datetime.now(),
                    "last_used": datetime.now()
                }
            
            self.key_usage[api_key]["request_count"] += 1
            self.key_usage[api_key]["last_used"] = datetime.now()
            
            logger.info(f"Valid API key used (requests: {self.key_usage[api_key]['request_count']})")
        else:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        
        return is_valid
    
    def add_api_key(self, key: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new API key"""
        self.valid_keys.add(key)
        if metadata:
            self.key_metadata[key] = metadata
        logger.info(f"New API key added: {key[:8]}...")
    
    def remove_api_key(self, key: str):
        """Remove an API key"""
        self.valid_keys.discard(key)
        self.key_usage.pop(key, None)
        self.key_metadata.pop(key, None)
        logger.info(f"API key removed: {key[:8]}...")
    
    def generate_api_key(self) -> str:
        """Generate a new secure API key"""
        return f"rca_{secrets.token_urlsafe(32)}"
    
    def get_key_usage(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for an API key"""
        return self.key_usage.get(api_key)
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys with their metadata"""
        keys_info = []
        for key in self.valid_keys:
            key_info = {
                "key_prefix": f"{key[:8]}...",
                "usage": self.key_usage.get(key, {}),
                "metadata": self.key_metadata.get(key, {})
            }
            keys_info.append(key_info)
        return keys_info


class JWTManager:
    """
    JWT token management for session-based authentication
    
    Features:
    - Token generation and validation
    - Token refresh
    - User claims management
    """
    
    def __init__(self):
        self.secret_key = config.jwt_secret_key
        self.algorithm = config.jwt_algorithm
        self.expire_minutes = config.access_token_expire_minutes
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token
        
        Args:
            data: Data to encode in the token
            expires_delta: Custom expiration time
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
    
    def refresh_token(self, token: str) -> str:
        """
        Refresh an existing token
        
        Args:
            token: Token to refresh
            
        Returns:
            New JWT token
        """
        try:
            # Verify current token (allowing expired for refresh)
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            # Remove expiration claim
            payload.pop("exp", None)
            
            # Create new token
            return self.create_access_token(payload)
            
        except jwt.JWTError:
            raise AuthenticationError("Invalid token for refresh")


# Global instances
api_key_manager = APIKeyManager()
jwt_manager = JWTManager()


# Authentication dependency functions
async def verify_api_key(api_key: str) -> str:
    """
    FastAPI dependency to verify API key
    
    Args:
        api_key: API key from request header
        
    Returns:
        Valid API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return api_key


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        credentials: Authorization credentials from request
        
    Returns:
        User information from token or API key
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    # Try API key authentication first
    try:
        await verify_api_key(token)
        return {
            "auth_type": "api_key",
            "key_prefix": f"{token[:8]}...",
            "authenticated": True
        }
    except HTTPException:
        pass
    
    # Try JWT token authentication
    try:
        payload = jwt_manager.verify_token(token)
        return {
            "auth_type": "jwt",
            "user_id": payload.get("sub"),
            "permissions": payload.get("permissions", []),
            "authenticated": True
        }
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


# Permission-based authentication
def require_permissions(required_permissions: List[str]):
    """
    Decorator to require specific permissions for an endpoint
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        FastAPI dependency function
    """
    async def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        # API keys have all permissions by default
        if current_user["auth_type"] == "api_key":
            return current_user
        
        # Check JWT permissions
        user_permissions = current_user.get("permissions", [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission}"
                )
        
        return current_user
    
    return permission_checker


# Utility functions
def hash_password(password: str) -> str:
    """Hash a password for storage"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_api_key_for_user(user_id: str, permissions: List[str] = None) -> str:
    """
    Create an API key for a specific user with permissions
    
    Args:
        user_id: Unique user identifier
        permissions: List of permissions for this key
        
    Returns:
        Generated API key
    """
    api_key = api_key_manager.generate_api_key()
    
    metadata = {
        "user_id": user_id,
        "permissions": permissions or [],
        "created_at": datetime.now().isoformat()
    }
    
    api_key_manager.add_api_key(api_key, metadata)
    
    logger.info(f"API key created for user {user_id}")
    return api_key


# Rate limiting decorator
def rate_limit(requests_per_minute: int = 60):
    """
    Rate limiting decorator for API endpoints
    
    Args:
        requests_per_minute: Maximum requests per minute
        
    Returns:
        FastAPI dependency function
    """
    request_times = {}
    
    async def rate_limiter(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_key = current_user.get("user_id") or current_user.get("key_prefix", "unknown")
        current_time = datetime.now()
        
        # Initialize user tracking
        if user_key not in request_times:
            request_times[user_key] = []
        
        # Remove old requests (outside 1-minute window)
        cutoff_time = current_time - timedelta(minutes=1)
        request_times[user_key] = [
            req_time for req_time in request_times[user_key]
            if req_time > cutoff_time
        ]
        
        # Check rate limit
        if len(request_times[user_key]) >= requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {requests_per_minute} requests per minute",
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        request_times[user_key].append(current_time)
        
        return current_user
    
    return rate_limiter


# Example usage
if __name__ == "__main__":
    # Test API key generation
    test_key = api_key_manager.generate_api_key()
    print(f"Generated API key: {test_key}")
    
    # Test JWT token creation
    test_data = {"sub": "user123", "permissions": ["read", "write"]}
    test_token = jwt_manager.create_access_token(test_data)
    print(f"Generated JWT token: {test_token[:50]}...")
    
    # Test token verification
    try:
        decoded = jwt_manager.verify_token(test_token)
        print(f"Token verified: {decoded}")
    except AuthenticationError as e:
        print(f"Token verification failed: {e}")
