"""
Security utilities for Universal Chatbot Framework

This module provides security-related utilities for all channel implementations.
"""

import hashlib
import hmac
import json
import logging
import re
import time
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import uuid
import secrets

import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography.fernet import Fernet
import asyncio
from functools import wraps

try:
    import redis  # type: ignore
except ImportError:
    redis = None

# Configure logging
logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.jwt_secret = config.get("jwt_secret", secrets.token_urlsafe(32))
        self.jwt_algorithm = config.get("jwt_algorithm", "HS256")
        self.jwt_expiry_hours = config.get("jwt_expiry_hours", 24)
        
        self.rate_limit_per_minute = config.get("rate_limit_per_minute", 60)
        self.rate_limit_per_hour = config.get("rate_limit_per_hour", 1000)
        
        self.redis_url = config.get("redis_url", "redis://localhost:6379")
        self.encryption_key = config.get("encryption_key")
        
        self.webhook_signature_header = config.get("webhook_signature_header", "X-Signature")
        self.webhook_secret = config.get("webhook_secret")
        
        self.enforce_https = config.get("enforce_https", True)
        self.allowed_origins = config.get("allowed_origins", [])


class CorrelationContext:
    """Context manager for correlation IDs."""
    
    _current_id: Optional[str] = None
    
    @classmethod
    def get_id(cls) -> str:
        """Get current correlation ID."""
        return cls._current_id or str(uuid.uuid4())
    
    @classmethod
    def set_id(cls, correlation_id: str):
        """Set correlation ID."""
        cls._current_id = correlation_id
    
    @classmethod
    def generate_new(cls) -> str:
        """Generate and set new correlation ID."""
        new_id = str(uuid.uuid4())
        cls.set_id(new_id)
        return new_id


class SecureLogger:
    """Secure logger with PII masking and correlation IDs."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.pii_patterns = [
            (r'\b\d{10,15}\b', '***PHONE***'),  # Phone numbers
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***'),  # Emails
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '***CARD***'),  # Credit cards
            (r'\b\d{3}-\d{2}-\d{4}\b', '***SSN***'),  # SSN
        ]
    
    def _mask_pii(self, message: str) -> str:
        """Mask PII in log messages."""
        for pattern, replacement in self.pii_patterns:
            message = re.sub(pattern, replacement, message)
        return message
    
    def _add_correlation_id(self, message: str) -> str:
        """Add correlation ID to log message."""
        correlation_id = CorrelationContext.get_id()
        return f"[{correlation_id}] {message}"
    
    def info(self, message: str, **kwargs):
        """Log info message with PII masking and correlation ID."""
        safe_message = self._mask_pii(self._add_correlation_id(message))
        self.logger.info(safe_message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with PII masking and correlation ID."""
        safe_message = self._mask_pii(self._add_correlation_id(message))
        self.logger.error(safe_message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with PII masking and correlation ID."""
        safe_message = self._mask_pii(self._add_correlation_id(message))
        self.logger.warning(safe_message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with PII masking and correlation ID."""
        safe_message = self._mask_pii(self._add_correlation_id(message))
        self.logger.debug(safe_message, **kwargs)


class JWTAuth:
    """JWT authentication handler."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.security = HTTPBearer()
    
    def create_token(self, user_id: str, permissions: Optional[List[str]] = None) -> str:
        """Create JWT token."""
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiry_hours),
            "iat": datetime.utcnow(),
            "correlation_id": CorrelationContext.get_id()
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def __call__(self, credentials: Optional[HTTPAuthorizationCredentials] = None) -> Dict[str, Any]:
        """FastAPI dependency for JWT authentication."""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )
        
        return self.verify_token(credentials.credentials)


class RateLimiter:
    """Redis-based rate limiter."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.redis_client = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not redis:
            logger.warning("Redis module not available. Rate limiting will be disabled.")
            return
            
        try:
            self.redis_client = redis.Redis.from_url(self.config.redis_url, decode_responses=True)
            await self.redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None  # Disable rate limiting if Redis fails
    
    async def is_allowed(self, identifier: str, limit_type: str = "minute") -> bool:
        """Check if request is allowed based on rate limit."""
        if not self.redis_client or not redis:
            return True  # Allow if Redis is not available
        
        try:
            now = int(time.time())
            window_size = 60 if limit_type == "minute" else 3600  # 1 hour
            limit = self.config.rate_limit_per_minute if limit_type == "minute" else self.config.rate_limit_per_hour
            
            window_start = now - (now % window_size)
            key = f"rate_limit:{identifier}:{limit_type}:{window_start}"
            
            # Use synchronous Redis methods since async redis might not be available
            current_count = self.redis_client.get(key)
            if current_count is None:
                self.redis_client.setex(key, window_size, 1)
                return True
            
            if int(current_count) >= limit:
                return False
            
            self.redis_client.incr(key)
            return True
        
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True  # Allow on error to avoid blocking legitimate traffic


class InputSanitizer:
    """Input sanitization utilities."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 4000) -> str:
        """Sanitize text input."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate to max length
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove potential script injections
        dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
            r'javascript:',
            r'on\w+\s*=',
            r'data:text/html',
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format."""
        if not phone:
            return False
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Check if it's a valid length (7-15 digits)
        return 7 <= len(digits) <= 15
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))


class WebhookVerifier:
    """Webhook signature verification."""
    
    def __init__(self, secret: str):
        self.secret = secret.encode('utf-8') if isinstance(secret, str) else secret
    
    def verify_signature(self, payload: bytes, signature: str, algorithm: str = "sha256") -> bool:
        """Verify webhook signature."""
        try:
            if algorithm == "sha256":
                expected_signature = hmac.new(
                    self.secret,
                    payload,
                    hashlib.sha256
                ).hexdigest()
                
                # Handle different signature formats
                if signature.startswith('sha256='):
                    signature = signature[7:]
                
                return hmac.compare_digest(expected_signature, signature)
            
            elif algorithm == "sha1":
                expected_signature = hmac.new(
                    self.secret,
                    payload,
                    hashlib.sha1
                ).hexdigest()
                
                if signature.startswith('sha1='):
                    signature = signature[6:]
                
                return hmac.compare_digest(expected_signature, signature)
            
            else:
                logger.error(f"Unsupported signature algorithm: {algorithm}")
                return False
                
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


class HTTPSEnforcer:
    """HTTPS enforcement middleware."""
    
    def __init__(self, enforce_https: bool = True):
        self.enforce_https = enforce_https
    
    async def __call__(self, request: Request, call_next):
        """Middleware to enforce HTTPS."""
        if self.enforce_https:
            if request.url.scheme != "https" and not request.url.hostname in ["localhost", "127.0.0.1"]:
                https_url = request.url.replace(scheme="https")
                return HTTPException(
                    status_code=status.HTTP_301_MOVED_PERMANENTLY,
                    headers={"Location": str(https_url)}
                )
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
            
            if last_exception:
                raise last_exception
            else:
                raise Exception("All retry attempts failed")
        
        return wrapper
    return decorator
