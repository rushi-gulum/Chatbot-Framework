"""
Tenant Middleware and Request Resolution
=======================================

Middleware for tenant detection and request routing in multi-tenant environment.

PHASE3-REFACTOR: Request-level tenant isolation and routing.

Features:
- Multiple tenant resolution strategies (domain, header, token, subdomain)
- Request validation and quota enforcement
- Tenant context injection
- Rate limiting per tenant
- Comprehensive request logging
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
import urllib.parse
from collections import defaultdict
import time

from .tenant_manager import Tenant, TenantContext, TenantManager, get_tenant_manager

logger = logging.getLogger(__name__)


class TenantResolutionStrategy(Enum):
    """Strategies for resolving tenant from request."""
    DOMAIN = "domain"
    SUBDOMAIN = "subdomain"
    HEADER = "header"
    TOKEN = "token"
    PATH_PREFIX = "path_prefix"
    QUERY_PARAM = "query_param"


@dataclass
class TenantResolutionConfig:
    """Configuration for tenant resolution."""
    strategy: TenantResolutionStrategy = TenantResolutionStrategy.DOMAIN
    
    # Header-based resolution
    header_name: str = "X-Tenant-ID"
    
    # Token-based resolution
    token_header: str = "Authorization"
    token_prefix: str = "Bearer "
    
    # Subdomain resolution
    domain_suffix: str = ".chatbot.example.com"
    
    # Path prefix resolution
    path_pattern: str = r"^/tenant/([^/]+)/"
    
    # Query parameter resolution
    query_param: str = "tenant_id"
    
    # Fallback options
    default_tenant_id: Optional[str] = None
    allow_fallback: bool = True


class TenantResolutionError(Exception):
    """Tenant resolution errors."""
    pass


class TenantQuotaExceededError(Exception):
    """Tenant quota exceeded error."""
    def __init__(self, tenant_id: str, violations: List[str]):
        self.tenant_id = tenant_id
        self.violations = violations
        super().__init__(f"Quota exceeded for tenant {tenant_id}: {', '.join(violations)}")


class ITenantResolver(ABC):
    """Interface for tenant resolution strategies."""
    
    @abstractmethod
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant ID from request data."""
        pass


class DomainTenantResolver(ITenantResolver):
    """Resolve tenant by exact domain match."""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
    
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant by domain."""
        host = request_data.get("host", "").lower()
        if not host:
            return None
        
        tenant = await self.tenant_manager.get_tenant_by_domain(host)
        return tenant.id if tenant else None


class SubdomainTenantResolver(ITenantResolver):
    """Resolve tenant by subdomain extraction."""
    
    def __init__(self, tenant_manager: TenantManager, domain_suffix: str):
        self.tenant_manager = tenant_manager
        self.domain_suffix = domain_suffix.lower()
    
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant by subdomain."""
        host = request_data.get("host", "").lower()
        if not host or not host.endswith(self.domain_suffix):
            return None
        
        subdomain = host[:-len(self.domain_suffix)]
        if not subdomain:
            return None
        
        # Try to find tenant by subdomain as domain
        tenant = await self.tenant_manager.get_tenant_by_domain(subdomain)
        return tenant.id if tenant else None


class HeaderTenantResolver(ITenantResolver):
    """Resolve tenant by HTTP header."""
    
    def __init__(self, tenant_manager: TenantManager, header_name: str):
        self.tenant_manager = tenant_manager
        self.header_name = header_name.lower()
    
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant by header."""
        headers = request_data.get("headers", {})
        
        # Case-insensitive header lookup
        for header, value in headers.items():
            if header.lower() == self.header_name:
                tenant_id = value.strip()
                if tenant_id:
                    # Verify tenant exists
                    tenant = await self.tenant_manager.get_tenant(tenant_id)
                    return tenant.id if tenant else None
        
        return None


class TokenTenantResolver(ITenantResolver):
    """Resolve tenant by extracting from JWT token."""
    
    def __init__(self, tenant_manager: TenantManager, token_header: str, token_prefix: str):
        self.tenant_manager = tenant_manager
        self.token_header = token_header
        self.token_prefix = token_prefix
    
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant by token."""
        headers = request_data.get("headers", {})
        
        auth_header = None
        for header, value in headers.items():
            if header.lower() == self.token_header.lower():
                auth_header = value
                break
        
        if not auth_header or not auth_header.startswith(self.token_prefix):
            return None
        
        token = auth_header[len(self.token_prefix):].strip()
        if not token:
            return None
        
        # Extract tenant from token (simplified - in production use proper JWT parsing)
        try:
            import base64
            import json
            
            # Decode JWT payload (without verification for simplicity)
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            payload = parts[1]
            # Add padding if needed
            payload += '=' * (4 - len(payload) % 4)
            
            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            
            tenant_id = data.get('tenant_id')
            if tenant_id:
                # Verify tenant exists
                tenant = await self.tenant_manager.get_tenant(tenant_id)
                return tenant.id if tenant else None
        
        except Exception as e:
            logger.warning(f"Token parsing error: {e}")
        
        return None


class PathPrefixTenantResolver(ITenantResolver):
    """Resolve tenant by path prefix pattern."""
    
    def __init__(self, tenant_manager: TenantManager, path_pattern: str):
        self.tenant_manager = tenant_manager
        self.path_pattern = re.compile(path_pattern)
    
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant by path prefix."""
        path = request_data.get("path", "")
        if not path:
            return None
        
        match = self.path_pattern.match(path)
        if not match:
            return None
        
        tenant_id = match.group(1)
        if tenant_id:
            # Verify tenant exists
            tenant = await self.tenant_manager.get_tenant(tenant_id)
            return tenant.id if tenant else None
        
        return None


class QueryParamTenantResolver(ITenantResolver):
    """Resolve tenant by query parameter."""
    
    def __init__(self, tenant_manager: TenantManager, query_param: str):
        self.tenant_manager = tenant_manager
        self.query_param = query_param
    
    async def resolve_tenant(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Resolve tenant by query parameter."""
        query_string = request_data.get("query_string", "")
        if not query_string:
            return None
        
        params = urllib.parse.parse_qs(query_string)
        tenant_values = params.get(self.query_param, [])
        
        if tenant_values:
            tenant_id = tenant_values[0]
            # Verify tenant exists
            tenant = await self.tenant_manager.get_tenant(tenant_id)
            return tenant.id if tenant else None
        
        return None


class TenantResolver:
    """
    Multi-strategy tenant resolver.
    
    PHASE3-REFACTOR: Flexible tenant resolution with multiple strategies.
    """
    
    def __init__(self, config: TenantResolutionConfig, tenant_manager: Optional[TenantManager] = None):
        self.config = config
        self.tenant_manager = tenant_manager or get_tenant_manager()
        
        # Initialize resolver based on strategy
        self.resolver = self._create_resolver()
    
    def _create_resolver(self) -> ITenantResolver:
        """Create resolver based on configuration."""
        if self.config.strategy == TenantResolutionStrategy.DOMAIN:
            return DomainTenantResolver(self.tenant_manager)
        
        elif self.config.strategy == TenantResolutionStrategy.SUBDOMAIN:
            return SubdomainTenantResolver(self.tenant_manager, self.config.domain_suffix)
        
        elif self.config.strategy == TenantResolutionStrategy.HEADER:
            return HeaderTenantResolver(self.tenant_manager, self.config.header_name)
        
        elif self.config.strategy == TenantResolutionStrategy.TOKEN:
            return TokenTenantResolver(self.tenant_manager, self.config.token_header, self.config.token_prefix)
        
        elif self.config.strategy == TenantResolutionStrategy.PATH_PREFIX:
            return PathPrefixTenantResolver(self.tenant_manager, self.config.path_pattern)
        
        elif self.config.strategy == TenantResolutionStrategy.QUERY_PARAM:
            return QueryParamTenantResolver(self.tenant_manager, self.config.query_param)
        
        else:
            raise ValueError(f"Unsupported tenant resolution strategy: {self.config.strategy}")
    
    async def resolve(self, request_data: Dict[str, Any]) -> Optional[Tenant]:
        """Resolve tenant from request data."""
        try:
            tenant_id = await self.resolver.resolve_tenant(request_data)
            
            if not tenant_id and self.config.allow_fallback and self.config.default_tenant_id:
                tenant_id = self.config.default_tenant_id
            
            if not tenant_id:
                return None
            
            tenant = await self.tenant_manager.get_tenant(tenant_id)
            if not tenant:
                logger.warning(f"Tenant {tenant_id} not found")
                return None
            
            return tenant
        
        except Exception as e:
            logger.error(f"Tenant resolution error: {e}")
            if self.config.allow_fallback and self.config.default_tenant_id:
                return await self.tenant_manager.get_tenant(self.config.default_tenant_id)
            return None


class TenantRateLimiter:
    """Rate limiter with per-tenant tracking."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """Remove old request timestamps."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep last hour
        
        for tenant_id in list(self.requests.keys()):
            self.requests[tenant_id] = [
                timestamp for timestamp in self.requests[tenant_id]
                if timestamp > cutoff_time
            ]
            
            if not self.requests[tenant_id]:
                del self.requests[tenant_id]
        
        self.last_cleanup = current_time
    
    def check_rate_limit(self, tenant: Tenant) -> bool:
        """Check if tenant is within rate limits."""
        self._cleanup_old_requests()
        
        current_time = time.time()
        tenant_requests = self.requests[tenant.id]
        
        # Check hourly limit
        hour_ago = current_time - 3600
        requests_this_hour = len([
            t for t in tenant_requests if t > hour_ago
        ])
        
        if requests_this_hour >= tenant.quotas.max_requests_per_hour:
            return False
        
        # Check daily limit (approximate using last 24 hours)
        day_ago = current_time - 86400
        requests_today = len([
            t for t in tenant_requests if t > day_ago
        ])
        
        if requests_today >= tenant.quotas.max_requests_per_day:
            return False
        
        return True
    
    def record_request(self, tenant_id: str):
        """Record request for tenant."""
        self.requests[tenant_id].append(time.time())


class TenantMiddleware:
    """
    Tenant-aware middleware for request processing.
    
    PHASE3-REFACTOR: Complete tenant isolation middleware.
    """
    
    def __init__(self, resolution_config: Optional[TenantResolutionConfig] = None,
                 tenant_manager: Optional[TenantManager] = None):
        self.resolution_config = resolution_config or TenantResolutionConfig()
        self.tenant_manager = tenant_manager or get_tenant_manager()
        self.resolver = TenantResolver(self.resolution_config, self.tenant_manager)
        self.rate_limiter = TenantRateLimiter()
        
        # Request tracking
        self.active_requests: Dict[str, TenantContext] = {}
    
    async def process_request(self, request_data: Dict[str, Any]) -> TenantContext:
        """
        Process incoming request and create tenant context.
        
        Args:
            request_data: Request information (host, headers, path, etc.)
        
        Returns:
            TenantContext for the request
        
        Raises:
            TenantResolutionError: If tenant cannot be resolved
            TenantQuotaExceededError: If tenant quota is exceeded
        """
        # Resolve tenant
        tenant = await self.resolver.resolve(request_data)
        if not tenant:
            raise TenantResolutionError("Could not resolve tenant from request")
        
        # Check tenant status
        if not tenant.is_active() and not tenant.is_trial():
            raise TenantResolutionError(f"Tenant {tenant.id} is not active (status: {tenant.status.value})")
        
        # Check rate limits
        if not self.rate_limiter.check_rate_limit(tenant):
            violations = tenant.get_quota_violations()
            raise TenantQuotaExceededError(tenant.id, violations)
        
        # Record request
        self.rate_limiter.record_request(tenant.id)
        await self.tenant_manager.record_request(tenant.id)
        
        # Create context
        context = TenantContext(
            tenant=tenant,
            user_id=request_data.get("user_id"),
            session_id=request_data.get("session_id"),
            ip_address=request_data.get("ip_address"),
            user_agent=request_data.get("headers", {}).get("user-agent", ""),
            headers=request_data.get("headers", {})
        )
        
        # Track active request
        self.active_requests[context.request_id] = context
        
        logger.debug(f"Processing request for tenant {tenant.id} ({tenant.name})")
        return context
    
    async def finalize_request(self, context: TenantContext, success: bool = True,
                             error: Optional[Exception] = None):
        """Finalize request processing."""
        # Remove from active requests
        if context.request_id in self.active_requests:
            del self.active_requests[context.request_id]
        
        # Log request completion
        elapsed_time = context.get_elapsed_time()
        
        log_data = {
            "tenant_id": context.tenant.id,
            "request_id": context.request_id,
            "elapsed_time": elapsed_time,
            "success": success,
            "error": str(error) if error else None
        }
        
        if success:
            logger.info(f"Request completed successfully", extra=log_data)
        else:
            logger.error(f"Request failed", extra=log_data)
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        active_count = len(self.active_requests)
        
        # Count requests per tenant
        tenant_counts = defaultdict(int)
        for context in self.active_requests.values():
            tenant_counts[context.tenant.id] += 1
        
        return {
            "active_requests": active_count,
            "requests_per_tenant": dict(tenant_counts),
            "resolution_strategy": self.resolution_config.strategy.value
        }


# Decorators for tenant-aware functions
def require_tenant(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator to require tenant context for function execution."""
    async def wrapper(*args, **kwargs):
        # Look for tenant context in arguments
        context = None
        for arg in args:
            if isinstance(arg, TenantContext):
                context = arg
                break
        
        if not context:
            context = kwargs.get('tenant_context')
        
        if not context:
            raise TenantResolutionError("Tenant context required but not provided")
        
        return await func(*args, **kwargs)
    
    return wrapper


def tenant_aware(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator to make function tenant-aware with automatic context injection."""
    async def wrapper(*args, **kwargs):
        # If tenant context not provided, try to resolve from current request
        context = kwargs.get('tenant_context')
        
        if not context:
            # In a real implementation, this would get context from request scope
            # For now, we'll pass through without context
            pass
        
        return await func(*args, **kwargs)
    
    return wrapper
