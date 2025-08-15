"""
Tenant Management System
=======================

Enterprise-grade multi-tenancy with complete data and configuration isolation.

PHASE3-REFACTOR: Tenant isolation for scalable SaaS deployment.

Features:
- Tenant lifecycle management
- Data isolation and encryption
- Per-tenant resource quotas
- Configuration inheritance
- Tenant analytics and monitoring
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"
    PENDING_ACTIVATION = "pending_activation"
    DELETED = "deleted"


class TenantTier(Enum):
    """Tenant service tiers."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantQuotas:
    """Resource quotas for tenant."""
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000
    max_storage_mb: int = 100
    max_users: int = 10
    max_channels: int = 3
    max_retrievers: int = 2
    max_llm_tokens_per_hour: int = 50000
    
    # Feature flags
    allow_custom_models: bool = False
    allow_api_access: bool = True
    allow_analytics_export: bool = False
    allow_webhook_integrations: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_requests_per_day": self.max_requests_per_day,
            "max_storage_mb": self.max_storage_mb,
            "max_users": self.max_users,
            "max_channels": self.max_channels,
            "max_retrievers": self.max_retrievers,
            "max_llm_tokens_per_hour": self.max_llm_tokens_per_hour,
            "features": {
                "custom_models": self.allow_custom_models,
                "api_access": self.allow_api_access,
                "analytics_export": self.allow_analytics_export,
                "webhook_integrations": self.allow_webhook_integrations
            }
        }


@dataclass
class TenantUsage:
    """Current usage metrics for tenant."""
    requests_today: int = 0
    requests_this_hour: int = 0
    storage_used_mb: float = 0.0
    active_users: int = 0
    llm_tokens_this_hour: int = 0
    
    # Reset timestamps
    last_hourly_reset: datetime = field(default_factory=datetime.utcnow)
    last_daily_reset: datetime = field(default_factory=datetime.utcnow)
    
    def reset_hourly(self):
        """Reset hourly counters."""
        self.requests_this_hour = 0
        self.llm_tokens_this_hour = 0
        self.last_hourly_reset = datetime.utcnow()
    
    def reset_daily(self):
        """Reset daily counters."""
        self.requests_today = 0
        self.last_daily_reset = datetime.utcnow()
    
    def check_quota_exceeded(self, quotas: TenantQuotas) -> List[str]:
        """Check which quotas are exceeded."""
        violations = []
        
        if self.requests_this_hour >= quotas.max_requests_per_hour:
            violations.append("hourly_requests")
        
        if self.requests_today >= quotas.max_requests_per_day:
            violations.append("daily_requests")
        
        if self.storage_used_mb >= quotas.max_storage_mb:
            violations.append("storage")
        
        if self.active_users >= quotas.max_users:
            violations.append("users")
        
        if self.llm_tokens_this_hour >= quotas.max_llm_tokens_per_hour:
            violations.append("llm_tokens")
        
        return violations


@dataclass
class Tenant:
    """
    Tenant entity with complete isolation configuration.
    
    PHASE3-REFACTOR: Complete tenant data model for enterprise multi-tenancy.
    """
    id: str
    name: str
    domain: str
    status: TenantStatus = TenantStatus.PENDING_ACTIVATION
    tier: TenantTier = TenantTier.FREE
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: Optional[datetime] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    
    # Resource management
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    usage: TenantUsage = field(default_factory=TenantUsage)
    
    # Database isolation
    database_schema: Optional[str] = None
    database_prefix: Optional[str] = None
    encryption_key: Optional[str] = None
    
    # Contact and billing
    admin_email: str = ""
    billing_email: str = ""
    
    # Custom fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.database_schema:
            self.database_schema = f"tenant_{self.id}"
        if not self.database_prefix:
            self.database_prefix = f"t_{self.id}_"
        if not self.encryption_key:
            self.encryption_key = str(uuid.uuid4())
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE
    
    def is_trial(self) -> bool:
        """Check if tenant is in trial."""
        return self.status == TenantStatus.TRIAL
    
    def can_make_request(self) -> bool:
        """Check if tenant can make requests."""
        if not self.is_active() and not self.is_trial():
            return False
        
        violations = self.usage.check_quota_exceeded(self.quotas)
        return len(violations) == 0
    
    def get_quota_violations(self) -> List[str]:
        """Get current quota violations."""
        return self.usage.check_quota_exceeded(self.quotas)
    
    def update_usage(self, **kwargs):
        """Update usage metrics."""
        for key, value in kwargs.items():
            if hasattr(self.usage, key):
                setattr(self.usage, key, value)
        
        self.last_active_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration for tenant."""
        return {
            "schema": self.database_schema or f"tenant_{self.id}",
            "table_prefix": self.database_prefix or f"t_{self.id}_",
            "encryption_key": self.encryption_key or str(uuid.uuid4())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tenant to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "status": self.status.value,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "config": self.config,
            "quotas": self.quotas.to_dict(),
            "admin_email": self.admin_email,
            "billing_email": self.billing_email,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tenant':
        """Create tenant from dictionary."""
        tenant = cls(
            id=data["id"],
            name=data["name"],
            domain=data["domain"],
            status=TenantStatus(data.get("status", "pending_activation")),
            tier=TenantTier(data.get("tier", "free")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
            config=data.get("config", {}),
            admin_email=data.get("admin_email", ""),
            billing_email=data.get("billing_email", ""),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )
        
        if data.get("last_active_at"):
            tenant.last_active_at = datetime.fromisoformat(data["last_active_at"])
        
        # Load quotas
        quota_data = data.get("quotas", {})
        if quota_data:
            features = quota_data.get("features", {})
            tenant.quotas = TenantQuotas(
                max_requests_per_hour=quota_data.get("max_requests_per_hour", 1000),
                max_requests_per_day=quota_data.get("max_requests_per_day", 10000),
                max_storage_mb=quota_data.get("max_storage_mb", 100),
                max_users=quota_data.get("max_users", 10),
                max_channels=quota_data.get("max_channels", 3),
                max_retrievers=quota_data.get("max_retrievers", 2),
                max_llm_tokens_per_hour=quota_data.get("max_llm_tokens_per_hour", 50000),
                allow_custom_models=features.get("custom_models", False),
                allow_api_access=features.get("api_access", True),
                allow_analytics_export=features.get("analytics_export", False),
                allow_webhook_integrations=features.get("webhook_integrations", False)
            )
        
        return tenant


@dataclass 
class TenantContext:
    """
    Runtime context for tenant-aware operations.
    
    PHASE3-REFACTOR: Request-scoped tenant context for isolation.
    """
    tenant: Tenant
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Performance tracking
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since context creation."""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "tenant_id": self.tenant.id,
            "tenant_name": self.tenant.name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "ip_address": self.ip_address,
            "elapsed_time": self.get_elapsed_time()
        }


class ITenantStore(ABC):
    """Interface for tenant data persistence."""
    
    @abstractmethod
    async def create_tenant(self, tenant: Tenant) -> bool:
        """Create new tenant."""
        pass
    
    @abstractmethod
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        pass
    
    @abstractmethod
    async def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        pass
    
    @abstractmethod
    async def update_tenant(self, tenant: Tenant) -> bool:
        """Update tenant."""
        pass
    
    @abstractmethod
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant."""
        pass
    
    @abstractmethod
    async def list_tenants(self, status: Optional[TenantStatus] = None, 
                          tier: Optional[TenantTier] = None) -> List[Tenant]:
        """List tenants with optional filters."""
        pass
    
    @abstractmethod
    async def update_usage(self, tenant_id: str, usage: TenantUsage) -> bool:
        """Update tenant usage metrics."""
        pass


class MemoryTenantStore(ITenantStore):
    """In-memory tenant store for development."""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.domain_index: Dict[str, str] = {}  # domain -> tenant_id
    
    async def create_tenant(self, tenant: Tenant) -> bool:
        """Create new tenant."""
        if tenant.id in self.tenants:
            return False
        
        if tenant.domain in self.domain_index:
            return False
        
        self.tenants[tenant.id] = tenant
        self.domain_index[tenant.domain] = tenant.id
        return True
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    async def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        tenant_id = self.domain_index.get(domain)
        return self.tenants.get(tenant_id) if tenant_id else None
    
    async def update_tenant(self, tenant: Tenant) -> bool:
        """Update tenant."""
        if tenant.id not in self.tenants:
            return False
        
        # Update domain index if changed
        old_tenant = self.tenants[tenant.id]
        if old_tenant.domain != tenant.domain:
            del self.domain_index[old_tenant.domain]
            self.domain_index[tenant.domain] = tenant.id
        
        tenant.updated_at = datetime.utcnow()
        self.tenants[tenant.id] = tenant
        return True
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant."""
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        del self.domain_index[tenant.domain]
        del self.tenants[tenant_id]
        return True
    
    async def list_tenants(self, status: Optional[TenantStatus] = None, 
                          tier: Optional[TenantTier] = None) -> List[Tenant]:
        """List tenants with optional filters."""
        tenants = list(self.tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        if tier:
            tenants = [t for t in tenants if t.tier == tier]
        
        return tenants
    
    async def update_usage(self, tenant_id: str, usage: TenantUsage) -> bool:
        """Update tenant usage metrics."""
        if tenant_id not in self.tenants:
            return False
        
        self.tenants[tenant_id].usage = usage
        self.tenants[tenant_id].updated_at = datetime.utcnow()
        return True


class TenantManager:
    """
    Central tenant management service.
    
    PHASE3-REFACTOR: Enterprise tenant lifecycle and quota management.
    """
    
    def __init__(self, tenant_store: Optional[ITenantStore] = None):
        self.tenant_store = tenant_store or MemoryTenantStore()
        
        # Tier configurations
        self.tier_quotas = {
            TenantTier.FREE: TenantQuotas(
                max_requests_per_hour=100,
                max_requests_per_day=500,
                max_storage_mb=10,
                max_users=1,
                max_channels=1,
                max_retrievers=1,
                max_llm_tokens_per_hour=5000,
                allow_custom_models=False,
                allow_api_access=False,
                allow_analytics_export=False,
                allow_webhook_integrations=False
            ),
            TenantTier.BASIC: TenantQuotas(
                max_requests_per_hour=1000,
                max_requests_per_day=5000,
                max_storage_mb=100,
                max_users=5,
                max_channels=2,
                max_retrievers=2,
                max_llm_tokens_per_hour=25000,
                allow_custom_models=False,
                allow_api_access=True,
                allow_analytics_export=False,
                allow_webhook_integrations=False
            ),
            TenantTier.PROFESSIONAL: TenantQuotas(
                max_requests_per_hour=5000,
                max_requests_per_day=25000,
                max_storage_mb=500,
                max_users=25,
                max_channels=5,
                max_retrievers=5,
                max_llm_tokens_per_hour=100000,
                allow_custom_models=True,
                allow_api_access=True,
                allow_analytics_export=True,
                allow_webhook_integrations=True
            ),
            TenantTier.ENTERPRISE: TenantQuotas(
                max_requests_per_hour=50000,
                max_requests_per_day=500000,
                max_storage_mb=5000,
                max_users=1000,
                max_channels=50,
                max_retrievers=20,
                max_llm_tokens_per_hour=1000000,
                allow_custom_models=True,
                allow_api_access=True,
                allow_analytics_export=True,
                allow_webhook_integrations=True
            )
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.usage_reset_task: Optional[asyncio.Task] = None
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.usage_reset_task = asyncio.create_task(self._usage_reset_loop())
    
    async def create_tenant(self, name: str, domain: str, tier: TenantTier = TenantTier.FREE,
                           admin_email: str = "", config: Optional[Dict[str, Any]] = None) -> Tenant:
        """Create new tenant."""
        tenant_id = str(uuid.uuid4())
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            domain=domain,
            tier=tier,
            admin_email=admin_email,
            config=config or {},
            quotas=self.tier_quotas[tier]
        )
        
        success = await self.tenant_store.create_tenant(tenant)
        if not success:
            raise ValueError(f"Failed to create tenant: domain '{domain}' may already exist")
        
        logger.info(f"Created tenant '{name}' ({tenant_id}) with tier {tier.value}")
        return tenant
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return await self.tenant_store.get_tenant(tenant_id)
    
    async def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        return await self.tenant_store.get_tenant_by_domain(domain)
    
    async def update_tenant_tier(self, tenant_id: str, new_tier: TenantTier) -> bool:
        """Update tenant tier and quotas."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        old_tier = tenant.tier
        tenant.tier = new_tier
        tenant.quotas = self.tier_quotas[new_tier]
        
        success = await self.tenant_store.update_tenant(tenant)
        if success:
            logger.info(f"Updated tenant {tenant_id} tier from {old_tier.value} to {new_tier.value}")
        
        return success
    
    async def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Suspend tenant account."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.status = TenantStatus.SUSPENDED
        tenant.metadata["suspension_reason"] = reason
        tenant.metadata["suspended_at"] = datetime.utcnow().isoformat()
        
        success = await self.tenant_store.update_tenant(tenant)
        if success:
            logger.warning(f"Suspended tenant {tenant_id}: {reason}")
        
        return success
    
    async def activate_tenant(self, tenant_id: str) -> bool:
        """Activate tenant account."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.status = TenantStatus.ACTIVE
        if "suspension_reason" in tenant.metadata:
            del tenant.metadata["suspension_reason"]
        if "suspended_at" in tenant.metadata:
            del tenant.metadata["suspended_at"]
        
        success = await self.tenant_store.update_tenant(tenant)
        if success:
            logger.info(f"Activated tenant {tenant_id}")
        
        return success
    
    async def record_request(self, tenant_id: str, tokens_used: int = 0) -> bool:
        """Record request and update usage."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        # Update usage counters
        tenant.usage.requests_today += 1
        tenant.usage.requests_this_hour += 1
        tenant.usage.llm_tokens_this_hour += tokens_used
        
        tenant.last_active_at = datetime.utcnow()
        
        # Save updated usage
        await self.tenant_store.update_usage(tenant_id, tenant.usage)
        return True
    
    async def check_quota_compliance(self, tenant_id: str) -> Dict[str, Any]:
        """Check tenant quota compliance."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}
        
        violations = tenant.get_quota_violations()
        
        return {
            "tenant_id": tenant_id,
            "can_make_request": len(violations) == 0,
            "violations": violations,
            "usage": {
                "requests_today": tenant.usage.requests_today,
                "requests_this_hour": tenant.usage.requests_this_hour,
                "storage_used_mb": tenant.usage.storage_used_mb,
                "llm_tokens_this_hour": tenant.usage.llm_tokens_this_hour
            },
            "quotas": tenant.quotas.to_dict()
        }
    
    async def get_tenant_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant analytics and usage statistics."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return {"error": "Tenant not found"}
        
        return {
            "tenant": tenant.to_dict(),
            "compliance": await self.check_quota_compliance(tenant_id),
            "account_age_days": (datetime.utcnow() - tenant.created_at).days,
            "last_active_days_ago": (datetime.utcnow() - tenant.last_active_at).days if tenant.last_active_at else None
        }
    
    async def _cleanup_loop(self):
        """Background cleanup of expired tenants."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up expired trial tenants
                trial_tenants = await self.tenant_store.list_tenants(status=TenantStatus.TRIAL)
                
                for tenant in trial_tenants:
                    # Trial expires after 30 days
                    if (datetime.utcnow() - tenant.created_at).days > 30:
                        tenant.status = TenantStatus.EXPIRED
                        await self.tenant_store.update_tenant(tenant)
                        logger.info(f"Expired trial tenant {tenant.id}")
                
            except Exception as e:
                logger.error(f"Tenant cleanup error: {e}")
    
    async def _usage_reset_loop(self):
        """Background task to reset usage counters."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Reset hourly counters every hour
                if current_time.minute == 0:
                    tenants = await self.tenant_store.list_tenants()
                    for tenant in tenants:
                        if (current_time - tenant.usage.last_hourly_reset).total_seconds() >= 3600:
                            tenant.usage.reset_hourly()
                            await self.tenant_store.update_usage(tenant.id, tenant.usage)
                
                # Reset daily counters at midnight
                if current_time.hour == 0 and current_time.minute == 0:
                    tenants = await self.tenant_store.list_tenants()
                    for tenant in tenants:
                        if (current_time - tenant.usage.last_daily_reset).days >= 1:
                            tenant.usage.reset_daily()
                            await self.tenant_store.update_usage(tenant.id, tenant.usage)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Usage reset error: {e}")
    
    async def shutdown(self):
        """Shutdown tenant manager."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.usage_reset_task:
            self.usage_reset_task.cancel()
        
        logger.info("Tenant manager shutdown complete")


# Global tenant manager
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get global tenant manager."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager


async def initialize_tenant_manager(tenant_store: Optional[ITenantStore] = None):
    """Initialize global tenant manager."""
    global _tenant_manager
    _tenant_manager = TenantManager(tenant_store)
    logger.info("Global tenant manager initialized")


async def shutdown_tenant_manager():
    """Shutdown global tenant manager."""
    global _tenant_manager
    if _tenant_manager:
        await _tenant_manager.shutdown()
        _tenant_manager = None
    logger.info("Global tenant manager shutdown")


@asynccontextmanager
async def tenant_context(tenant: Tenant, **kwargs):
    """Context manager for tenant-aware operations."""
    context = TenantContext(tenant=tenant, **kwargs)
    
    # Record request start
    tenant_manager = get_tenant_manager()
    await tenant_manager.record_request(tenant.id)
    
    try:
        yield context
    finally:
        # Log request completion
        logger.debug(f"Request completed for tenant {tenant.id} in {context.get_elapsed_time():.3f}s")
