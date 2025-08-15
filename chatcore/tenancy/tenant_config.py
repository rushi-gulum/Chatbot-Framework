"""
Tenant Configuration Management
==============================

Per-tenant configuration system with inheritance and validation.

PHASE3-REFACTOR: Tenant-specific configuration isolation and management.

Features:
- Hierarchical configuration inheritance (global -> tier -> tenant)
- Configuration validation and schema enforcement
- Encrypted sensitive configuration storage
- Dynamic configuration updates
- Configuration versioning and rollback
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import yaml
from cryptography.fernet import Fernet
import base64

from .tenant_manager import Tenant, TenantTier

logger = logging.getLogger(__name__)


class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    TIER = "tier"
    TENANT = "tenant"


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SECRET = "secret"  # Encrypted value


@dataclass
class ConfigSchema:
    """Schema definition for configuration values."""
    key: str
    config_type: ConfigType
    required: bool = False
    default_value: Any = None
    description: str = ""
    validation_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    
    def validate(self, value: Any) -> bool:
        """Validate configuration value against schema."""
        if value is None:
            return not self.required
        
        # Type validation
        if self.config_type == ConfigType.STRING and not isinstance(value, str):
            return False
        elif self.config_type == ConfigType.INTEGER and not isinstance(value, int):
            return False
        elif self.config_type == ConfigType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.config_type == ConfigType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.config_type == ConfigType.LIST and not isinstance(value, list):
            return False
        elif self.config_type == ConfigType.DICT and not isinstance(value, dict):
            return False
        
        # Allowed values validation
        if self.allowed_values and value not in self.allowed_values:
            return False
        
        # Range validation (only for numeric types)
        if self.min_value is not None and isinstance(value, (int, float)) and value < self.min_value:
            return False
        if self.max_value is not None and isinstance(value, (int, float)) and value > self.max_value:
            return False
        
        # Pattern validation
        if self.validation_pattern and isinstance(value, str):
            import re
            if not re.match(self.validation_pattern, value):
                return False
        
        return True


@dataclass
class ConfigVersion:
    """Configuration version for rollback support."""
    version: int
    config: Dict[str, Any]
    created_at: datetime
    created_by: str = ""
    description: str = ""


@dataclass
class TenantConfig:
    """
    Tenant-specific configuration with inheritance and validation.
    
    PHASE3-REFACTOR: Complete tenant configuration isolation.
    """
    tenant_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    encrypted_config: Dict[str, str] = field(default_factory=dict)
    
    # Versioning
    version: int = 1
    versions: List[ConfigVersion] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set_value(self, key: str, value: Any, updated_by: str = ""):
        """Set configuration value."""
        old_value = self.config.get(key)
        self.config[key] = value
        self.updated_at = datetime.utcnow()
        self.updated_by = updated_by
        
        logger.debug(f"Updated config {key} for tenant {self.tenant_id}: {old_value} -> {value}")
    
    def remove_value(self, key: str, updated_by: str = ""):
        """Remove configuration value."""
        if key in self.config:
            del self.config[key]
            self.updated_at = datetime.utcnow()
            self.updated_by = updated_by
    
    def create_version(self, description: str = "", created_by: str = ""):
        """Create new configuration version."""
        version = ConfigVersion(
            version=self.version,
            config=self.config.copy(),
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description
        )
        
        self.versions.append(version)
        self.version += 1
        
        # Keep only last 10 versions
        if len(self.versions) > 10:
            self.versions.pop(0)
    
    def rollback_to_version(self, version_number: int, updated_by: str = "") -> bool:
        """Rollback to specific configuration version."""
        target_version = None
        for version in self.versions:
            if version.version == version_number:
                target_version = version
                break
        
        if not target_version:
            return False
        
        self.config = target_version.config.copy()
        self.updated_at = datetime.utcnow()
        self.updated_by = updated_by
        
        logger.info(f"Rolled back tenant {self.tenant_id} config to version {version_number}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "config": self.config,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by
        }


class IConfigStore(ABC):
    """Interface for configuration persistence."""
    
    @abstractmethod
    async def load_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Load tenant configuration."""
        pass
    
    @abstractmethod
    async def save_config(self, config: TenantConfig) -> bool:
        """Save tenant configuration."""
        pass
    
    @abstractmethod
    async def delete_config(self, tenant_id: str) -> bool:
        """Delete tenant configuration."""
        pass
    
    @abstractmethod
    async def list_configs(self) -> List[str]:
        """List all tenant configuration IDs."""
        pass


class MemoryConfigStore(IConfigStore):
    """In-memory configuration store for development."""
    
    def __init__(self):
        self.configs: Dict[str, TenantConfig] = {}
    
    async def load_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Load tenant configuration."""
        return self.configs.get(tenant_id)
    
    async def save_config(self, config: TenantConfig) -> bool:
        """Save tenant configuration."""
        self.configs[config.tenant_id] = config
        return True
    
    async def delete_config(self, tenant_id: str) -> bool:
        """Delete tenant configuration."""
        if tenant_id in self.configs:
            del self.configs[tenant_id]
            return True
        return False
    
    async def list_configs(self) -> List[str]:
        """List all tenant configuration IDs."""
        return list(self.configs.keys())


class ConfigEncryption:
    """Configuration encryption utility."""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.key = master_key.encode()
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, value: str) -> str:
        """Encrypt configuration value."""
        encrypted = self.cipher.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt configuration value."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()


class TenantConfigManager:
    """
    Central configuration management for multi-tenant system.
    
    PHASE3-REFACTOR: Hierarchical configuration with validation and encryption.
    """
    
    def __init__(self, config_store: Optional[IConfigStore] = None, 
                 master_key: Optional[str] = None):
        self.config_store = config_store or MemoryConfigStore()
        self.encryption = ConfigEncryption(master_key)
        
        # Configuration schemas
        self.schemas: Dict[str, ConfigSchema] = {}
        
        # Default configurations by tier
        self.tier_configs: Dict[TenantTier, Dict[str, Any]] = {}
        
        # Global default configuration
        self.global_config: Dict[str, Any] = {}
        
        # Initialize default schemas and configurations
        self._initialize_default_schemas()
        self._initialize_default_configs()
    
    def _initialize_default_schemas(self):
        """Initialize default configuration schemas."""
        default_schemas = [
            # LLM Configuration
            ConfigSchema("llm.provider", ConfigType.STRING, True, "openai", 
                        "LLM provider name", allowed_values=["openai", "anthropic", "azure", "local"]),
            ConfigSchema("llm.model", ConfigType.STRING, True, "gpt-3.5-turbo", 
                        "LLM model name"),
            ConfigSchema("llm.api_key", ConfigType.SECRET, True, "", 
                        "LLM API key (encrypted)"),
            ConfigSchema("llm.temperature", ConfigType.FLOAT, False, 0.7, 
                        "LLM temperature", min_value=0.0, max_value=2.0),
            ConfigSchema("llm.max_tokens", ConfigType.INTEGER, False, 2048, 
                        "Maximum tokens per request", min_value=1, max_value=32768),
            
            # Retriever Configuration
            ConfigSchema("retriever.provider", ConfigType.STRING, True, "pinecone", 
                        "Vector store provider", allowed_values=["pinecone", "weaviate", "chroma", "qdrant"]),
            ConfigSchema("retriever.index_name", ConfigType.STRING, True, "", 
                        "Vector store index name"),
            ConfigSchema("retriever.api_key", ConfigType.SECRET, False, "", 
                        "Vector store API key (encrypted)"),
            ConfigSchema("retriever.top_k", ConfigType.INTEGER, False, 5, 
                        "Number of retrieved documents", min_value=1, max_value=50),
            ConfigSchema("retriever.similarity_threshold", ConfigType.FLOAT, False, 0.7, 
                        "Similarity threshold", min_value=0.0, max_value=1.0),
            
            # Channel Configuration
            ConfigSchema("channels.web.enabled", ConfigType.BOOLEAN, False, True, 
                        "Enable web channel"),
            ConfigSchema("channels.whatsapp.enabled", ConfigType.BOOLEAN, False, False, 
                        "Enable WhatsApp channel"),
            ConfigSchema("channels.voice.enabled", ConfigType.BOOLEAN, False, False, 
                        "Enable voice channel"),
            
            # Rate Limiting
            ConfigSchema("rate_limit.requests_per_hour", ConfigType.INTEGER, False, 1000, 
                        "Requests per hour limit", min_value=1),
            ConfigSchema("rate_limit.tokens_per_hour", ConfigType.INTEGER, False, 50000, 
                        "Tokens per hour limit", min_value=1),
            
            # Analytics
            ConfigSchema("analytics.enabled", ConfigType.BOOLEAN, False, True, 
                        "Enable analytics collection"),
            ConfigSchema("analytics.retention_days", ConfigType.INTEGER, False, 30, 
                        "Analytics data retention", min_value=1, max_value=365),
            
            # Security
            ConfigSchema("security.encryption_enabled", ConfigType.BOOLEAN, False, True, 
                        "Enable data encryption"),
            ConfigSchema("security.session_timeout", ConfigType.INTEGER, False, 3600, 
                        "Session timeout in seconds", min_value=300, max_value=86400),
        ]
        
        for schema in default_schemas:
            self.schemas[schema.key] = schema
    
    def _initialize_default_configs(self):
        """Initialize default configurations by tier."""
        # Global defaults
        self.global_config = {
            "llm.provider": "openai",
            "llm.model": "gpt-3.5-turbo",
            "llm.temperature": 0.7,
            "llm.max_tokens": 2048,
            "retriever.provider": "pinecone",
            "retriever.top_k": 5,
            "retriever.similarity_threshold": 0.7,
            "channels.web.enabled": True,
            "channels.whatsapp.enabled": False,
            "channels.voice.enabled": False,
            "analytics.enabled": True,
            "analytics.retention_days": 30,
            "security.encryption_enabled": True,
            "security.session_timeout": 3600
        }
        
        # Tier-specific configurations
        self.tier_configs = {
            TenantTier.FREE: {
                "rate_limit.requests_per_hour": 100,
                "rate_limit.tokens_per_hour": 5000,
                "llm.model": "gpt-3.5-turbo",
                "retriever.top_k": 3,
                "analytics.retention_days": 7
            },
            TenantTier.BASIC: {
                "rate_limit.requests_per_hour": 1000,
                "rate_limit.tokens_per_hour": 25000,
                "llm.model": "gpt-3.5-turbo",
                "retriever.top_k": 5,
                "analytics.retention_days": 30,
                "channels.whatsapp.enabled": True
            },
            TenantTier.PROFESSIONAL: {
                "rate_limit.requests_per_hour": 5000,
                "rate_limit.tokens_per_hour": 100000,
                "llm.model": "gpt-4",
                "retriever.top_k": 10,
                "analytics.retention_days": 90,
                "channels.whatsapp.enabled": True,
                "channels.voice.enabled": True
            },
            TenantTier.ENTERPRISE: {
                "rate_limit.requests_per_hour": 50000,
                "rate_limit.tokens_per_hour": 1000000,
                "llm.model": "gpt-4",
                "retriever.top_k": 20,
                "analytics.retention_days": 365,
                "channels.whatsapp.enabled": True,
                "channels.voice.enabled": True
            }
        }
    
    def register_schema(self, schema: ConfigSchema):
        """Register configuration schema."""
        self.schemas[schema.key] = schema
        logger.info(f"Registered config schema: {schema.key}")
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schemas."""
        errors = []
        
        for key, value in config.items():
            if key not in self.schemas:
                errors.append(f"Unknown configuration key: {key}")
                continue
            
            schema = self.schemas[key]
            if not schema.validate(value):
                errors.append(f"Invalid value for {key}: {value}")
        
        # Check required fields
        for key, schema in self.schemas.items():
            if schema.required and key not in config:
                errors.append(f"Required configuration missing: {key}")
        
        return errors
    
    async def get_tenant_config(self, tenant: Tenant) -> TenantConfig:
        """Get complete tenant configuration with inheritance."""
        # Load tenant-specific config
        tenant_config = await self.config_store.load_config(tenant.id)
        if not tenant_config:
            tenant_config = TenantConfig(tenant_id=tenant.id)
        
        # Build inherited configuration
        inherited_config = self._build_inherited_config(tenant)
        
        # Merge with tenant-specific overrides
        final_config = inherited_config.copy()
        final_config.update(tenant_config.config)
        
        tenant_config.config = final_config
        return tenant_config
    
    def _build_inherited_config(self, tenant: Tenant) -> Dict[str, Any]:
        """Build configuration with inheritance chain."""
        config = {}
        
        # Start with global defaults
        config.update(self.global_config)
        
        # Apply tier-specific overrides
        tier_config = self.tier_configs.get(tenant.tier, {})
        config.update(tier_config)
        
        return config
    
    async def update_tenant_config(self, tenant_id: str, updates: Dict[str, Any], 
                                 updated_by: str = "") -> bool:
        """Update tenant configuration."""
        # Validate updates
        errors = self.validate_config(updates)
        if errors:
            logger.error(f"Configuration validation errors for tenant {tenant_id}: {errors}")
            return False
        
        # Load existing config
        config = await self.config_store.load_config(tenant_id)
        if not config:
            config = TenantConfig(tenant_id=tenant_id)
        
        # Create version before update
        config.create_version("Pre-update backup", updated_by)
        
        # Apply updates
        for key, value in updates.items():
            # Encrypt secrets
            if key in self.schemas and self.schemas[key].config_type == ConfigType.SECRET:
                config.encrypted_config[key] = self.encryption.encrypt(str(value))
            else:
                config.set_value(key, value, updated_by)
        
        # Save updated config
        success = await self.config_store.save_config(config)
        if success:
            logger.info(f"Updated configuration for tenant {tenant_id}")
        
        return success
    
    async def get_config_value(self, tenant: Tenant, key: str, default: Any = None) -> Any:
        """Get specific configuration value for tenant."""
        config = await self.get_tenant_config(tenant)
        
        # Check if it's an encrypted value
        if key in config.encrypted_config:
            try:
                return self.encryption.decrypt(config.encrypted_config[key])
            except Exception as e:
                logger.error(f"Failed to decrypt config {key} for tenant {tenant.id}: {e}")
                return default
        
        return config.get_value(key, default)
    
    async def delete_tenant_config(self, tenant_id: str) -> bool:
        """Delete tenant configuration."""
        success = await self.config_store.delete_config(tenant_id)
        if success:
            logger.info(f"Deleted configuration for tenant {tenant_id}")
        return success
    
    async def export_config(self, tenant_id: str, format: str = "yaml") -> str:
        """Export tenant configuration."""
        config = await self.config_store.load_config(tenant_id)
        if not config:
            return ""
        
        export_data = {
            "tenant_id": config.tenant_id,
            "version": config.version,
            "config": config.config,
            "updated_at": config.updated_at.isoformat()
        }
        
        if format.lower() == "yaml":
            return yaml.dump(export_data, default_flow_style=False)
        else:
            return json.dumps(export_data, indent=2)
    
    async def import_config(self, tenant_id: str, config_data: str, 
                          format: str = "yaml", updated_by: str = "") -> bool:
        """Import tenant configuration."""
        try:
            if format.lower() == "yaml":
                data = yaml.safe_load(config_data)
            else:
                data = json.loads(config_data)
            
            config_values = data.get("config", {})
            return await self.update_tenant_config(tenant_id, config_values, updated_by)
        
        except Exception as e:
            logger.error(f"Failed to import config for tenant {tenant_id}: {e}")
            return False
    
    async def get_config_history(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get configuration version history."""
        config = await self.config_store.load_config(tenant_id)
        if not config:
            return []
        
        return [
            {
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "created_by": v.created_by,
                "description": v.description
            }
            for v in config.versions
        ]
    
    async def rollback_config(self, tenant_id: str, version: int, 
                            updated_by: str = "") -> bool:
        """Rollback configuration to specific version."""
        config = await self.config_store.load_config(tenant_id)
        if not config:
            return False
        
        success = config.rollback_to_version(version, updated_by)
        if success:
            await self.config_store.save_config(config)
        
        return success


# Global configuration manager
_config_manager: Optional[TenantConfigManager] = None


def get_config_manager() -> TenantConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = TenantConfigManager()
    return _config_manager


async def initialize_config_manager(config_store: Optional[IConfigStore] = None,
                                  master_key: Optional[str] = None):
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = TenantConfigManager(config_store, master_key)
    logger.info("Global configuration manager initialized")


# Configuration helpers
async def get_tenant_config_value(tenant: Tenant, key: str, default: Any = None) -> Any:
    """Helper to get tenant configuration value."""
    config_manager = get_config_manager()
    return await config_manager.get_config_value(tenant, key, default)
