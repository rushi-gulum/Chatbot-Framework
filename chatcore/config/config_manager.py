"""
Enterprise Configuration Manager
===============================

Centralized configuration management with environment-based settings,
validation, hot-reloading, and structured logging support.

REFACTORED: Complete configuration system for enterprise deployments.
"""

import os
import yaml
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from functools import lru_cache

# REFACTORED: Optional hot-reload capability
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HOT_RELOAD_AVAILABLE = True
except ImportError:
    HOT_RELOAD_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

# REFACTORED: Added structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatcore.log')
    ]
)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentMode(Enum):
    """Deployment modes for scaling configuration."""
    SINGLE_INSTANCE = "single"
    MULTI_INSTANCE = "multi"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_authentication: bool = True
    enable_rate_limiting: bool = True
    jwt_secret_key: str = ""
    jwt_expiry_hours: int = 24
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000
    encryption_key: str = ""
    ssl_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    def __post_init__(self):
        """Validate security configuration."""
        if self.enable_authentication and not self.jwt_secret_key:
            raise ValueError("JWT secret key is required when authentication is enabled")
        if not self.encryption_key:
            self.encryption_key = os.environ.get("ENCRYPTION_KEY", "")


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    backend: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database_name: str = "chatcore"
    username: str = ""
    password: str = ""
    connection_pool_size: int = 10
    query_timeout: float = 30.0
    use_ssl: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    def __post_init__(self):
        """Resolve environment variables for database config."""
        self.username = self.username or os.environ.get("DB_USERNAME", "")
        self.password = self.password or os.environ.get("DB_PASSWORD", "")


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    default_provider: str = "openai"
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 30.0
    retry_attempts: int = 3
    enable_caching: bool = True
    
    def __post_init__(self):
        """Set default providers if not specified."""
        if not self.providers:
            self.providers = {
                "openai": {
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "model": "gpt-3.5-turbo",
                    "max_tokens": self.max_tokens
                },
                "anthropic": {
                    "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": self.max_tokens
                }
            }


@dataclass
class RetrieverConfig:
    """Retriever configuration settings."""
    embedder_type: str = "openai"
    vector_store_type: str = "faiss"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 10
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    cache_embeddings: bool = True
    
    embedder_config: Dict[str, Any] = field(default_factory=dict)
    vector_store_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityConfig:
    """Observability and monitoring configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_structured_logging: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "json"
    retention_days: int = 30
    
    # External integrations
    prometheus_enabled: bool = False
    jaeger_enabled: bool = False
    elastic_enabled: bool = False
    
    # Endpoints
    prometheus_endpoint: str = "/metrics"
    health_endpoint: str = "/health"


@dataclass
class ScalingConfig:
    """Horizontal scaling configuration."""
    deployment_mode: DeploymentMode = DeploymentMode.SINGLE_INSTANCE
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    enable_auto_scaling: bool = False
    load_balancer_enabled: bool = False


@dataclass
class Settings:
    """
    Comprehensive application settings with environment-specific configuration.
    
    REFACTORED: Centralized settings management with validation and type safety.
    """
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "ChatCore"
    version: str = "2.0.0"
    debug_mode: bool = False
    
    # Core configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    
    # Channel configurations
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT


# Configuration file handler with conditional watchdog support
if HOT_RELOAD_AVAILABLE and FileSystemEventHandler:
    class WatchdogConfigHandler(FileSystemEventHandler):
        """File system event handler for configuration hot-reloading."""
        
        def __init__(self, config_manager):
            super().__init__()
            self.config_manager = config_manager
            
        def on_modified(self, event):
            """Handle configuration file modifications."""
            if not event.is_directory and event.src_path.endswith('.yaml'):
                logger.info(f"Configuration file modified: {event.src_path}")
                asyncio.create_task(self.config_manager.reload_config())
    
    # Alias for consistent usage
    ConfigFileHandler = WatchdogConfigHandler
else:
    class ConfigFileHandler:
        """Dummy file handler when watchdog is not available."""
        
        def __init__(self, config_manager):
            self.config_manager = config_manager


class ConfigManager:
    """
    Enterprise configuration manager with hot-reloading and validation.
    
    REFACTORED: Complete configuration management system for scalable deployments.
    """
    
    def __init__(self, config_path: Optional[str] = None, enable_hot_reload: bool = False):
        self.config_path = config_path or self._find_config_path()
        self.enable_hot_reload = enable_hot_reload and HOT_RELOAD_AVAILABLE
        self._settings: Optional[Settings] = None
        self._observer = None
        
        # Load initial configuration
        self.load_config()
        
        # Setup hot-reloading if enabled and available
        if self.enable_hot_reload:
            self._setup_hot_reload()
    
    def _find_config_path(self) -> str:
        """Find configuration file path based on environment."""
        env = os.environ.get("ENVIRONMENT", "development")
        config_dir = Path(__file__).parent
        
        # Try environment-specific config first
        env_config = config_dir / f"settings.{env}.yaml"
        if env_config.exists():
            return str(env_config)
        
        # Fall back to default config
        default_config = config_dir / "settings.yaml"
        if default_config.exists():
            return str(default_config)
        
        raise FileNotFoundError("No configuration file found")
    
    def load_config(self) -> Settings:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Merge environment variables
            config_data = self._merge_environment_variables(config_data)
            
            # Create Settings object
            self._settings = self._create_settings_from_dict(config_data)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._settings
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def reload_config(self) -> Settings:
        """Asynchronously reload configuration."""
        try:
            return self.load_config()
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            # Return current settings if reload fails
            if self._settings is None:
                raise RuntimeError("No valid configuration available")
            return self._settings
    
    def _merge_environment_variables(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables with configuration data."""
        # REFACTORED: Support for environment variable overrides
        env_overrides = {
            'database.username': 'DB_USERNAME',
            'database.password': 'DB_PASSWORD',
            'database.host': 'DB_HOST',
            'database.port': 'DB_PORT',
            'security.jwt_secret_key': 'JWT_SECRET_KEY',
            'security.encryption_key': 'ENCRYPTION_KEY',
            'llm.providers.openai.api_key': 'OPENAI_API_KEY',
            'llm.providers.anthropic.api_key': 'ANTHROPIC_API_KEY',
        }
        
        for config_path, env_var in env_overrides.items():
            env_value = os.environ.get(env_var)
            if env_value:
                self._set_nested_value(config_data, config_path, env_value)
        
        return config_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _create_settings_from_dict(self, config_data: Dict[str, Any]) -> Settings:
        """Create Settings object from configuration dictionary."""
        # REFACTORED: Type-safe configuration creation
        settings_dict = {}
        
        # Basic settings
        settings_dict['environment'] = Environment(config_data.get('environment', 'development'))
        settings_dict['app_name'] = config_data.get('app_name', 'ChatCore')
        settings_dict['version'] = config_data.get('version', '2.0.0')
        settings_dict['debug_mode'] = config_data.get('debug_mode', False)
        
        # Security configuration
        if 'security' in config_data:
            settings_dict['security'] = SecurityConfig(**config_data['security'])
        
        # Database configuration
        if 'database' in config_data:
            settings_dict['database'] = DatabaseConfig(**config_data['database'])
        
        # LLM configuration
        if 'llm' in config_data:
            settings_dict['llm'] = LLMConfig(**config_data['llm'])
        
        # Retriever configuration
        if 'retriever' in config_data:
            settings_dict['retriever'] = RetrieverConfig(**config_data['retriever'])
        
        # Observability configuration
        if 'observability' in config_data:
            settings_dict['observability'] = ObservabilityConfig(**config_data['observability'])
        
        # Scaling configuration
        if 'scaling' in config_data:
            scaling_data = config_data['scaling'].copy()
            if 'deployment_mode' in scaling_data:
                scaling_data['deployment_mode'] = DeploymentMode(scaling_data['deployment_mode'])
            settings_dict['scaling'] = ScalingConfig(**scaling_data)
        
        # Channel configurations
        settings_dict['channels'] = config_data.get('channels', {})
        
        return Settings(**settings_dict)
    
    def _setup_hot_reload(self):
        """Setup file system monitoring for hot-reloading."""
        if not HOT_RELOAD_AVAILABLE or Observer is None or self._observer:
            return
        
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        config_dir = Path(self.config_path).parent
        
        self._observer.schedule(handler, str(config_dir), recursive=False)
        self._observer.start()
        
        logger.info("Configuration hot-reloading enabled")
    
    @property
    def settings(self) -> Settings:
        """Get current settings."""
        if not self._settings:
            self.load_config()
        if self._settings is None:
            raise RuntimeError("Failed to load configuration")
        return self._settings
    
    def get_channel_config(self, channel_name: str) -> Dict[str, Any]:
        """Get configuration for specific channel."""
        return self.settings.channels.get(channel_name, {})
    
    def get_llm_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for specific LLM provider."""
        return self.settings.llm.providers.get(provider_name, {})
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._observer:
            self._observer.stop()
            self._observer.join()


# REFACTORED: Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_settings() -> Settings:
    """Get current application settings."""
    return get_config_manager().settings
