"""
Configuration Validation System
==============================

Comprehensive validation for configuration settings with detailed error reporting
and security checks.

REFACTORED: Added robust validation system for configuration integrity.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import re
import ipaddress
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Validation error details."""
    field_path: str
    message: str
    severity: str = "error"  # error, warning, info
    suggested_value: Optional[Any] = None


class ValidationResult:
    """Result of configuration validation."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.is_valid = True
    
    def add_error(self, field_path: str, message: str, suggested_value: Optional[Any] = None):
        """Add validation error."""
        self.errors.append(ValidationError(field_path, message, "error", suggested_value))
        self.is_valid = False
    
    def add_warning(self, field_path: str, message: str):
        """Add validation warning."""
        self.warnings.append(ValidationError(field_path, message, "warning"))
    
    def get_summary(self) -> str:
        """Get validation summary."""
        if self.is_valid:
            return f"Configuration valid. {len(self.warnings)} warnings."
        return f"Configuration invalid. {len(self.errors)} errors, {len(self.warnings)} warnings."


class ConfigValidator:
    """
    Comprehensive configuration validator.
    
    REFACTORED: Enterprise-grade validation with security checks.
    """
    
    @staticmethod
    def validate_port(port: Union[int, str], field_path: str, result: ValidationResult):
        """Validate port number."""
        try:
            port_num = int(port)
            if not (1 <= port_num <= 65535):
                result.add_error(field_path, f"Port {port_num} is out of valid range (1-65535)")
            elif port_num < 1024:
                result.add_warning(field_path, f"Port {port_num} is privileged (< 1024)")
        except (ValueError, TypeError):
            result.add_error(field_path, f"Invalid port value: {port}")
    
    @staticmethod
    def validate_url(url: str, field_path: str, result: ValidationResult):
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            result.add_error(field_path, f"Invalid URL format: {url}")
        elif not url.startswith('https://'):
            result.add_warning(field_path, f"Non-HTTPS URL detected: {url}")
    
    @staticmethod
    def validate_ip_address(ip: str, field_path: str, result: ValidationResult):
        """Validate IP address."""
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            result.add_error(field_path, f"Invalid IP address: {ip}")
    
    @staticmethod
    def validate_file_path(path: str, field_path: str, result: ValidationResult, must_exist: bool = False):
        """Validate file path."""
        try:
            file_path = Path(path)
            if must_exist and not file_path.exists():
                result.add_error(field_path, f"File does not exist: {path}")
            elif not file_path.parent.exists():
                result.add_error(field_path, f"Parent directory does not exist: {path}")
        except Exception as e:
            result.add_error(field_path, f"Invalid file path {path}: {str(e)}")
    
    @staticmethod
    def validate_api_key(api_key: str, field_path: str, result: ValidationResult, provider: str = ""):
        """Validate API key format."""
        if not api_key:
            result.add_error(field_path, "API key is required")
            return
        
        if api_key in ["", "your_api_key", "sk-xxxxxxx"]:
            result.add_error(field_path, "Placeholder API key detected")
            return
        
        # Provider-specific validation
        if provider.lower() == "openai":
            if not api_key.startswith("sk-"):
                result.add_error(field_path, "OpenAI API key should start with 'sk-'")
            elif len(api_key) < 50:
                result.add_error(field_path, "OpenAI API key seems too short")
        
        elif provider.lower() == "anthropic":
            if not api_key.startswith("sk-ant-"):
                result.add_error(field_path, "Anthropic API key should start with 'sk-ant-'")
    
    @staticmethod
    def validate_security_config(config: Dict[str, Any], result: ValidationResult):
        """Validate security configuration."""
        prefix = "security"
        
        # JWT secret validation
        jwt_secret = config.get("jwt_secret_key", "")
        if config.get("enable_authentication", True):
            if not jwt_secret:
                result.add_error(f"{prefix}.jwt_secret_key", "JWT secret key is required when authentication is enabled")
            elif len(jwt_secret) < 32:
                result.add_error(f"{prefix}.jwt_secret_key", "JWT secret key should be at least 32 characters")
        
        # Encryption key validation
        encryption_key = config.get("encryption_key", "")
        if encryption_key and len(encryption_key) != 32:
            result.add_error(f"{prefix}.encryption_key", "Encryption key must be exactly 32 characters")
        
        # Rate limiting validation
        max_per_minute = config.get("max_requests_per_minute", 100)
        max_per_hour = config.get("max_requests_per_hour", 1000)
        
        if max_per_minute <= 0:
            result.add_error(f"{prefix}.max_requests_per_minute", "Must be positive")
        if max_per_hour <= 0:
            result.add_error(f"{prefix}.max_requests_per_hour", "Must be positive")
        if max_per_hour < max_per_minute * 60:
            result.add_warning(f"{prefix}.max_requests_per_hour", "Hourly limit seems low compared to per-minute limit")
        
        # CORS validation
        cors_origins = config.get("cors_origins", [])
        if "*" in cors_origins and len(cors_origins) > 1:
            result.add_warning(f"{prefix}.cors_origins", "Wildcard CORS with specific origins may cause issues")
    
    @staticmethod
    def validate_database_config(config: Dict[str, Any], result: ValidationResult):
        """Validate database configuration."""
        prefix = "database"
        
        # Connection parameters
        host = config.get("host", "")
        if not host:
            result.add_error(f"{prefix}.host", "Database host is required")
        elif host not in ["localhost", "127.0.0.1"]:
            ConfigValidator.validate_ip_address(host, f"{prefix}.host", result)
        
        # Port validation
        port = config.get("port", 5432)
        ConfigValidator.validate_port(port, f"{prefix}.port", result)
        
        # Credentials
        username = config.get("username", "")
        password = config.get("password", "")
        
        if not username:
            result.add_error(f"{prefix}.username", "Database username is required")
        if not password:
            result.add_error(f"{prefix}.password", "Database password is required")
        elif len(password) < 8:
            result.add_warning(f"{prefix}.password", "Database password should be at least 8 characters")
        
        # Connection pool
        pool_size = config.get("connection_pool_size", 10)
        if not isinstance(pool_size, int) or pool_size <= 0:
            result.add_error(f"{prefix}.connection_pool_size", "Connection pool size must be positive integer")
        elif pool_size > 100:
            result.add_warning(f"{prefix}.connection_pool_size", "Large connection pool may impact performance")
        
        # SSL validation
        if not config.get("use_ssl", True):
            result.add_warning(f"{prefix}.use_ssl", "SSL is disabled - consider enabling for production")
    
    @staticmethod
    def validate_llm_config(config: Dict[str, Any], result: ValidationResult):
        """Validate LLM configuration."""
        prefix = "llm"
        
        # Default provider
        default_provider = config.get("default_provider", "")
        if not default_provider:
            result.add_error(f"{prefix}.default_provider", "Default LLM provider is required")
        
        # Provider configurations
        providers = config.get("providers", {})
        if not providers:
            result.add_error(f"{prefix}.providers", "At least one LLM provider must be configured")
        
        for provider_name, provider_config in providers.items():
            provider_prefix = f"{prefix}.providers.{provider_name}"
            
            # API key validation
            api_key = provider_config.get("api_key", "")
            ConfigValidator.validate_api_key(api_key, f"{provider_prefix}.api_key", result, provider_name)
            
            # Model validation
            model = provider_config.get("model", "")
            if not model:
                result.add_error(f"{provider_prefix}.model", "Model name is required")
            
            # Token limits
            max_tokens = provider_config.get("max_tokens", 2048)
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                result.add_error(f"{provider_prefix}.max_tokens", "Max tokens must be positive integer")
            elif max_tokens > 32000:
                result.add_warning(f"{provider_prefix}.max_tokens", "Very high token limit may be expensive")
    
    @staticmethod
    def validate_retriever_config(config: Dict[str, Any], result: ValidationResult):
        """Validate retriever configuration."""
        prefix = "retriever"
        
        # Embedder type
        embedder_type = config.get("embedder_type", "")
        valid_embedders = ["openai", "huggingface", "cohere", "sentence_transformers"]
        if embedder_type not in valid_embedders:
            result.add_error(f"{prefix}.embedder_type", f"Invalid embedder type. Must be one of: {valid_embedders}")
        
        # Vector store type
        vector_store_type = config.get("vector_store_type", "")
        valid_stores = ["faiss", "pinecone", "chroma", "milvus", "weaviate"]
        if vector_store_type not in valid_stores:
            result.add_error(f"{prefix}.vector_store_type", f"Invalid vector store type. Must be one of: {valid_stores}")
        
        # Chunk configuration
        chunk_size = config.get("chunk_size", 1000)
        chunk_overlap = config.get("chunk_overlap", 100)
        
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            result.add_error(f"{prefix}.chunk_size", "Chunk size must be positive integer")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            result.add_error(f"{prefix}.chunk_overlap", "Chunk overlap must be non-negative integer")
        if chunk_overlap >= chunk_size:
            result.add_error(f"{prefix}.chunk_overlap", "Chunk overlap must be less than chunk size")
        
        # Search parameters
        top_k = config.get("top_k", 10)
        similarity_threshold = config.get("similarity_threshold", 0.7)
        
        if not isinstance(top_k, int) or top_k <= 0:
            result.add_error(f"{prefix}.top_k", "Top K must be positive integer")
        elif top_k > 100:
            result.add_warning(f"{prefix}.top_k", "High top_k value may impact performance")
        
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            result.add_error(f"{prefix}.similarity_threshold", "Similarity threshold must be between 0.0 and 1.0")
    
    @staticmethod
    def validate_observability_config(config: Dict[str, Any], result: ValidationResult):
        """Validate observability configuration."""
        prefix = "observability"
        
        # Metrics port
        metrics_port = config.get("metrics_port", 9090)
        ConfigValidator.validate_port(metrics_port, f"{prefix}.metrics_port", result)
        
        # Log level
        log_level = config.get("log_level", "INFO")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            result.add_error(f"{prefix}.log_level", f"Invalid log level. Must be one of: {valid_levels}")
        
        # Retention days
        retention_days = config.get("retention_days", 30)
        if not isinstance(retention_days, int) or retention_days <= 0:
            result.add_error(f"{prefix}.retention_days", "Retention days must be positive integer")
        elif retention_days > 365:
            result.add_warning(f"{prefix}.retention_days", "Long retention period may consume significant storage")
    
    @classmethod
    def validate_settings(cls, settings_dict: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete settings configuration.
        
        REFACTORED: Comprehensive validation with detailed error reporting.
        """
        result = ValidationResult()
        
        # Environment validation
        environment = settings_dict.get("environment", "")
        valid_environments = ["development", "staging", "production", "testing"]
        if environment not in valid_environments:
            result.add_error("environment", f"Invalid environment. Must be one of: {valid_environments}")
        
        # Component validations
        if "security" in settings_dict:
            cls.validate_security_config(settings_dict["security"], result)
        
        if "database" in settings_dict:
            cls.validate_database_config(settings_dict["database"], result)
        
        if "llm" in settings_dict:
            cls.validate_llm_config(settings_dict["llm"], result)
        
        if "retriever" in settings_dict:
            cls.validate_retriever_config(settings_dict["retriever"], result)
        
        if "observability" in settings_dict:
            cls.validate_observability_config(settings_dict["observability"], result)
        
        # Production-specific validations
        if environment == "production":
            cls._validate_production_requirements(settings_dict, result)
        
        logger.info(f"Configuration validation completed: {result.get_summary()}")
        return result
    
    @staticmethod
    def _validate_production_requirements(settings_dict: Dict[str, Any], result: ValidationResult):
        """Validate production-specific requirements."""
        # Debug mode should be disabled
        if settings_dict.get("debug_mode", False):
            result.add_error("debug_mode", "Debug mode must be disabled in production")
        
        # SSL should be enabled
        database_config = settings_dict.get("database", {})
        if not database_config.get("use_ssl", True):
            result.add_error("database.use_ssl", "SSL must be enabled in production")
        
        # Security configurations
        security_config = settings_dict.get("security", {})
        if not security_config.get("enable_authentication", True):
            result.add_error("security.enable_authentication", "Authentication must be enabled in production")
        
        if not security_config.get("ssl_enabled", True):
            result.add_error("security.ssl_enabled", "SSL must be enabled in production")
        
        # CORS should not allow all origins
        cors_origins = security_config.get("cors_origins", [])
        if "*" in cors_origins:
            result.add_error("security.cors_origins", "Wildcard CORS origins not recommended in production")
