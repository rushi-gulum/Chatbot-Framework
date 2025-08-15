"""
Configuration Management Module
===============================

REFACTORED: Enterprise-grade configuration system with security and validation.

This module provides:
- Centralized configuration management
- Environment-specific configuration loading
- Configuration validation and security
- Dependency injection container
- Secret management with encryption
"""

from .config_manager import (
    Settings, ConfigManager, Environment, DeploymentMode,
    SecurityConfig, DatabaseConfig, LLMConfig, RetrieverConfig,
    ObservabilityConfig, ScalingConfig, ChannelConfig,
    load_config, get_config_manager, reload_config
)

from .validation import (
    ConfigValidator, ValidationError, ValidationResult,
    validate_config, get_validation_errors
)

from .security import (
    SecretManager, EnvironmentSecrets,
    encrypt_config_secrets, decrypt_config_secrets,
    get_safe_config_for_logging, get_secret_manager
)

from .dependency_injection import (
    DIContainer, ServiceDescriptor, ServiceLifetime, ServiceScope,
    Injectable, get_container, configure_services, injectable
)

__version__ = "2.0.0"  # REFACTORED: Upgraded architecture version

__all__ = [
    # Configuration Management
    "Settings", "ConfigManager", "Environment", "DeploymentMode",
    "SecurityConfig", "DatabaseConfig", "LLMConfig", "RetrieverConfig",
    "ObservabilityConfig", "ScalingConfig", "ChannelConfig",
    "load_config", "get_config_manager", "reload_config",
    
    # Validation
    "ConfigValidator", "ValidationError", "ValidationResult",
    "validate_config", "get_validation_errors",
    
    # Security
    "SecretManager", "EnvironmentSecrets",
    "encrypt_config_secrets", "decrypt_config_secrets",
    "get_safe_config_for_logging", "get_secret_manager",
    
    # Dependency Injection
    "DIContainer", "ServiceDescriptor", "ServiceLifetime", "ServiceScope",
    "Injectable", "get_container", "configure_services", "injectable"
]
