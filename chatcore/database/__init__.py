"""
Database Module for Chatbot Framework
====================================

A comprehensive, enterprise-grade database abstraction layer supporting
multiple backends with built-in security, caching, and audit logging.

Key Features:
- Multi-backend support (Firestore, PostgreSQL, extensible)
- Field-level and record-level encryption
- Redis-based intelligent caching
- Comprehensive audit logging
- Connection pooling and async operations
- Type-safe interfaces

Quick Start:
    from database import DatabaseFactory, DatabaseConfig
    
    config = DatabaseConfig(backend="postgresql", database_name="chatcore")
    factory = DatabaseFactory()
    database = factory.create_database(config)
    
    await database.connect()
    user_id = await database.create("users", {"name": "John", "email": "john@example.com"})
    user = await database.get("users", user_id)

Classes:
    DatabaseConfig: Configuration for database connections
    BaseDatabase: Abstract base class for database implementations
    DatabaseFactory: Factory for creating database instances
    EncryptionManager: Data encryption and decryption
    CacheManager: Redis-based caching
    AuditLogger: Operation and security event logging
    QueryResult: Query result container
    Various exceptions for error handling
"""

__version__ = "1.0.0"
__author__ = "Chatbot Framework Team"

from typing import Dict, Any, Optional

# Core interfaces and configuration
from .base import (
    BaseDatabase,
    DatabaseConfig,
    DatabaseType,
    DatabaseRecord,
    QueryResult,
    QueryFilter,
    OperationType,
    DatabaseError,
    ConnectionError,
    QueryError,
    ValidationError
)

# Factory for creating database instances
from .factory import DatabaseFactory

# Security and caching components
from .encryption import EncryptionManager
from .cache import CacheManager
from .audit import AuditLogger, AuditEvent

# Database implementations
try:
    from .firestore_impl import FirestoreDatabase
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

try:
    from .postgresql_impl import PostgreSQLDatabase
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

# Export all public classes and functions
__all__ = [
    # Core interfaces
    "BaseDatabase",
    "DatabaseConfig", 
    "DatabaseRecord",
    "QueryResult",
    "QueryFilter",
    "OperationType",
    
    # Factory
    "DatabaseFactory",
    
    # Components
    "EncryptionManager",
    "CacheManager", 
    "AuditLogger",
    "AuditEvent",
    
    # Exceptions
    "DatabaseError",
    "ConnectionError", 
    "QueryError",
    "ValidationError",
    
    # Implementations (if available)
    *([FirestoreDatabase] if FIRESTORE_AVAILABLE else []),
    *([PostgreSQLDatabase] if POSTGRESQL_AVAILABLE else []),
    
    # Version and availability
    "__version__",
    "FIRESTORE_AVAILABLE",
    "POSTGRESQL_AVAILABLE"
]

# Convenience functions for quick setup

def create_database(backend: str, **kwargs) -> BaseDatabase:
    """
    Convenience function to create a database instance.
    
    Args:
        backend: Database backend ("firestore" or "postgresql")
        **kwargs: Database configuration parameters
        
    Returns:
        Configured database instance
        
    Example:
        database = create_database(
            backend="postgresql",
            host="localhost", 
            database_name="chatcore",
            username="user",
            password="pass"
        )
    """
    import asyncio
    
    # Convert string backend to DatabaseType
    backend_type = DatabaseType(backend)
    
    config = DatabaseConfig(backend=backend_type, **kwargs)
    factory = DatabaseFactory()
    
    # Run async operation in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(factory.create_database(config))
    finally:
        loop.close()

def create_complete_stack(
    backend: str,
    encryption_key: str,
    cache_config: Optional[Dict[str, Any]] = None,
    audit_config: Optional[Dict[str, Any]] = None,
    **db_kwargs
) -> BaseDatabase:
    """
    Convenience function to create a complete database stack with all components.
    
    Args:
        backend: Database backend ("firestore" or "postgresql") 
        encryption_key: 32-character encryption key
        cache_config: Redis cache configuration
        audit_config: Audit logging configuration
        **db_kwargs: Database configuration parameters
        
    Returns:
        Fully configured database instance with encryption, caching, and auditing
        
    Example:
        database = create_complete_stack(
            backend="postgresql",
            encryption_key="my-32-character-encryption-key",
            cache_config={"host": "localhost", "port": 6379},
            audit_config={"log_level": "INFO"},
            host="localhost",
            database_name="chatcore"
        )
    """
    import asyncio
    
    # Convert string backend to DatabaseType
    backend_type = DatabaseType(backend)
    
    config = DatabaseConfig(backend=backend_type, **db_kwargs)
    factory = DatabaseFactory()
    
    # Run async operation in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(factory.create_complete_stack(
            config, 
            encryption_key=encryption_key,
            cache_config=cache_config or {},
            audit_config=audit_config or {}
        ))
    finally:
        loop.close()

def load_config_and_create(config_file: str, **overrides) -> BaseDatabase:
    """
    Load configuration from file and create database instance.
    
    Args:
        config_file: Path to YAML configuration file
        **overrides: Configuration overrides
        
    Returns:
        Configured database instance
        
    Example:
        database = load_config_and_create(
            "config/database.yaml",
            backend="postgresql"  # Override backend
        )
    """
    import asyncio
    
    factory = DatabaseFactory()
    config = factory.load_config(config_file)
    
    # Apply overrides
    for key, value in overrides.items():
        setattr(config, key, value)
    
    # Run async operations in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(factory.create_database(config))
    finally:
        loop.close()

# Module-level constants
DEFAULT_CACHE_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "default_ttl": 300
}

DEFAULT_AUDIT_CONFIG = {
    "log_level": "INFO",
    "enable_console": True,
    "enable_file": False
}

# Backend availability check
def check_backend_availability():
    """
    Check which database backends are available.
    
    Returns:
        Dict with backend availability status
    """
    return {
        "firestore": FIRESTORE_AVAILABLE,
        "postgresql": POSTGRESQL_AVAILABLE
    }

def list_available_backends():
    """
    Get list of available database backends.
    
    Returns:
        List of available backend names
    """
    backends = []
    if FIRESTORE_AVAILABLE:
        backends.append("firestore")
    if POSTGRESQL_AVAILABLE:
        backends.append("postgresql")
    return backends

# Version information
def get_version_info():
    """
    Get detailed version and availability information.
    
    Returns:
        Dict with version and backend availability
    """
    return {
        "version": __version__,
        "backends": check_backend_availability(),
        "features": {
            "encryption": True,
            "caching": True, 
            "auditing": True,
            "async_operations": True,
            "connection_pooling": True
        }
    }
