"""
Database Factory
===============

Factory pattern implementation for dynamic database backend loading.
Supports configuration-driven backend selection and initialization.

Supported Backends:
- Firestore (Google Cloud)
- PostgreSQL (with async support)
- MongoDB (future implementation)
- Redis (for caching)
- SQLite (for development/testing)

Usage:
    factory = DatabaseFactory()
    database = await factory.create_database(config)
"""

import asyncio
from typing import Dict, Any, Type, Optional
import logging
from pathlib import Path

from .base import BaseDatabase, DatabaseConfig, DatabaseType, DatabaseError
from .encryption import EncryptionManager
from .cache import CacheManager, CacheConfig
from .audit import AuditLogger, AuditConfig

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """
    Factory for creating database instances based on configuration.
    
    Implements the factory pattern to abstract database creation
    and provide consistent initialization across different backends.
    """
    
    # Registry of available database implementations
    _database_registry: Dict[DatabaseType, Type[BaseDatabase]] = {}
    _initialized = False
    
    @classmethod
    def register_database(
        cls,
        database_type: DatabaseType,
        implementation_class: Type[BaseDatabase]
    ) -> None:
        """
        Register a database implementation.
        
        Args:
            database_type: Database type enum
            implementation_class: Database implementation class
        """
        cls._database_registry[database_type] = implementation_class
        logger.info(f"Registered database implementation: {database_type.value}")
    
    @classmethod
    def _initialize_registry(cls) -> None:
        """Initialize the database registry with available implementations."""
        if cls._initialized:
            return
        
        # Import and register available implementations
        try:
            from .firestore_impl import FirestoreDatabase
            cls.register_database(DatabaseType.FIRESTORE, FirestoreDatabase)
        except ImportError as e:
            logger.warning(f"Firestore implementation not available: {e}")
        
        try:
            from .postgresql_impl import PostgreSQLDatabase
            cls.register_database(DatabaseType.POSTGRESQL, PostgreSQLDatabase)
        except ImportError as e:
            logger.warning(f"PostgreSQL implementation not available: {e}")
        
        # Future implementations
        # try:
        #     from .mongodb_impl import MongoDBDatabase
        #     cls.register_database(DatabaseType.MONGODB, MongoDBDatabase)
        # except ImportError:
        #     logger.info("MongoDB implementation not available")
        
        # try:
        #     from .sqlite_impl import SQLiteDatabase
        #     cls.register_database(DatabaseType.SQLITE, SQLiteDatabase)
        # except ImportError:
        #     logger.info("SQLite implementation not available")
        
        cls._initialized = True
        logger.info(f"Database factory initialized with {len(cls._database_registry)} implementations")
    
    def __init__(self):
        """Initialize the database factory."""
        self._initialize_registry()
        self._encryption_manager = None
        self._cache_manager = None
        self._audit_logger = None
    
    async def create_database(
        self,
        config: DatabaseConfig,
        encryption_manager: Optional[EncryptionManager] = None,
        cache_manager: Optional[CacheManager] = None,
        audit_logger: Optional[AuditLogger] = None
    ) -> BaseDatabase:
        """
        Create a database instance based on configuration.
        
        Args:
            config: Database configuration
            encryption_manager: Optional encryption manager
            cache_manager: Optional cache manager
            audit_logger: Optional audit logger
            
        Returns:
            Configured database instance
            
        Raises:
            DatabaseError: If backend is not supported or initialization fails
        """
        # Validate configuration
        await self._validate_config(config)
        
        # Get implementation class
        if config.backend not in self._database_registry:
            available_backends = list(self._database_registry.keys())
            raise DatabaseError(
                f"Unsupported database backend: {config.backend.value}. "
                f"Available backends: {[b.value for b in available_backends]}"
            )
        
        implementation_class = self._database_registry[config.backend]
        
        try:
            # Create database instance
            database = implementation_class(config)
            
            # Inject optional components
            if hasattr(database, 'set_encryption_manager') and encryption_manager:
                database.set_encryption_manager(encryption_manager)
            
            if hasattr(database, 'set_cache_manager') and cache_manager:
                database.set_cache_manager(cache_manager)
            
            if hasattr(database, 'set_audit_logger') and audit_logger:
                database.set_audit_logger(audit_logger)
            
            # Initialize connection
            await database.connect()
            
            # Perform health check
            health_status = await database.health_check()
            if health_status.get('status') != 'healthy':
                logger.warning(f"Database health check warning: {health_status}")
            
            logger.info(f"Successfully created {config.backend.value} database instance")
            return database
            
        except Exception as e:
            logger.error(f"Failed to create database instance: {e}")
            raise DatabaseError(f"Database initialization failed: {e}", original_error=e)
    
    async def create_complete_database_stack(
        self,
        database_config: DatabaseConfig,
        cache_config: Optional[CacheConfig] = None,
        audit_config: Optional[AuditConfig] = None,
        encryption_key: Optional[str] = None
    ) -> BaseDatabase:
        """
        Create a complete database stack with all components.
        
        Args:
            database_config: Database configuration
            cache_config: Cache configuration
            audit_config: Audit configuration
            encryption_key: Encryption key
            
        Returns:
            Fully configured database instance with all components
        """
        components = {}
        
        try:
            # Create encryption manager
            if database_config.encryption_level.value != "none":
                encryption_manager = EncryptionManager(
                    encryption_key=encryption_key or database_config.encryption_key,
                    encryption_level=database_config.encryption_level
                )
                components['encryption'] = encryption_manager
                logger.info("Created encryption manager")
            
            # Create cache manager
            if database_config.enable_caching and cache_config:
                cache_manager = CacheManager(cache_config)
                await cache_manager.connect()
                components['cache'] = cache_manager
                logger.info("Created and connected cache manager")
            
            # Create audit logger
            if database_config.enable_audit and audit_config:
                audit_logger = AuditLogger(audit_config)
                components['audit'] = audit_logger
                logger.info("Created audit logger")
            
            # Create database with components
            database = await self.create_database(
                config=database_config,
                encryption_manager=components.get('encryption'),
                cache_manager=components.get('cache'),
                audit_logger=components.get('audit')
            )
            
            logger.info("Successfully created complete database stack")
            return database
            
        except Exception as e:
            # Cleanup any created components
            await self._cleanup_components(components)
            raise DatabaseError(f"Failed to create database stack: {e}", original_error=e)
    
    async def create_from_config_file(
        self,
        config_file_path: str,
        config_section: str = "database"
    ) -> BaseDatabase:
        """
        Create database from configuration file.
        
        Args:
            config_file_path: Path to configuration file
            config_section: Configuration section name
            
        Returns:
            Configured database instance
        """
        try:
            config_data = await self._load_config_file(config_file_path, config_section)
            
            # Create database config
            database_config = DatabaseConfig(**config_data.get('database', {}))
            
            # Create component configs
            cache_config = None
            if config_data.get('cache'):
                cache_config = CacheConfig(**config_data['cache'])
            
            audit_config = None
            if config_data.get('audit'):
                audit_config = AuditConfig(**config_data['audit'])
            
            # Get encryption key
            encryption_key = config_data.get('encryption_key')
            
            return await self.create_complete_database_stack(
                database_config=database_config,
                cache_config=cache_config,
                audit_config=audit_config,
                encryption_key=encryption_key
            )
            
        except Exception as e:
            logger.error(f"Failed to create database from config file: {e}")
            raise DatabaseError(f"Config file loading failed: {e}", original_error=e)
    
    async def _validate_config(self, config: DatabaseConfig) -> None:
        """
        Validate database configuration.
        
        Args:
            config: Database configuration to validate
            
        Raises:
            DatabaseError: If configuration is invalid
        """
        # Basic validation
        if not config.database_name:
            raise DatabaseError("Database name is required")
        
        # Backend-specific validation
        if config.backend == DatabaseType.POSTGRESQL:
            if not config.host:
                raise DatabaseError("PostgreSQL host is required")
            if not config.username:
                raise DatabaseError("PostgreSQL username is required")
        
        elif config.backend == DatabaseType.FIRESTORE:
            # Firestore uses service account credentials
            if not config.backend_config.get('service_account_path') and \
               not config.backend_config.get('service_account_json'):
                logger.warning("No Firestore service account configured")
        
        # Security validation
        if config.use_ssl and config.backend in [DatabaseType.POSTGRESQL]:
            logger.info("SSL/TLS encryption enabled for database connection")
        
        # Performance validation
        if config.connection_pool_size > 50:
            logger.warning(f"Large connection pool size: {config.connection_pool_size}")
    
    async def _load_config_file(
        self,
        config_file_path: str,
        config_section: str
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file_path: Path to configuration file
            config_section: Configuration section
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_file_path)
        
        if not config_path.exists():
            raise DatabaseError(f"Configuration file not found: {config_file_path}")
        
        try:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                import yaml
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
            else:
                raise DatabaseError(f"Unsupported config file format: {config_path.suffix}")
            
            # Extract section
            if config_section in full_config:
                return full_config[config_section]
            else:
                logger.warning(f"Config section '{config_section}' not found, using full config")
                return full_config
                
        except Exception as e:
            raise DatabaseError(f"Failed to load config file: {e}", original_error=e)
    
    async def _cleanup_components(self, components: Dict[str, Any]) -> None:
        """
        Cleanup created components in case of initialization failure.
        
        Args:
            components: Dictionary of created components
        """
        for name, component in components.items():
            try:
                if hasattr(component, 'disconnect'):
                    await component.disconnect()
                elif hasattr(component, 'close'):
                    await component.close()
                logger.debug(f"Cleaned up component: {name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup component {name}: {e}")
    
    def get_available_backends(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available database backends.
        
        Returns:
            Dictionary of backend information
        """
        backends = {}
        
        for db_type, implementation_class in self._database_registry.items():
            backends[db_type.value] = {
                'name': db_type.value,
                'class': implementation_class.__name__,
                'module': implementation_class.__module__,
                'description': implementation_class.__doc__ or "No description available"
            }
        
        return backends
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics and status."""
        return {
            'initialized': self._initialized,
            'registered_backends': len(self._database_registry),
            'available_backends': list(self._database_registry.keys()),
            'factory_class': self.__class__.__name__
        }
    
    async def create_complete_stack(
        self,
        config: DatabaseConfig,
        encryption_key: Optional[str] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        audit_config: Optional[Dict[str, Any]] = None
    ) -> BaseDatabase:
        """
        Create a complete database stack with all components.
        
        Args:
            config: Database configuration
            encryption_key: Encryption key
            cache_config: Cache configuration
            audit_config: Audit configuration
            
        Returns:
            Fully configured database instance with all components
        """
        # Convert dict configs to proper config objects if needed
        cache_cfg = CacheConfig(**cache_config) if cache_config else None
        audit_cfg = AuditConfig(**audit_config) if audit_config else None
        
        return await self.create_complete_database_stack(
            database_config=config,
            cache_config=cache_cfg,
            audit_config=audit_cfg,
            encryption_key=encryption_key
        )
    
    def load_config(self, config_file: str, config_section: str = "database") -> DatabaseConfig:
        """
        Load database configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            config_section: Section name in the config file
            
        Returns:
            Database configuration object
        """
        import asyncio
        
        # Run the async method in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            config_data = loop.run_until_complete(
                self._load_config_file(config_file, config_section)
            )
            return DatabaseConfig(**config_data)
        finally:
            loop.close()
