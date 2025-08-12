"""
Base classes and interfaces for the database module.

This module provides the abstract base classes and core types
for the chatcore database abstraction layer.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Type
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None, **kwargs):
        super().__init__(message)
        self.original_error = original_error
        self.metadata = kwargs


class ConnectionError(DatabaseError):
    """Exception raised for database connection issues."""
    pass


class QueryError(DatabaseError):
    """Exception raised for query execution issues."""
    pass


class ValidationError(DatabaseError):
    """Exception raised for data validation issues."""
    pass


class ConfigurationError(DatabaseError):
    """Exception raised for configuration issues."""
    pass


class DatabaseType(Enum):
    """Supported database types."""
    FIRESTORE = "firestore"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    SQLITE = "sqlite"
    REDIS = "redis"


class EncryptionLevel(Enum):
    """Encryption level for database operations."""
    NONE = "none"
    FIELD = "field"
    DOCUMENT = "document"
    COLLECTION = "collection"


class CacheStrategy(Enum):
    """Cache strategy for database operations."""
    NONE = "none"
    READ_THROUGH = "read_through"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"


class OperationType(Enum):
    """Database operation types for auditing."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    BATCH_CREATE = "batch_create"
    BATCH_UPDATE = "batch_update"
    BATCH_DELETE = "batch_delete"
    QUERY = "query"
    AGGREGATE = "aggregate"


@dataclass
class QueryFilter:
    """Query filter specification."""
    field: str
    operator: str  # "==", "!=", ">", ">=", "<", "<=", "in", "not_in", "array_contains"
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value
        }


@dataclass
class DatabaseRecord:
    """Database record with metadata."""
    id: str
    data: Dict[str, Any]
    collection: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        result = {
            "id": self.id,
            "data": self.data,
            "collection": self.collection,
            "metadata": self.metadata
        }
        
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        if self.version is not None:
            result["version"] = self.version
            
        return result


@dataclass
class DatabaseConfig:
    """Configuration for database instances."""
    backend: DatabaseType
    host: str = "localhost"
    port: Optional[int] = None
    database_name: str = "chatcore"
    username: Optional[str] = None
    password: Optional[str] = None
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60
    ssl_enabled: bool = False
    use_ssl: bool = False  # Alias for ssl_enabled
    ssl_cert_path: Optional[str] = None
    encryption_level: EncryptionLevel = EncryptionLevel.NONE
    encryption_key: Optional[str] = None
    enable_caching: bool = False
    cache_strategy: CacheStrategy = CacheStrategy.READ_THROUGH
    cache_ttl: int = 3600
    enable_audit: bool = False
    audit_level: str = "INFO"
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_options: Dict[str, Any] = field(default_factory=dict)
    backend_config: Dict[str, Any] = field(default_factory=dict)  # Backend-specific configuration
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Sync ssl flags
        if self.ssl_enabled and not self.use_ssl:
            self.use_ssl = True
        elif self.use_ssl and not self.ssl_enabled:
            self.ssl_enabled = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                data[key] = value.value
            elif isinstance(value, (str, int, float, bool, type(None))):
                data[key] = value
            elif isinstance(value, dict):
                data[key] = value.copy()
            else:
                data[key] = str(value)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "backend" in data and isinstance(data["backend"], str):
            data["backend"] = DatabaseType(data["backend"])
        if "encryption_level" in data and isinstance(data["encryption_level"], str):
            data["encryption_level"] = EncryptionLevel(data["encryption_level"])
        if "cache_strategy" in data and isinstance(data["cache_strategy"], str):
            data["cache_strategy"] = CacheStrategy(data["cache_strategy"])
        
        return cls(**data)


@dataclass
class CacheConfig:
    """Configuration for cache operations."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    key_prefix: str = "chatcore:"
    default_ttl: int = 3600
    max_ttl: int = 86400
    compression_enabled: bool = True
    serialization_format: str = "json"


@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    log_rotation: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_sensitive_data: bool = False
    audit_operations: List[str] = field(default_factory=lambda: ["create", "update", "delete"])
    performance_tracking: bool = True


@dataclass
class QueryResult:
    """Result from a database query."""
    data: List[Dict[str, Any]]
    count: int
    execution_time: float
    cache_hit: bool = False
    total_count: Optional[int] = None  # Total count without limit/offset
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return self.count
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def first(self) -> Optional[Dict[str, Any]]:
        """Get the first result."""
        return self.data[0] if self.data else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "data": self.data,
            "count": self.count,
            "execution_time": self.execution_time,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata
        }
        
        if self.total_count is not None:
            result["total_count"] = self.total_count
            
        return result


@dataclass
class HealthStatus:
    """Health status of database connection."""
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if the database is healthy."""
        return self.status == "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            "status": self.status,
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "errors": self.errors
        }


class BaseDatabase(ABC):
    """
    Abstract base class for database implementations.
    
    This class defines the interface that all database implementations
    must follow, providing a consistent API across different backends.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database instance.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._connected = False
        self._encryption_manager = None
        self._cache_manager = None
        self._audit_logger = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def set_encryption_manager(self, encryption_manager) -> None:
        """Set the encryption manager for this database instance."""
        self._encryption_manager = encryption_manager
        self.logger.debug("Encryption manager set")
    
    def set_cache_manager(self, cache_manager) -> None:
        """Set the cache manager for this database instance."""
        self._cache_manager = cache_manager
        self.logger.debug("Cache manager set")
    
    def set_audit_logger(self, audit_logger) -> None:
        """Set the audit logger for this database instance."""
        self._audit_logger = audit_logger
        self.logger.debug("Audit logger set")
    
    @property
    def is_connected(self) -> bool:
        """Check if the database is connected."""
        return self._connected
    
    @property
    def database_type(self) -> DatabaseType:
        """Get the database type."""
        return self.config.backend
    
    # Connection Management
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the database.
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the database connection.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database.
        
        Returns:
            Health status information
        """
        pass
    
    # Basic CRUD Operations
    @abstractmethod
    async def create(
        self,
        collection: str,
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Create a new document.
        
        Args:
            collection: Collection/table name
            data: Document data
            **kwargs: Additional options
            
        Returns:
            Document ID
            
        Raises:
            QueryError: If creation fails
        """
        pass
    
    @abstractmethod
    async def read(
        self,
        collection: str,
        document_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs
    ) -> QueryResult:
        """
        Read documents from the database.
        
        Args:
            collection: Collection/table name
            document_id: Specific document ID
            filters: Filter conditions
            limit: Maximum number of results
            offset: Number of results to skip
            **kwargs: Additional options
            
        Returns:
            Query results
            
        Raises:
            QueryError: If read fails
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        collection: str,
        document_id: str,
        data: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Update a document.
        
        Args:
            collection: Collection/table name
            document_id: Document ID
            data: Updated data
            **kwargs: Additional options
            
        Returns:
            True if successful
            
        Raises:
            QueryError: If update fails
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        collection: str,
        document_id: str,
        **kwargs
    ) -> bool:
        """
        Delete a document.
        
        Args:
            collection: Collection/table name
            document_id: Document ID
            **kwargs: Additional options
            
        Returns:
            True if successful
            
        Raises:
            QueryError: If deletion fails
        """
        pass
    
    # Batch Operations
    @abstractmethod
    async def batch_create(
        self,
        collection: str,
        data: List[Dict[str, Any]],
        **kwargs
    ) -> List[str]:
        """
        Create multiple documents in batch.
        
        Args:
            collection: Collection/table name
            data: List of document data
            **kwargs: Additional options
            
        Returns:
            List of document IDs
            
        Raises:
            QueryError: If batch creation fails
        """
        pass
    
    @abstractmethod
    async def batch_update(
        self,
        collection: str,
        updates: List[Dict[str, Any]],
        **kwargs
    ) -> int:
        """
        Update multiple documents in batch.
        
        Args:
            collection: Collection/table name
            updates: List of update operations
            **kwargs: Additional options
            
        Returns:
            Number of updated documents
            
        Raises:
            QueryError: If batch update fails
        """
        pass
    
    @abstractmethod
    async def batch_delete(
        self,
        collection: str,
        document_ids: List[str],
        **kwargs
    ) -> int:
        """
        Delete multiple documents in batch.
        
        Args:
            collection: Collection/table name
            document_ids: List of document IDs
            **kwargs: Additional options
            
        Returns:
            Number of deleted documents
            
        Raises:
            QueryError: If batch deletion fails
        """
        pass
    
    # Query Operations
    @abstractmethod
    async def find(
        self,
        collection: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs
    ) -> QueryResult:
        """
        Find documents matching query criteria.
        
        Args:
            collection: Collection/table name
            query: Query conditions
            projection: Fields to include/exclude
            sort: Sort specification
            limit: Maximum number of results
            offset: Number of results to skip
            **kwargs: Additional options
            
        Returns:
            Query results
            
        Raises:
            QueryError: If query fails
        """
        pass
    
    @abstractmethod
    async def count(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Count documents matching query criteria.
        
        Args:
            collection: Collection/table name
            query: Query conditions
            **kwargs: Additional options
            
        Returns:
            Document count
            
        Raises:
            QueryError: If count fails
        """
        pass
    
    @abstractmethod
    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]],
        **kwargs
    ) -> QueryResult:
        """
        Perform aggregation operations.
        
        Args:
            collection: Collection/table name
            pipeline: Aggregation pipeline
            **kwargs: Additional options
            
        Returns:
            Aggregation results
            
        Raises:
            QueryError: If aggregation fails
        """
        pass
    
    # Utility Methods
    async def backup(
        self,
        backup_path: Union[str, Path],
        collections: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to save backup
            collections: Specific collections to backup
            **kwargs: Additional options
            
        Returns:
            True if successful
            
        Raises:
            DatabaseError: If backup fails
        """
        self.logger.warning("Backup not implemented for this database type")
        return False
    
    async def restore(
        self,
        backup_path: Union[str, Path],
        **kwargs
    ) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            **kwargs: Additional options
            
        Returns:
            True if successful
            
        Raises:
            DatabaseError: If restore fails
        """
        self.logger.warning("Restore not implemented for this database type")
        return False
    
    def _encrypt_data(self, data: Any) -> Any:
        """Encrypt data if encryption manager is available."""
        if self._encryption_manager and data is not None:
            return self._encryption_manager.encrypt(data)
        return data
    
    def _decrypt_data(self, data: Any) -> Any:
        """Decrypt data if encryption manager is available."""
        if self._encryption_manager and data is not None:
            return self._encryption_manager.decrypt(data)
        return data
    
    async def _cache_get(self, key: str) -> Optional[Any]:
        """Get data from cache if cache manager is available."""
        if self._cache_manager:
            return await self._cache_manager.get(key)
        return None
    
    async def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set data in cache if cache manager is available."""
        if self._cache_manager:
            await self._cache_manager.set(key, value, ttl=ttl)
    
    async def _cache_delete(self, key: str) -> None:
        """Delete data from cache if cache manager is available."""
        if self._cache_manager:
            await self._cache_manager.delete(key)
    
    def _audit_log(self, operation: str, collection: str, **kwargs) -> None:
        """Log operation if audit logger is available."""
        if self._audit_logger:
            self._audit_logger.log_operation(
                operation=operation,
                collection=collection,
                database_type=self.database_type.value,
                **kwargs
            )
    
    def validate_collection_name(self, collection: str) -> None:
        """Validate collection/table name."""
        if not collection:
            raise ValidationError("Collection name cannot be empty")
        
        if not isinstance(collection, str):
            raise ValidationError("Collection name must be a string")
        
        # Check for invalid characters (basic validation)
        if any(c in collection for c in ['/', '\\', '..', '\x00']):
            raise ValidationError(f"Collection name contains invalid characters: {collection}")
        
        # Check length
        if len(collection) > 100:
            raise ValidationError(f"Collection name too long: {len(collection)} characters")
    
    def validate_document_id(self, document_id: str) -> None:
        """Validate document ID."""
        if not document_id:
            raise ValidationError("Document ID cannot be empty")
        
        if not isinstance(document_id, str):
            raise ValidationError("Document ID must be a string")
        
        # Check for invalid characters
        if any(c in document_id for c in ['/', '\\', '..', '\x00']):
            raise ValidationError(f"Document ID contains invalid characters: {document_id}")
        
        # Check length
        if len(document_id) > 200:
            raise ValidationError(f"Document ID too long: {len(document_id)} characters")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.config.backend.value}, connected={self._connected})"
    
    def __str__(self) -> str:
        return f"{self.config.backend.value} database at {self.config.host}:{self.config.port or 'default'}"


# Type aliases for convenience
DatabaseInstance = BaseDatabase
ConfigDict = Dict[str, Any]
QueryCondition = Dict[str, Any]
UpdateOperation = Dict[str, Any]
