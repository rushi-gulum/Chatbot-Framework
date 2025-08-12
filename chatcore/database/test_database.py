"""
Unit Tests for Database Module
=============================

Comprehensive test suite for the database abstraction layer including:
- BaseDatabase interface compliance
- Encryption functionality
- Cache integration
- Audit logging
- Multiple backend implementations
- Error handling and recovery

Test Structure:
- Unit tests for individual components
- Integration tests for full stack
- Mock tests for external dependencies
- Performance and load testing helpers
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Test imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.base import (
    BaseDatabase, DatabaseConfig, DatabaseRecord, QueryResult, QueryFilter,
    OperationType, DatabaseError, ConnectionError, QueryError, ValidationError
)
from database.encryption import EncryptionManager
from database.cache import CacheManager
from database.audit import AuditLogger, AuditEvent
from database.factory import DatabaseFactory


class TestDatabaseConfig:
    """Test database configuration validation and creation."""
    
    def test_config_creation_minimal(self):
        """Test creating config with minimal required fields."""
        config = DatabaseConfig(
            backend="firestore",
            database_name="test_db"
        )
        
        assert config.backend == "firestore"
        assert config.database_name == "test_db"
        assert config.host is None
        assert config.port is None
        assert config.connection_pool_size == 10
        assert config.query_timeout == 30.0
        assert config.use_ssl is True
    
    def test_config_creation_full(self):
        """Test creating config with all fields."""
        backend_config = {
            "project_id": "test-project",
            "credentials_path": "/path/to/creds.json"
        }
        
        config = DatabaseConfig(
            backend="postgresql",
            host="localhost",
            port=5432,
            database_name="chatcore_test",
            username="test_user",
            password="test_pass",
            connection_pool_size=20,
            query_timeout=60.0,
            use_ssl=False,
            ssl_cert_path="/path/to/cert.pem",
            backend_config=backend_config
        )
        
        assert config.backend == "postgresql"
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database_name == "chatcore_test"
        assert config.username == "test_user"
        assert config.password == "test_pass"
        assert config.connection_pool_size == 20
        assert config.query_timeout == 60.0
        assert config.use_ssl is False
        assert config.backend_config == backend_config
    
    def test_config_validation_invalid_backend(self):
        """Test config validation with invalid backend."""
        with pytest.raises(ValidationError):
            DatabaseConfig(
                backend="invalid_backend",
                database_name="test_db"
            )
    
    def test_config_validation_missing_database_name(self):
        """Test config validation with missing database name."""
        with pytest.raises(ValidationError):
            DatabaseConfig(backend="firestore")


class TestEncryptionManager:
    """Test encryption and decryption functionality."""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager for testing."""
        return EncryptionManager(
            encryption_key="test_key_32_chars_long_12345678",
            sensitive_fields=["password", "ssn", "credit_card"]
        )
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt_data(self, encryption_manager):
        """Test encrypting and decrypting data."""
        test_data = {
            "id": "user123",
            "name": "John Doe",
            "password": "secret123",
            "email": "john@example.com"
        }
        
        # Encrypt data
        encrypted = await encryption_manager.encrypt_data(test_data.copy())
        
        # Verify sensitive field is encrypted
        assert encrypted["password"] != "secret123"
        assert encrypted["name"] == "John Doe"  # Non-sensitive unchanged
        assert encrypted["email"] == "john@example.com"
        
        # Decrypt data
        decrypted = await encryption_manager.decrypt_data(encrypted)
        
        # Verify decryption
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_encrypt_record_level(self, encryption_manager):
        """Test record-level encryption."""
        test_data = {
            "id": "doc123",
            "content": "sensitive document content",
            "metadata": {"type": "confidential"}
        }
        
        encrypted = await encryption_manager.encrypt_record(test_data)
        
        # Verify entire record is encrypted
        assert "_encrypted_data" in encrypted
        assert "content" not in encrypted
        assert encrypted["id"] == "doc123"  # ID preserved
        
        # Decrypt record
        decrypted = await encryption_manager.decrypt_record(encrypted)
        assert decrypted == test_data
    
    def test_detect_sensitive_fields(self, encryption_manager):
        """Test sensitive field detection."""
        test_data = {
            "name": "John",
            "password": "secret",
            "user_ssn": "123-45-6789",
            "email": "john@example.com"
        }
        
        sensitive = encryption_manager._detect_sensitive_fields(test_data)
        
        assert "password" in sensitive
        assert "user_ssn" in sensitive  # Contains 'ssn'
        assert "name" not in sensitive
        assert "email" not in sensitive
    
    @pytest.mark.asyncio
    async def test_encryption_with_none_values(self, encryption_manager):
        """Test encryption handles None values correctly."""
        test_data = {
            "id": "test123",
            "password": None,
            "name": "John Doe"
        }
        
        encrypted = await encryption_manager.encrypt_data(test_data.copy())
        decrypted = await encryption_manager.decrypt_data(encrypted)
        
        assert decrypted == test_data


class TestCacheManager:
    """Test cache functionality."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager for testing."""
        config = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "default_ttl": 300
        }
        
        # Mock Redis for testing
        mock_redis = AsyncMock()
        
        with patch('database.cache.aioredis.from_url', return_value=mock_redis):
            cache = CacheManager(config)
            await cache.connect()
            cache._redis = mock_redis  # Override with mock
            return cache
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager):
        """Test basic cache set and get operations."""
        test_data = {"id": "test123", "name": "Test User"}
        
        # Mock Redis responses
        cache_manager._redis.set.return_value = True
        cache_manager._redis.get.return_value = json.dumps(test_data)
        
        # Test set
        await cache_manager.set("test_key", test_data, ttl=300)
        cache_manager._redis.set.assert_called_once()
        
        # Test get
        result = await cache_manager.get("test_key")
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_cache_record_key_building(self, cache_manager):
        """Test record cache key building."""
        key = cache_manager.build_record_key("users", "user123")
        assert key == "record:users:user123"
    
    @pytest.mark.asyncio
    async def test_cache_query_key_building(self, cache_manager):
        """Test query cache key building."""
        filters = {"status": "active", "age": {">=": 18}}
        key = cache_manager.build_query_key(
            "users", filters, "created_at", 10, 0
        )
        
        assert "query:users:" in key
        assert key == cache_manager.build_query_key(
            "users", filters, "created_at", 10, 0
        )  # Consistent hashing
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation patterns."""
        cache_manager._redis.delete.return_value = 1
        
        # Test single key deletion
        await cache_manager.delete("test_key")
        cache_manager._redis.delete.assert_called_with("test_key")
        
        # Test pattern invalidation
        cache_manager._redis.eval.return_value = 5
        result = await cache_manager.invalidate_pattern("user:*")
        
        assert result == 5
        cache_manager._redis.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_metrics(self, cache_manager):
        """Test cache metrics collection."""
        # Mock hit
        cache_manager._redis.get.return_value = '{"test": "data"}'
        await cache_manager.get("test_key")
        
        # Mock miss
        cache_manager._redis.get.return_value = None
        await cache_manager.get("missing_key")
        
        metrics = cache_manager.get_metrics()
        
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == 0.5


class TestAuditLogger:
    """Test audit logging functionality."""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for testing."""
        return AuditLogger(
            log_level="INFO",
            enable_console=False,
            enable_file=False
        )
    
    @pytest.mark.asyncio
    async def test_log_operation(self, audit_logger):
        """Test operation logging."""
        with patch.object(audit_logger._logger, 'info') as mock_log:
            trace_id = await audit_logger.log_operation(
                operation=OperationType.CREATE,
                collection="users",
                document_id="user123",
                execution_time_ms=150.5,
                records_affected=1,
                success=True
            )
            
            assert trace_id is not None
            mock_log.assert_called_once()
            
            # Verify log message contains expected fields
            call_args = mock_log.call_args[0][0]
            assert "CREATE" in call_args
            assert "users" in call_args
            assert "user123" in call_args
    
    @pytest.mark.asyncio
    async def test_log_query(self, audit_logger):
        """Test query logging."""
        filters = {"status": "active"}
        
        with patch.object(audit_logger._logger, 'info') as mock_log:
            trace_id = await audit_logger.log_query(
                collection="users",
                filters=filters,
                result_count=25,
                execution_time_ms=75.2,
                success=True
            )
            
            assert trace_id is not None
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_logger):
        """Test security event logging."""
        with patch.object(audit_logger._logger, 'warning') as mock_log:
            await audit_logger.log_security_event(
                event_type="unauthorized_access",
                severity="HIGH",
                details={"user_ip": "192.168.1.100", "attempted_action": "admin_access"}
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "unauthorized_access" in call_args
            assert "HIGH" in call_args
    
    def test_create_audit_event(self, audit_logger):
        """Test audit event creation."""
        event = audit_logger._create_audit_event(
            operation=OperationType.UPDATE,
            collection="products",
            success=True,
            execution_time_ms=200.0
        )
        
        assert isinstance(event, AuditEvent)
        assert event.operation == OperationType.UPDATE
        assert event.collection == "products"
        assert event.success is True
        assert event.execution_time_ms == 200.0
        assert event.timestamp is not None
        assert event.trace_id is not None


class TestDatabaseFactory:
    """Test database factory functionality."""
    
    def test_factory_registration(self):
        """Test backend registration."""
        factory = DatabaseFactory()
        
        # Mock database class
        class MockDatabase:
            def __init__(self, config):
                self.config = config
        
        factory.register_backend("mock", MockDatabase)
        
        assert "mock" in factory._backends
        assert factory._backends["mock"] == MockDatabase
    
    def test_factory_create_database(self):
        """Test database creation through factory."""
        factory = DatabaseFactory()
        
        # Mock database class
        class MockDatabase:
            def __init__(self, config):
                self.config = config
        
        factory.register_backend("mock", MockDatabase)
        
        config = DatabaseConfig(backend="mock", database_name="test")
        database = factory.create_database(config)
        
        assert isinstance(database, MockDatabase)
        assert database.config == config
    
    def test_factory_create_invalid_backend(self):
        """Test factory with invalid backend."""
        factory = DatabaseFactory()
        config = DatabaseConfig(backend="invalid", database_name="test")
        
        with pytest.raises(ValueError, match="Unknown database backend"):
            factory.create_database(config)
    
    @patch('database.factory.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_factory_load_config(self, mock_file, mock_yaml):
        """Test loading configuration from file."""
        mock_yaml.return_value = {
            "database": {
                "backend": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database_name": "test_db",
                "username": "test_user",
                "password": "test_pass"
            }
        }
        
        factory = DatabaseFactory()
        config = factory.load_config("test_config.yaml")
        
        assert config.backend == "postgresql"
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database_name == "test_db"
    
    def test_factory_create_complete_stack(self):
        """Test creating complete database stack."""
        factory = DatabaseFactory()
        
        # Mock all components
        with patch('database.factory.EncryptionManager') as mock_encryption, \
             patch('database.factory.CacheManager') as mock_cache, \
             patch('database.factory.AuditLogger') as mock_audit:
            
            mock_encryption.return_value = Mock()
            mock_cache.return_value = Mock()
            mock_audit.return_value = Mock()
            
            config = DatabaseConfig(backend="firestore", database_name="test")
            
            # Mock database class
            class MockDatabase:
                def __init__(self, config):
                    self.config = config
                    self.set_encryption_manager = Mock()
                    self.set_cache_manager = Mock()
                    self.set_audit_logger = Mock()
            
            factory.register_backend("firestore", MockDatabase)
            
            database = factory.create_complete_stack(
                config,
                encryption_key="test_key",
                cache_config={"host": "localhost"},
                audit_config={"log_level": "INFO"}
            )
            
            assert isinstance(database, MockDatabase)
            database.set_encryption_manager.assert_called_once()
            database.set_cache_manager.assert_called_once()
            database.set_audit_logger.assert_called_once()


class MockDatabaseImplementation(BaseDatabase):
    """Mock database implementation for testing base functionality."""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._data = {}  # In-memory storage
        self._connected = False
    
    async def connect(self) -> None:
        self._connected = True
        self._is_connected = True
    
    async def disconnect(self) -> None:
        self._connected = False
        self._is_connected = False
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            'status': 'healthy' if self._connected else 'disconnected',
            'mock_backend': True
        }
    
    async def create(self, collection: str, data: DatabaseRecord, document_id: Optional[str] = None) -> str:
        self.validate_collection_name(collection)
        if document_id:
            self.validate_document_id(document_id)
        
        doc_id = document_id or f"doc_{len(self._data)}"
        
        if collection not in self._data:
            self._data[collection] = {}
        
        self._data[collection][doc_id] = data.copy()
        return doc_id
    
    async def get(self, collection: str, document_id: str) -> Optional[DatabaseRecord]:
        self.validate_collection_name(collection)
        self.validate_document_id(document_id)
        
        if collection not in self._data:
            return None
        
        return self._data[collection].get(document_id)
    
    async def update(self, collection: str, document_id: str, data: DatabaseRecord, upsert: bool = False) -> bool:
        self.validate_collection_name(collection)
        self.validate_document_id(document_id)
        
        if collection not in self._data:
            if upsert:
                self._data[collection] = {}
            else:
                return False
        
        if document_id not in self._data[collection]:
            if upsert:
                self._data[collection][document_id] = data.copy()
                return True
            else:
                return False
        
        self._data[collection][document_id].update(data)
        return True
    
    async def delete(self, collection: str, document_id: str) -> bool:
        self.validate_collection_name(collection)
        self.validate_document_id(document_id)
        
        if collection not in self._data or document_id not in self._data[collection]:
            return False
        
        del self._data[collection][document_id]
        return True
    
    async def query(self, collection: str, filters: Optional[QueryFilter] = None, **kwargs) -> QueryResult:
        self.validate_collection_name(collection)
        
        if collection not in self._data:
            return QueryResult(data=[], total_count=0, page=1, page_size=0, has_more=False)
        
        # Simple filter implementation for testing
        results = list(self._data[collection].values())
        
        if filters:
            filtered_results = []
            for record in results:
                match = True
                for field, value in filters.items():
                    if field not in record or record[field] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(record)
            results = filtered_results
        
        return QueryResult(
            data=results,
            total_count=len(results),
            page=1,
            page_size=len(results),
            has_more=False
        )
    
    async def bulk_insert(self, collection: str, records: List[DatabaseRecord]) -> List[str]:
        ids = []
        for record in records:
            doc_id = await self.create(collection, record)
            ids.append(doc_id)
        return ids
    
    async def bulk_update(self, collection: str, updates: List[Dict[str, Any]]) -> int:
        count = 0
        for update in updates:
            doc_id = update.get('id')
            data = update.get('data', {})
            if doc_id and await self.update(collection, doc_id, data):
                count += 1
        return count
    
    async def bulk_delete(self, collection: str, filters: QueryFilter) -> int:
        if collection not in self._data:
            return 0
        
        to_delete = []
        for doc_id, record in self._data[collection].items():
            match = True
            for field, value in filters.items():
                if field not in record or record[field] != value:
                    match = False
                    break
            if match:
                to_delete.append(doc_id)
        
        for doc_id in to_delete:
            del self._data[collection][doc_id]
        
        return len(to_delete)


class TestBaseDatabase:
    """Test base database functionality using mock implementation."""
    
    @pytest.fixture
    def database(self):
        """Create mock database for testing."""
        config = DatabaseConfig(backend="mock", database_name="test_db")
        return MockDatabaseImplementation(config)
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, database):
        """Test database connection lifecycle."""
        # Initially disconnected
        assert not database.is_connected
        
        # Connect
        await database.connect()
        assert database.is_connected
        
        # Health check
        health = await database.health_check()
        assert health['status'] == 'healthy'
        
        # Disconnect
        await database.disconnect()
        assert not database.is_connected
    
    @pytest.mark.asyncio
    async def test_crud_operations(self, database):
        """Test basic CRUD operations."""
        await database.connect()
        
        # Create
        test_data = {"name": "John Doe", "email": "john@example.com"}
        doc_id = await database.create("users", test_data)
        assert doc_id is not None
        
        # Read
        retrieved = await database.get("users", doc_id)
        assert retrieved == test_data
        
        # Update
        update_data = {"email": "john.doe@example.com"}
        updated = await database.update("users", doc_id, update_data)
        assert updated is True
        
        # Verify update
        retrieved = await database.get("users", doc_id)
        assert retrieved["email"] == "john.doe@example.com"
        assert retrieved["name"] == "John Doe"
        
        # Delete
        deleted = await database.delete("users", doc_id)
        assert deleted is True
        
        # Verify deletion
        retrieved = await database.get("users", doc_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_query_operations(self, database):
        """Test query operations."""
        await database.connect()
        
        # Insert test data
        users = [
            {"name": "Alice", "age": 25, "status": "active"},
            {"name": "Bob", "age": 30, "status": "active"},
            {"name": "Charlie", "age": 35, "status": "inactive"}
        ]
        
        for user in users:
            await database.create("users", user)
        
        # Query all users
        result = await database.query("users")
        assert result.total_count == 3
        assert len(result.data) == 3
        
        # Query with filters
        result = await database.query("users", {"status": "active"})
        assert result.total_count == 2
        assert len(result.data) == 2
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, database):
        """Test bulk operations."""
        await database.connect()
        
        # Bulk insert
        records = [
            {"name": "User 1", "type": "test"},
            {"name": "User 2", "type": "test"},
            {"name": "User 3", "type": "test"}
        ]
        
        ids = await database.bulk_insert("users", records)
        assert len(ids) == 3
        
        # Bulk update
        updates = [
            {"id": ids[0], "data": {"status": "updated"}},
            {"id": ids[1], "data": {"status": "updated"}}
        ]
        
        updated_count = await database.bulk_update("users", updates)
        assert updated_count == 2
        
        # Bulk delete
        deleted_count = await database.bulk_delete("users", {"type": "test"})
        assert deleted_count == 3
    
    def test_validation_methods(self, database):
        """Test input validation methods."""
        # Valid collection name
        database.validate_collection_name("users")
        database.validate_collection_name("user_profiles")
        database.validate_collection_name("user-profiles")
        
        # Invalid collection names
        with pytest.raises(ValidationError):
            database.validate_collection_name("")
        
        with pytest.raises(ValidationError):
            database.validate_collection_name("user spaces")
        
        with pytest.raises(ValidationError):
            database.validate_collection_name("user@profiles")
        
        # Valid document IDs
        database.validate_document_id("user123")
        database.validate_document_id("user-123")
        database.validate_document_id("user_123")
        
        # Invalid document IDs
        with pytest.raises(ValidationError):
            database.validate_document_id("")
        
        with pytest.raises(ValidationError):
            database.validate_document_id("user 123")
        
        with pytest.raises(ValidationError):
            database.validate_document_id("user@123")


class TestIntegration:
    """Integration tests for complete database stack."""
    
    @pytest.mark.asyncio
    async def test_full_stack_integration(self):
        """Test full database stack with all components."""
        # Create configuration
        config = DatabaseConfig(
            backend="mock",
            database_name="integration_test"
        )
        
        # Create factory and register mock backend
        factory = DatabaseFactory()
        factory.register_backend("mock", MockDatabaseImplementation)
        
        # Mock external dependencies
        with patch('database.factory.EncryptionManager') as mock_encryption, \
             patch('database.factory.CacheManager') as mock_cache, \
             patch('database.factory.AuditLogger') as mock_audit:
            
            # Setup mocks
            encryption_manager = Mock()
            encryption_manager.encrypt_data = AsyncMock(side_effect=lambda x: x)
            encryption_manager.decrypt_data = AsyncMock(side_effect=lambda x: x)
            mock_encryption.return_value = encryption_manager
            
            cache_manager = Mock()
            cache_manager.get = AsyncMock(return_value=None)
            cache_manager.set = AsyncMock()
            cache_manager.delete = AsyncMock()
            cache_manager.build_record_key = Mock(return_value="cache_key")
            mock_cache.return_value = cache_manager
            
            audit_logger = Mock()
            audit_logger.log_operation = AsyncMock(return_value="trace_123")
            mock_audit.return_value = audit_logger
            
            # Create complete stack
            database = factory.create_complete_stack(
                config,
                encryption_key="test_key",
                cache_config={"host": "localhost"},
                audit_config={"log_level": "INFO"}
            )
            
            # Test operations
            await database.connect()
            
            test_data = {"name": "Integration Test", "value": 42}
            doc_id = await database.create("test_collection", test_data)
            
            retrieved = await database.get("test_collection", doc_id)
            assert retrieved == test_data
            
            # Verify component interactions
            database.set_encryption_manager.assert_called_once_with(encryption_manager)
            database.set_cache_manager.assert_called_once_with(cache_manager)
            database.set_audit_logger.assert_called_once_with(audit_logger)


def mock_open(*args, **kwargs):
    """Helper function to create a mock open context manager."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open(*args, **kwargs)


# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_data = {
        "database": {
            "backend": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database_name": "test_db",
            "username": "test_user",
            "password": "test_pass",
            "use_ssl": False
        },
        "encryption": {
            "key": "test_encryption_key_32_chars_long"
        },
        "cache": {
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "audit": {
            "log_level": "INFO",
            "enable_console": True,
            "enable_file": False
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        f.flush()
        yield f.name
    
    os.unlink(f.name)


# Performance test helpers
class PerformanceTestHelper:
    """Helper class for performance and load testing."""
    
    @staticmethod
    async def measure_operation_time(operation_func, *args, **kwargs):
        """Measure the execution time of an async operation."""
        import time
        start_time = time.time()
        result = await operation_func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    
    @staticmethod
    async def run_concurrent_operations(operation_func, args_list, max_concurrency=10):
        """Run multiple operations concurrently with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def limited_operation(args):
            async with semaphore:
                return await operation_func(*args)
        
        tasks = [limited_operation(args) for args in args_list]
        return await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
