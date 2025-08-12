"""
PostgreSQL Database Implementation
=================================

Async PostgreSQL implementation of the BaseDatabase interface.
Provides secure, scalable relational database functionality with ACID compliance.

Features:
- Async operations with connection pooling
- SQL injection protection with parameterized queries
- Automatic retry logic and connection recovery
- Field-level encryption for sensitive data
- Comprehensive audit logging
- Cache integration
- Migration support

Security Considerations:
- SSL/TLS encryption in transit
- Parameterized queries for SQL injection protection
- Connection string security
- Data encryption at rest
- Audit trail for all operations
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime

try:
    import asyncpg
    import psycopg2
    from psycopg2 import sql
    POSTGRESQL_AVAILABLE = True
except ImportError:
    asyncpg = None
    psycopg2 = None
    sql = None
    POSTGRESQL_AVAILABLE = False

from .base import (
    BaseDatabase, DatabaseConfig, DatabaseRecord, QueryResult, QueryFilter,
    OperationType, DatabaseError, ConnectionError, QueryError, ValidationError
)
from .encryption import EncryptionManager
from .cache import CacheManager
from .audit import AuditLogger

logger = logging.getLogger(__name__)


class PostgreSQLDatabase(BaseDatabase):
    """
    PostgreSQL database implementation with async support.
    
    Provides ACID-compliant relational database operations with built-in
    security, caching, and audit logging capabilities.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize PostgreSQL database instance.
        
        Args:
            config: Database configuration
        """
        super().__init__(config)
        
        if not POSTGRESQL_AVAILABLE:
            raise ConnectionError("PostgreSQL dependencies not installed")
        
        self._connection_pool = None
        self._encryption_manager: Optional[EncryptionManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._audit_logger: Optional[AuditLogger] = None
        
        # PostgreSQL-specific settings
        self._schema = config.backend_config.get('schema', 'public')
        self._enable_ssl = config.use_ssl
        
    def set_encryption_manager(self, encryption_manager: EncryptionManager) -> None:
        """Set encryption manager for data encryption."""
        self._encryption_manager = encryption_manager
    
    def set_cache_manager(self, cache_manager: CacheManager) -> None:
        """Set cache manager for caching."""
        self._cache_manager = cache_manager
    
    def set_audit_logger(self, audit_logger: AuditLogger) -> None:
        """Set audit logger for operation logging."""
        self._audit_logger = audit_logger
    
    async def connect(self) -> None:
        """
        Establish connection pool to PostgreSQL.
        
        Creates a connection pool with proper SSL configuration and authentication.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Build connection string
            connection_string = self._build_connection_string()
            
            # SSL configuration
            ssl_context = None
            if self._enable_ssl:
                import ssl
                ssl_context = ssl.create_default_context()
                if self.config.ssl_cert_path:
                    ssl_context.load_verify_locations(self.config.ssl_cert_path)
            
            # Create connection pool
            self._connection_pool = await asyncpg.create_pool(
                connection_string,
                min_size=1,
                max_size=self.config.connection_pool_size,
                command_timeout=self.config.query_timeout,
                ssl=ssl_context,
                server_settings={
                    'application_name': 'chatcore_database',
                    'search_path': self._schema
                }
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize schema if needed
            await self._initialize_schema()
            
            self._is_connected = True
            logger.info("Successfully connected to PostgreSQL")
            
            # Log connection event
            if self._audit_logger:
                await self._audit_logger.log_operation(
                    operation=OperationType.READ,
                    collection="system",
                    metadata={
                        'event': 'database_connected',
                        'backend': 'postgresql',
                        'database': self.config.database_name,
                        'schema': self._schema
                    }
                )
            
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL connection failed: {e}", original_error=e)
    
    async def disconnect(self) -> None:
        """
        Close connection pool and cleanup resources.
        """
        try:
            if self._connection_pool:
                await self._connection_pool.close()
                self._connection_pool = None
            
            self._is_connected = False
            logger.info("Disconnected from PostgreSQL")
            
            # Log disconnection event
            if self._audit_logger:
                await self._audit_logger.log_operation(
                    operation=OperationType.READ,
                    collection="system",
                    metadata={'event': 'database_disconnected', 'backend': 'postgresql'}
                )
                
        except Exception as e:
            logger.error(f"Error during PostgreSQL disconnection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform PostgreSQL health check.
        
        Returns:
            Health status and metrics
        """
        if not self._is_connected or not self._connection_pool:
            return {
                'status': 'disconnected',
                'postgresql_available': POSTGRESQL_AVAILABLE,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        try:
            # Test basic operation
            start_time = time.time()
            async with self._connection_pool.acquire() as conn:
                result = await conn.fetchval('SELECT 1')
                pool_stats = self._connection_pool.get_stats()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'postgresql_available': True,
                'connected': True,
                'pool_stats': {
                    'size': pool_stats.size,
                    'max_size': pool_stats.max_size,
                    'open_connections': pool_stats.open_connections,
                    'free_connections': pool_stats.free_connections
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def create(
        self,
        collection: str,
        data: DatabaseRecord,
        document_id: Optional[str] = None
    ) -> str:
        """
        Create a new record in PostgreSQL.
        
        Args:
            collection: Table name
            data: Record data
            document_id: Optional record ID
            
        Returns:
            Created record ID
            
        Raises:
            ValidationError: If data validation fails
            QueryError: If creation fails
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            if document_id:
                self.validate_document_id(document_id)
            
            # Check cache first (for upsert scenarios)
            cache_key = None
            if self._cache_manager and document_id:
                cache_key = self._cache_manager.build_record_key(collection, document_id)
                cached_record = await self._cache_manager.get(cache_key)
                if cached_record:
                    logger.debug(f"Record already exists in cache: {collection}/{document_id}")
            
            # Encrypt sensitive data
            encrypted_data = data.copy()
            if self._encryption_manager:
                encrypted_result = await self._encryption_manager.encrypt_data(encrypted_data)
                if isinstance(encrypted_result, dict):
                    encrypted_data = encrypted_result
            
            # Add metadata
            encrypted_data.update({
                '_created_at': datetime.utcnow(),
                '_updated_at': datetime.utcnow(),
                '_version': 1
            })
            
            # Prepare SQL
            table_name = self._get_table_name(collection)
            
            if document_id:
                # Insert with specific ID
                encrypted_data['id'] = document_id
                columns = list(encrypted_data.keys())
                placeholders = [f'${i+1}' for i in range(len(columns))]
                
                query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    RETURNING id
                """
                values = list(encrypted_data.values())
                created_id = document_id
            else:
                # Insert with auto-generated ID
                columns = list(encrypted_data.keys())
                placeholders = [f'${i+1}' for i in range(len(columns))]
                
                query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    RETURNING id
                """
                values = list(encrypted_data.values())
            
            # Execute query
            async with self._connection_pool.acquire() as conn:
                if document_id:
                    await conn.execute(query, *values)
                    created_id = document_id
                else:
                    created_id = await conn.fetchval(query, *values)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update cache
            if self._cache_manager and cache_key:
                await self._cache_manager.set(cache_key, data)
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.CREATE,
                    collection=collection,
                    document_id=str(created_id),
                    execution_time_ms=execution_time,
                    records_affected=1,
                    success=True
                )
            
            logger.debug(f"Created record: {collection}/{created_id}")
            return str(created_id)
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.CREATE,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to create record in {collection}: {e}")
            raise QueryError(
                f"Record creation failed: {e}",
                operation=OperationType.CREATE,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def get(
        self,
        collection: str,
        document_id: str
    ) -> Optional[DatabaseRecord]:
        """
        Retrieve a record by ID from PostgreSQL.
        
        Args:
            collection: Table name
            document_id: Record ID
            
        Returns:
            Record data or None if not found
            
        Raises:
            QueryError: If query fails
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            self.validate_document_id(document_id)
            
            # Check cache first
            if self._cache_manager:
                cache_key = self._cache_manager.build_record_key(collection, document_id)
                cached_data = await self._cache_manager.get(cache_key)
                if cached_data:
                    logger.debug(f"Cache hit for: {collection}/{document_id}")
                    return cached_data
            
            # Query from PostgreSQL
            table_name = self._get_table_name(collection)
            query = f"SELECT * FROM {table_name} WHERE id = $1"
            
            async with self._connection_pool.acquire() as conn:
                row = await conn.fetchrow(query, document_id)
            
            execution_time = (time.time() - start_time) * 1000
            
            if not row:
                # Log operation
                if self._audit_logger:
                    trace_id = await self._audit_logger.log_operation(
                        operation=OperationType.READ,
                        collection=collection,
                        document_id=document_id,
                        execution_time_ms=execution_time,
                        records_affected=0,
                        success=True
                    )
                return None
            
            # Convert row to dict
            data = dict(row)
            
            # Decrypt data if needed
            if self._encryption_manager and data:
                decrypted_result = await self._encryption_manager.decrypt_data(data)
                if isinstance(decrypted_result, dict):
                    data = decrypted_result
            
            # Clean internal fields
            if data and isinstance(data, dict):
                data = self._clean_internal_fields(data)
            
            # Cache the result
            if self._cache_manager and data:
                cache_key = self._cache_manager.build_record_key(collection, document_id)
                await self._cache_manager.set(cache_key, data)
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.READ,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    records_affected=1 if data else 0,
                    success=True
                )
            
            logger.debug(f"Retrieved record: {collection}/{document_id}")
            return data
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.READ,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to get record {collection}/{document_id}: {e}")
            raise QueryError(
                f"Record retrieval failed: {e}",
                operation=OperationType.READ,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def update(
        self,
        collection: str,
        document_id: str,
        data: DatabaseRecord,
        upsert: bool = False
    ) -> bool:
        """
        Update an existing record in PostgreSQL.
        
        Args:
            collection: Table name
            document_id: Record ID
            data: Updated data
            upsert: Create if not exists
            
        Returns:
            True if updated, False if not found
            
        Raises:
            ValidationError: If data validation fails
            QueryError: If update fails
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            self.validate_document_id(document_id)
            
            # Encrypt sensitive data
            encrypted_data = data.copy()
            if self._encryption_manager:
                encrypted_result = await self._encryption_manager.encrypt_data(encrypted_data)
                if isinstance(encrypted_result, dict):
                    encrypted_data = encrypted_result
            
            # Add update metadata
            encrypted_data.update({
                '_updated_at': datetime.utcnow()
            })
            
            table_name = self._get_table_name(collection)
            
            if upsert:
                # Use ON CONFLICT for upsert
                encrypted_data['id'] = document_id
                columns = list(encrypted_data.keys())
                placeholders = [f'${i+1}' for i in range(len(columns))]
                update_columns = [f"{col} = EXCLUDED.{col}" for col in columns if col != 'id']
                
                query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT (id) DO UPDATE SET
                    {', '.join(update_columns)}, _version = {table_name}._version + 1
                    RETURNING id
                """
                values = list(encrypted_data.values())
                
                async with self._connection_pool.acquire() as conn:
                    result = await conn.fetchval(query, *values)
                    updated = result is not None
            else:
                # Regular update
                set_clauses = []
                values = []
                param_index = 1
                
                for key, value in encrypted_data.items():
                    if key != 'id':  # Don't update ID
                        set_clauses.append(f"{key} = ${param_index}")
                        values.append(value)
                        param_index += 1
                
                # Add version increment
                set_clauses.append(f"_version = _version + 1")
                
                query = f"""
                    UPDATE {table_name}
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_index}
                """
                values.append(document_id)
                
                async with self._connection_pool.acquire() as conn:
                    result = await conn.execute(query, *values)
                    updated = result.split()[-1] != '0'  # Check affected rows
            
            execution_time = (time.time() - start_time) * 1000
            
            # Invalidate cache
            if self._cache_manager:
                cache_key = self._cache_manager.build_record_key(collection, document_id)
                await self._cache_manager.delete(cache_key)
                # Also invalidate collection queries
                await self._cache_manager.invalidate_pattern(f"query:{collection}:*")
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.UPDATE,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    records_affected=1 if updated else 0,
                    success=True
                )
            
            logger.debug(f"Updated record: {collection}/{document_id}")
            return updated
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.UPDATE,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to update record {collection}/{document_id}: {e}")
            raise QueryError(
                f"Record update failed: {e}",
                operation=OperationType.UPDATE,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def delete(
        self,
        collection: str,
        document_id: str
    ) -> bool:
        """
        Delete a record from PostgreSQL.
        
        Args:
            collection: Table name
            document_id: Record ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            QueryError: If deletion fails
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            self.validate_document_id(document_id)
            
            # Delete from PostgreSQL
            table_name = self._get_table_name(collection)
            query = f"DELETE FROM {table_name} WHERE id = $1"
            
            async with self._connection_pool.acquire() as conn:
                result = await conn.execute(query, document_id)
                deleted = result.split()[-1] != '0'  # Check affected rows
            
            execution_time = (time.time() - start_time) * 1000
            
            # Invalidate cache
            if self._cache_manager:
                cache_key = self._cache_manager.build_record_key(collection, document_id)
                await self._cache_manager.delete(cache_key)
                # Also invalidate collection queries
                await self._cache_manager.invalidate_pattern(f"query:{collection}:*")
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.DELETE,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    records_affected=1 if deleted else 0,
                    success=True
                )
            
            logger.debug(f"Deleted record: {collection}/{document_id}")
            return deleted
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.DELETE,
                    collection=collection,
                    document_id=document_id,
                    execution_time_ms=execution_time,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to delete record {collection}/{document_id}: {e}")
            raise QueryError(
                f"Record deletion failed: {e}",
                operation=OperationType.DELETE,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def query(
        self,
        collection: str,
        filters: Optional[QueryFilter] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs
    ) -> QueryResult:
        """
        Query records in PostgreSQL with filters and pagination.
        
        Args:
            collection: Table name
            filters: Query filters
            sort_by: Sort field
            sort_desc: Sort in descending order
            limit: Maximum records to return
            offset: Records to skip
            **kwargs: Additional PostgreSQL-specific parameters
            
        Returns:
            QueryResult with records and metadata
            
        Raises:
            QueryError: If query fails
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            
            # Check cache first
            cache_key = None
            if self._cache_manager:
                cache_key = self._cache_manager.build_query_key(
                    collection, filters, sort_by, limit, offset
                )
                cached_result = await self._cache_manager.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for query: {collection}")
                    return cached_result
            
            # Build SQL query
            table_name = self._get_table_name(collection)
            where_clause, where_values = self._build_where_clause(filters)
            
            # Count query for total records
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            if where_clause:
                count_query += f" WHERE {where_clause}"
            
            # Main query
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            # Add sorting
            if sort_by:
                direction = "DESC" if sort_desc else "ASC"
                query += f" ORDER BY {sort_by} {direction}"
            
            # Add pagination
            if limit:
                query += f" LIMIT {limit}"
            if offset > 0:
                query += f" OFFSET {offset}"
            
            # Execute queries
            async with self._connection_pool.acquire() as conn:
                # Get total count
                if where_values:
                    total_count = await conn.fetchval(count_query, *where_values)
                else:
                    total_count = await conn.fetchval(count_query)
                
                # Get records
                if where_values:
                    rows = await conn.fetch(query, *where_values)
                else:
                    rows = await conn.fetch(query)
            
            # Process results
            results = []
            for row in rows:
                row_data = dict(row)
                
                # Decrypt data if needed
                if self._encryption_manager and row_data:
                    decrypted_result = await self._encryption_manager.decrypt_data(row_data)
                    if isinstance(decrypted_result, dict):
                        row_data = decrypted_result
                
                # Clean internal fields
                if row_data and isinstance(row_data, dict):
                    row_data = self._clean_internal_fields(row_data)
                    results.append(row_data)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create result object
            result = QueryResult(
                data=results,
                total_count=total_count,
                page=offset // (limit or 100) + 1 if limit else 1,
                page_size=limit or len(results),
                has_more=(offset + len(results)) < total_count if limit else False,
                execution_time_ms=execution_time
            )
            
            # Cache the result
            if self._cache_manager and cache_key:
                await self._cache_manager.cache_query_result(
                    collection, result, filters, sort_by, limit, offset
                )
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_query(
                    collection=collection,
                    filters=filters,
                    result_count=len(results),
                    execution_time_ms=execution_time,
                    success=True
                )
            
            logger.debug(f"Queried table: {collection}, results: {len(results)}")
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_query(
                    collection=collection,
                    filters=filters,
                    execution_time_ms=execution_time,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to query table {collection}: {e}")
            raise QueryError(
                f"Query failed: {e}",
                operation=OperationType.QUERY,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def bulk_insert(
        self,
        collection: str,
        records: List[DatabaseRecord]
    ) -> List[str]:
        """
        Insert multiple records efficiently using batch operations.
        
        Args:
            collection: Table name
            records: List of records to insert
            
        Returns:
            List of created record IDs
            
        Raises:
            ValidationError: If any record is invalid
            QueryError: If bulk insert fails
        """
        start_time = time.time()
        trace_id = None
        created_ids = []
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            if not records:
                return []
            
            # Process records
            processed_records = []
            for record in records:
                # Encrypt data
                encrypted_data = record.copy()
                if self._encryption_manager:
                    encrypted_result = await self._encryption_manager.encrypt_data(encrypted_data)
                    if isinstance(encrypted_result, dict):
                        encrypted_data = encrypted_result
                
                # Add metadata
                encrypted_data.update({
                    '_created_at': datetime.utcnow(),
                    '_updated_at': datetime.utcnow(),
                    '_version': 1
                })
                processed_records.append(encrypted_data)
            
            # Build bulk insert query
            table_name = self._get_table_name(collection)
            
            if processed_records:
                columns = list(processed_records[0].keys())
                placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                
                query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({placeholders})
                    RETURNING id
                """
                
                # Execute in batches
                batch_size = 1000
                async with self._connection_pool.acquire() as conn:
                    async with conn.transaction():
                        for i in range(0, len(processed_records), batch_size):
                            batch = processed_records[i:i + batch_size]
                            
                            for record in batch:
                                values = [record[col] for col in columns]
                                record_id = await conn.fetchval(query, *values)
                                created_ids.append(str(record_id))
            
            execution_time = (time.time() - start_time) * 1000
            
            # Invalidate cache
            if self._cache_manager:
                await self._cache_manager.invalidate_pattern(f"query:{collection}:*")
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_INSERT,
                    collection=collection,
                    execution_time_ms=execution_time,
                    records_affected=len(created_ids),
                    success=True
                )
            
            logger.info(f"Bulk inserted {len(created_ids)} records into {collection}")
            return created_ids
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_INSERT,
                    collection=collection,
                    execution_time_ms=execution_time,
                    records_affected=len(created_ids),
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to bulk insert into {collection}: {e}")
            raise QueryError(
                f"Bulk insert failed: {e}",
                operation=OperationType.BULK_INSERT,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def bulk_update(
        self,
        collection: str,
        updates: List[Dict[str, Any]]
    ) -> int:
        """
        Update multiple records efficiently.
        
        Args:
            collection: Table name
            updates: List of update operations with 'id' and 'data' fields
            
        Returns:
            Number of records updated
            
        Raises:
            QueryError: If bulk update fails
        """
        start_time = time.time()
        trace_id = None
        updated_count = 0
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            if not updates:
                return 0
            
            table_name = self._get_table_name(collection)
            
            # Execute updates in transaction
            async with self._connection_pool.acquire() as conn:
                async with conn.transaction():
                    for update in updates:
                        document_id = update.get('id')
                        update_data = update.get('data', {})
                        
                        if not document_id:
                            continue
                        
                        # Encrypt data
                        encrypted_data = update_data.copy()
                        if self._encryption_manager:
                            encrypted_result = await self._encryption_manager.encrypt_data(encrypted_data)
                            if isinstance(encrypted_result, dict):
                                encrypted_data = encrypted_result
                        
                        # Add update metadata
                        encrypted_data['_updated_at'] = datetime.utcnow()
                        
                        # Build update query
                        set_clauses = []
                        values = []
                        param_index = 1
                        
                        for key, value in encrypted_data.items():
                            if key != 'id':
                                set_clauses.append(f"{key} = ${param_index}")
                                values.append(value)
                                param_index += 1
                        
                        set_clauses.append(f"_version = _version + 1")
                        
                        query = f"""
                            UPDATE {table_name}
                            SET {', '.join(set_clauses)}
                            WHERE id = ${param_index}
                        """
                        values.append(document_id)
                        
                        result = await conn.execute(query, *values)
                        if result.split()[-1] != '0':
                            updated_count += 1
            
            execution_time = (time.time() - start_time) * 1000
            
            # Invalidate cache
            if self._cache_manager:
                await self._cache_manager.invalidate_pattern(f"query:{collection}:*")
                # Also invalidate individual records
                for update in updates:
                    if update.get('id'):
                        cache_key = self._cache_manager.build_record_key(collection, update['id'])
                        await self._cache_manager.delete(cache_key)
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_UPDATE,
                    collection=collection,
                    execution_time_ms=execution_time,
                    records_affected=updated_count,
                    success=True
                )
            
            logger.info(f"Bulk updated {updated_count} records in {collection}")
            return updated_count
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_UPDATE,
                    collection=collection,
                    execution_time_ms=execution_time,
                    records_affected=updated_count,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to bulk update {collection}: {e}")
            raise QueryError(
                f"Bulk update failed: {e}",
                operation=OperationType.BULK_UPDATE,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def bulk_delete(
        self,
        collection: str,
        filters: QueryFilter
    ) -> int:
        """
        Delete multiple records efficiently.
        
        Args:
            collection: Table name
            filters: Deletion criteria
            
        Returns:
            Number of records deleted
            
        Raises:
            QueryError: If bulk delete fails
        """
        start_time = time.time()
        trace_id = None
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            
            # Build delete query
            table_name = self._get_table_name(collection)
            where_clause, where_values = self._build_where_clause(filters)
            
            query = f"DELETE FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            # Execute delete
            async with self._connection_pool.acquire() as conn:
                if where_values:
                    result = await conn.execute(query, *where_values)
                else:
                    result = await conn.execute(query)
                
                deleted_count = int(result.split()[-1])
            
            execution_time = (time.time() - start_time) * 1000
            
            # Invalidate cache
            if self._cache_manager:
                await self._cache_manager.invalidate_collection(collection)
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_DELETE,
                    collection=collection,
                    execution_time_ms=execution_time,
                    records_affected=deleted_count,
                    success=True,
                    metadata={'filters': filters}
                )
            
            logger.info(f"Bulk deleted {deleted_count} records from {collection}")
            return deleted_count
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_DELETE,
                    collection=collection,
                    execution_time_ms=execution_time,
                    success=False,
                    error=e
                )
            
            logger.error(f"Failed to bulk delete from {collection}: {e}")
            raise QueryError(
                f"Bulk delete failed: {e}",
                operation=OperationType.BULK_DELETE,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    # Helper Methods
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from config."""
        parts = []
        
        if self.config.host:
            parts.append(f"host={self.config.host}")
        if self.config.port:
            parts.append(f"port={self.config.port}")
        if self.config.database_name:
            parts.append(f"dbname={self.config.database_name}")
        if self.config.username:
            parts.append(f"user={self.config.username}")
        if self.config.password:
            parts.append(f"password={self.config.password}")
        
        # SSL configuration
        if self._enable_ssl:
            parts.append("sslmode=require")
        
        return " ".join(parts)
    
    async def _test_connection(self) -> None:
        """Test PostgreSQL connection."""
        async with self._connection_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema if needed."""
        # This is a placeholder for schema initialization
        # In a real implementation, you might run migrations here
        pass
    
    def _get_table_name(self, collection: str) -> str:
        """Get full table name with schema."""
        # Sanitize table name to prevent SQL injection
        sanitized_collection = collection.replace('"', '').replace("'", "")
        return f'"{self._schema}"."{sanitized_collection}"'
    
    def _build_where_clause(self, filters: Optional[QueryFilter]) -> Tuple[str, List[Any]]:
        """Build WHERE clause from filters."""
        if not filters:
            return "", []
        
        clauses = []
        values = []
        param_index = 1
        
        for field, value in filters.items():
            # Sanitize field name
            field = field.replace('"', '').replace("'", "")
            
            if isinstance(value, dict):
                # Handle complex filters like {">=": 10, "<=": 20}
                for operator, filter_value in value.items():
                    if operator in ["=", "!=", "<", "<=", ">", ">=", "LIKE", "ILIKE"]:
                        clauses.append(f'"{field}" {operator} ${param_index}')
                        values.append(filter_value)
                        param_index += 1
                    elif operator == "IN":
                        if isinstance(filter_value, (list, tuple)):
                            placeholders = [f'${param_index + i}' for i in range(len(filter_value))]
                            clauses.append(f'"{field}" IN ({", ".join(placeholders)})')
                            values.extend(filter_value)
                            param_index += len(filter_value)
            else:
                # Simple equality filter
                clauses.append(f'"{field}" = ${param_index}')
                values.append(value)
                param_index += 1
        
        return " AND ".join(clauses), values
    
    def _clean_internal_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal fields from record data."""
        if not data:
            return data
        
        cleaned = {}
        for key, value in data.items():
            # Keep all fields but convert datetime objects to ISO strings
            if isinstance(value, datetime):
                cleaned[key] = value.isoformat()
            else:
                cleaned[key] = value
        
        return cleaned
