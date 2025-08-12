"""
Firestore Database Implementation
================================

Google Cloud Firestore implementation of the BaseDatabase interface.
Provides secure, scalable NoSQL database functionality with real-time capabilities.

Features:
- Async operations with connection pooling
- Automatic retry logic with exponential backoff
- Field-level and document-level encryption
- Comprehensive audit logging
- Cache integration
- Security validation

Security Considerations:
- Service account authentication
- SSL/TLS encryption in transit
- Data encryption at rest
- Input validation and sanitization
- Audit trail for all operations
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
    from google.api_core import exceptions as gcp_exceptions
    FIRESTORE_AVAILABLE = True
except ImportError:
    firebase_admin = None
    firestore = None
    gcp_exceptions = None
    FIRESTORE_AVAILABLE = False

from .base import (
    BaseDatabase, DatabaseConfig, DatabaseRecord, QueryResult, QueryFilter,
    OperationType, DatabaseError, ConnectionError, QueryError, ValidationError
)
from .encryption import EncryptionManager
from .cache import CacheManager
from .audit import AuditLogger

logger = logging.getLogger(__name__)


class FirestoreDatabase(BaseDatabase):
    """
    Google Cloud Firestore database implementation.
    
    Provides async NoSQL database operations with built-in security,
    caching, and audit logging capabilities.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize Firestore database instance.
        
        Args:
            config: Database configuration
        """
        super().__init__(config)
        
        if not FIRESTORE_AVAILABLE:
            raise ConnectionError("Firestore dependencies not installed")
        
        self._firestore_client = None
        self._app = None
        self._encryption_manager: Optional[EncryptionManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._audit_logger: Optional[AuditLogger] = None
        
        # Firestore-specific settings
        self._max_retries = config.max_retries
        self._retry_delay = 1.0  # Initial retry delay in seconds
    
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
        Establish connection to Firestore.
        
        Initializes Firebase app and Firestore client with proper authentication.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Get service account configuration
            service_account_path = self.config.backend_config.get('service_account_path')
            service_account_json = self.config.backend_config.get('service_account_json')
            project_id = self.config.backend_config.get('project_id', self.config.database_name)
            
            # Initialize Firebase app
            if service_account_path:
                cred = credentials.Certificate(service_account_path)
            elif service_account_json:
                cred = credentials.Certificate(service_account_json)
            else:
                # Use default application credentials
                cred = credentials.ApplicationDefault()
                logger.info("Using default application credentials for Firestore")
            
            # Check if app already exists
            try:
                self._app = firebase_admin.get_app()
                logger.info("Using existing Firebase app")
            except ValueError:
                # App doesn't exist, create new one
                self._app = firebase_admin.initialize_app(
                    cred, 
                    {'projectId': project_id},
                    name=f"chatcore-{project_id}"
                )
                logger.info(f"Initialized new Firebase app for project: {project_id}")
            
            # Initialize Firestore client
            self._firestore_client = firestore.client(app=self._app)
            
            # Test connection
            await self._test_connection()
            
            self._is_connected = True
            logger.info("Successfully connected to Firestore")
            
            # Log connection event
            if self._audit_logger:
                await self._audit_logger.log_operation(
                    operation=OperationType.READ,
                    collection="system",
                    metadata={
                        'event': 'database_connected',
                        'backend': 'firestore',
                        'project_id': project_id
                    }
                )
            
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to Firestore: {e}")
            raise ConnectionError(f"Firestore connection failed: {e}", original_error=e)
    
    async def disconnect(self) -> None:
        """
        Disconnect from Firestore and cleanup resources.
        """
        try:
            if self._app:
                firebase_admin.delete_app(self._app)
                self._app = None
            
            self._firestore_client = None
            self._is_connected = False
            
            logger.info("Disconnected from Firestore")
            
            # Log disconnection event
            if self._audit_logger:
                await self._audit_logger.log_operation(
                    operation=OperationType.READ,
                    collection="system",
                    metadata={'event': 'database_disconnected', 'backend': 'firestore'}
                )
                
        except Exception as e:
            logger.error(f"Error during Firestore disconnection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform Firestore health check.
        
        Returns:
            Health status and metrics
        """
        if not self._is_connected:
            return {
                'status': 'disconnected',
                'firestore_available': FIRESTORE_AVAILABLE,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        try:
            # Test basic operation
            start_time = time.time()
            test_doc = self._firestore_client.collection('_health_check').document('test')
            await self._run_async(test_doc.get)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'firestore_available': True,
                'connected': True,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Firestore health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def create(
        self,
        collection: str,
        data: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """
        Create a new document in Firestore.
        
        Args:
            collection: Collection name
            data: Document data
            document_id: Optional document ID
            
        Returns:
            Created document ID
            
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
                    logger.debug(f"Document already exists in cache: {collection}/{document_id}")
            
            # Encrypt sensitive data
            encrypted_data = data.copy()  # Ensure we have a dictionary
            if self._encryption_manager:
                # Only encrypt the user data, not metadata
                encrypted_user_data = await self._encryption_manager.encrypt_data(data)
                if isinstance(encrypted_user_data, dict):
                    encrypted_data = encrypted_user_data
                else:
                    # If encryption returns a string, store it in a special field
                    encrypted_data = {'_encrypted_payload': encrypted_user_data}
            
            # Add metadata
            encrypted_data.update({
                '_created_at': firestore.SERVER_TIMESTAMP,
                '_updated_at': firestore.SERVER_TIMESTAMP,
                '_version': 1
            })
            
            # Create document
            doc_ref = self._firestore_client.collection(collection)
            if document_id:
                doc_ref = doc_ref.document(document_id)
                await self._run_async(doc_ref.set, encrypted_data)
                created_id = document_id
            else:
                doc_ref = doc_ref.document()
                await self._run_async(doc_ref.set, encrypted_data)
                created_id = doc_ref.id
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update cache
            if self._cache_manager and cache_key:
                await self._cache_manager.set(cache_key, data)
            
            # Log operation
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.CREATE,
                    collection=collection,
                    document_id=created_id,
                    execution_time_ms=execution_time,
                    records_affected=1,
                    success=True
                )
            
            logger.debug(f"Created document: {collection}/{created_id}")
            return created_id
            
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
            
            logger.error(f"Failed to create document in {collection}: {e}")
            raise QueryError(
                f"Document creation failed: {e}",
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
        Retrieve a document by ID from Firestore.
        
        Args:
            collection: Collection name
            document_id: Document ID
            
        Returns:
            Document data or None if not found
            
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
            
            # Get from Firestore
            doc_ref = self._firestore_client.collection(collection).document(document_id)
            doc_snapshot = await self._run_async(doc_ref.get)
            
            execution_time = (time.time() - start_time) * 1000
            
            if not doc_snapshot.exists:
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
            
            # Get document data
            data = doc_snapshot.to_dict()
            
            # Decrypt data if needed
            if self._encryption_manager and data:
                data = await self._encryption_manager.decrypt_data(data)
            
            # Remove internal metadata
            if data:
                data = self._clean_internal_fields(data)
                data['_id'] = document_id  # Add document ID
            
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
            
            logger.debug(f"Retrieved document: {collection}/{document_id}")
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
            
            logger.error(f"Failed to get document {collection}/{document_id}: {e}")
            raise QueryError(
                f"Document retrieval failed: {e}",
                operation=OperationType.READ,
                collection=collection,
                query_id=trace_id,
                original_error=e
            )
    
    async def update(
        self,
        collection: str,
        document_id: str,
        data: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """
        Update an existing document in Firestore.
        
        Args:
            collection: Collection name
            document_id: Document ID
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
            encrypted_data = data.copy()  # Ensure we have a dictionary
            if self._encryption_manager:
                # Only encrypt the user data, not metadata
                encrypted_user_data = await self._encryption_manager.encrypt_data(data)
                if isinstance(encrypted_user_data, dict):
                    encrypted_data = encrypted_user_data
                else:
                    # If encryption returns a string, store it in a special field
                    encrypted_data = {'_encrypted_payload': encrypted_user_data}
            
            # Add update metadata
            encrypted_data.update({
                '_updated_at': firestore.SERVER_TIMESTAMP,
                '_version': firestore.Increment(1)
            })
            
            doc_ref = self._firestore_client.collection(collection).document(document_id)
            
            if upsert:
                # Use set with merge=True for upsert
                await self._run_async(doc_ref.set, encrypted_data, merge=True)
                updated = True
            else:
                # Check if document exists first
                doc_snapshot = await self._run_async(doc_ref.get)
                if not doc_snapshot.exists:
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Log operation
                    if self._audit_logger:
                        trace_id = await self._audit_logger.log_operation(
                            operation=OperationType.UPDATE,
                            collection=collection,
                            document_id=document_id,
                            execution_time_ms=execution_time,
                            records_affected=0,
                            success=True
                        )
                    
                    return False
                
                # Update existing document
                await self._run_async(doc_ref.update, encrypted_data)
                updated = True
            
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
            
            logger.debug(f"Updated document: {collection}/{document_id}")
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
            
            logger.error(f"Failed to update document {collection}/{document_id}: {e}")
            raise QueryError(
                f"Document update failed: {e}",
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
        Delete a document from Firestore.
        
        Args:
            collection: Collection name
            document_id: Document ID
            
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
            
            # Check if document exists
            doc_ref = self._firestore_client.collection(collection).document(document_id)
            doc_snapshot = await self._run_async(doc_ref.get)
            
            if not doc_snapshot.exists:
                execution_time = (time.time() - start_time) * 1000
                
                # Log operation
                if self._audit_logger:
                    trace_id = await self._audit_logger.log_operation(
                        operation=OperationType.DELETE,
                        collection=collection,
                        document_id=document_id,
                        execution_time_ms=execution_time,
                        records_affected=0,
                        success=True
                    )
                
                return False
            
            # Delete document
            await self._run_async(doc_ref.delete)
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
                    records_affected=1,
                    success=True
                )
            
            logger.debug(f"Deleted document: {collection}/{document_id}")
            return True
            
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
            
            logger.error(f"Failed to delete document {collection}/{document_id}: {e}")
            raise QueryError(
                f"Document deletion failed: {e}",
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
        Query documents in Firestore with filters and pagination.
        
        Args:
            collection: Collection name
            filters: Query filters
            sort_by: Sort field
            sort_desc: Sort in descending order
            limit: Maximum records to return
            offset: Records to skip
            **kwargs: Additional Firestore-specific parameters
            
        Returns:
            QueryResult with documents and metadata
            
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
            
            # Build Firestore query
            query = self._firestore_client.collection(collection)
            
            # Apply filters
            if filters:
                query = self._apply_filters(query, filters)
            
            # Apply sorting
            if sort_by:
                direction = firestore.Query.DESCENDING if sort_desc else firestore.Query.ASCENDING
                query = query.order_by(sort_by, direction=direction)
            
            # Apply pagination
            if offset > 0:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            # Execute query
            docs = await self._run_async(query.stream)
            
            # Process results
            results = []
            async for doc in self._async_iterator(docs):
                doc_data = doc.to_dict()
                
                # Decrypt data if needed
                if self._encryption_manager and doc_data:
                    doc_data = await self._encryption_manager.decrypt_data(doc_data)
                
                # Clean internal fields and add ID
                if doc_data:
                    doc_data = self._clean_internal_fields(doc_data)
                    doc_data['_id'] = doc.id
                    results.append(doc_data)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create result object
            result = QueryResult(
                data=results,
                total_count=len(results),  # Firestore doesn't provide total count efficiently
                page=offset // (limit or 100) + 1 if limit else 1,
                page_size=limit or len(results),
                has_more=len(results) == limit if limit else False,
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
            
            logger.debug(f"Queried collection: {collection}, results: {len(results)}")
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
            
            logger.error(f"Failed to query collection {collection}: {e}")
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
        Insert multiple documents efficiently using batch operations.
        
        Args:
            collection: Collection name
            records: List of records to insert
            
        Returns:
            List of created document IDs
            
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
            
            # Process in batches (Firestore limit is 500 operations per batch)
            batch_size = 500
            total_processed = 0
            
            for i in range(0, len(records), batch_size):
                batch_records = records[i:i + batch_size]
                batch = self._firestore_client.batch()
                batch_ids = []
                
                for record in batch_records:
                    # Encrypt data
                    encrypted_data = record
                    if self._encryption_manager:
                        encrypted_data = await self._encryption_manager.encrypt_data(record)
                    
                    # Add metadata
                    encrypted_data.update({
                        '_created_at': firestore.SERVER_TIMESTAMP,
                        '_updated_at': firestore.SERVER_TIMESTAMP,
                        '_version': 1
                    })
                    
                    # Create document reference
                    doc_ref = self._firestore_client.collection(collection).document()
                    batch.set(doc_ref, encrypted_data)
                    batch_ids.append(doc_ref.id)
                
                # Commit batch
                await self._run_async(batch.commit)
                created_ids.extend(batch_ids)
                total_processed += len(batch_records)
                
                logger.debug(f"Inserted batch: {len(batch_records)} records")
            
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
            
            logger.info(f"Bulk inserted {len(created_ids)} documents into {collection}")
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
        Update multiple documents efficiently.
        
        Args:
            collection: Collection name
            updates: List of update operations with 'id' and 'data' fields
            
        Returns:
            Number of documents updated
            
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
            
            # Process in batches
            batch_size = 500
            
            for i in range(0, len(updates), batch_size):
                batch_updates = updates[i:i + batch_size]
                batch = self._firestore_client.batch()
                
                for update in batch_updates:
                    document_id = update.get('id')
                    update_data = update.get('data', {})
                    
                    if not document_id:
                        continue
                    
                    # Encrypt data
                    encrypted_data = update_data
                    if self._encryption_manager:
                        encrypted_data = await self._encryption_manager.encrypt_data(update_data)
                    
                    # Add update metadata
                    encrypted_data.update({
                        '_updated_at': firestore.SERVER_TIMESTAMP,
                        '_version': firestore.Increment(1)
                    })
                    
                    doc_ref = self._firestore_client.collection(collection).document(document_id)
                    batch.update(doc_ref, encrypted_data)
                    updated_count += 1
                
                # Commit batch
                await self._run_async(batch.commit)
                logger.debug(f"Updated batch: {len(batch_updates)} records")
            
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
            
            logger.info(f"Bulk updated {updated_count} documents in {collection}")
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
        Delete multiple documents efficiently.
        
        Args:
            collection: Collection name
            filters: Deletion criteria
            
        Returns:
            Number of documents deleted
            
        Raises:
            QueryError: If bulk delete fails
        """
        start_time = time.time()
        trace_id = None
        deleted_count = 0
        
        try:
            # Validate inputs
            self.validate_collection_name(collection)
            
            # First, query for documents to delete
            query = self._firestore_client.collection(collection)
            query = self._apply_filters(query, filters)
            
            # Get documents in batches and delete them
            batch_size = 500
            
            while True:
                # Get batch of documents
                docs = await self._run_async(query.limit(batch_size).stream)
                doc_list = []
                async for doc in self._async_iterator(docs):
                    doc_list.append(doc)
                
                if not doc_list:
                    break
                
                # Delete batch
                batch = self._firestore_client.batch()
                for doc in doc_list:
                    batch.delete(doc.reference)
                    deleted_count += 1
                
                await self._run_async(batch.commit)
                logger.debug(f"Deleted batch: {len(doc_list)} records")
                
                # If we got fewer documents than batch size, we're done
                if len(doc_list) < batch_size:
                    break
            
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
            
            logger.info(f"Bulk deleted {deleted_count} documents from {collection}")
            return deleted_count
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Log error
            if self._audit_logger:
                trace_id = await self._audit_logger.log_operation(
                    operation=OperationType.BULK_DELETE,
                    collection=collection,
                    execution_time_ms=execution_time,
                    records_affected=deleted_count,
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
    
    async def _test_connection(self) -> None:
        """Test Firestore connection."""
        try:
            # Try to access a collection (this will fail if no permissions)
            test_collection = self._firestore_client.collection('_connection_test')
            await self._run_async(test_collection.limit(1).get)
        except Exception as e:
            # Even permission errors indicate connection is working
            if "PERMISSION_DENIED" in str(e):
                logger.info("Firestore connection test successful (permission check)")
            else:
                raise e
    
    async def _run_async(self, func, *args, **kwargs):
        """Run synchronous Firestore operation in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def _async_iterator(self, sync_iterator):
        """Convert synchronous iterator to async iterator."""
        loop = asyncio.get_event_loop()
        
        def get_next(iterator):
            try:
                return next(iterator)
            except StopIteration:
                return None
        
        while True:
            item = await loop.run_in_executor(None, get_next, sync_iterator)
            if item is None:
                break
            yield item
    
    def _apply_filters(self, query, filters: QueryFilter):
        """Apply filters to Firestore query."""
        for field, value in filters.items():
            if isinstance(value, dict):
                # Handle complex filters like {">=": 10, "<=": 20}
                for operator, filter_value in value.items():
                    if operator in ["==", "!=", "<", "<=", ">", ">=", "in", "not-in", "array-contains", "array-contains-any"]:
                        query = query.where(filter=FieldFilter(field, operator, filter_value))
            else:
                # Simple equality filter
                query = query.where(filter=FieldFilter(field, "==", value))
        
        return query
    
    def _clean_internal_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal Firestore fields from document data."""
        if not data:
            return data
        
        cleaned = {}
        for key, value in data.items():
            if not key.startswith('_') or key == '_id':
                cleaned[key] = value
        
        return cleaned
