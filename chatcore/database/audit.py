"""
Database Audit Logger
====================

Provides comprehensive auditing and logging for all database operations.
Tracks operations with trace IDs for debugging and compliance requirements.

Features:
- Operation tracking with trace IDs
- Performance metrics collection
- Security event logging
- Compliance audit trails
- Structured logging for analysis

Usage:
    audit_logger = AuditLogger(config)
    
    # Log database operation
    await audit_logger.log_operation(
        operation=OperationType.CREATE,
        collection="users",
        document_id="user123",
        metadata={"user_id": "admin", "ip": "192.168.1.1"}
    )
"""

import asyncio
import time
import json
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from collections import deque

from .base import OperationType, DatabaseError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit logging levels."""
    MINIMAL = "minimal"      # Only track errors and security events
    STANDARD = "standard"    # Track all operations with basic metadata
    DETAILED = "detailed"    # Track all operations with full metadata
    VERBOSE = "verbose"      # Track everything including performance metrics


@dataclass
class AuditEvent:
    """Individual audit event record."""
    
    # Core identifiers (required fields first)
    trace_id: str
    operation_id: str
    operation: OperationType
    collection: str
    
    # Optional fields with defaults
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document_id: Optional[str] = None
    
    # Performance metrics
    execution_time_ms: float = 0.0
    records_affected: int = 0
    
    # Security context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Operation context
    success: bool = True
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for logging."""
        return {
            'trace_id': self.trace_id,
            'operation_id': self.operation_id,
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation.value,
            'collection': self.collection,
            'document_id': self.document_id,
            'execution_time_ms': self.execution_time_ms,
            'records_affected': self.records_affected,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'success': self.success,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'metadata': self.metadata
        }


@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    
    # Logging behavior
    audit_level: AuditLevel = AuditLevel.STANDARD
    log_sensitive_data: bool = False
    log_query_details: bool = True
    log_performance_metrics: bool = True
    
    # Storage settings
    buffer_size: int = 1000
    flush_interval: int = 60  # seconds
    max_retention_days: int = 90
    
    # Output destinations
    log_to_file: bool = True
    log_file_path: str = "logs/database_audit.log"
    log_to_database: bool = False
    audit_database_config: Optional[Dict[str, Any]] = None
    
    # Security settings
    encrypt_audit_logs: bool = True
    mask_sensitive_fields: bool = True
    
    # Performance settings
    async_logging: bool = True
    batch_size: int = 100


class AuditLogger:
    """
    Comprehensive audit logger for database operations.
    
    Provides structured logging with trace IDs, performance metrics,
    and security context for compliance and debugging.
    """
    
    def __init__(self, config: AuditConfig):
        """
        Initialize audit logger.
        
        Args:
            config: Audit configuration
        """
        self.config = config
        self._event_buffer: deque = deque(maxlen=config.buffer_size)
        self._lock = threading.Lock()
        self._flush_task = None
        self._running = False
        
        # Setup structured logger
        self._setup_logger()
        
        # Start background flushing if async logging is enabled
        if config.async_logging:
            self._start_background_flush()
    
    def _setup_logger(self) -> None:
        """Setup structured logging configuration."""
        self.audit_logger = logging.getLogger(f"{__name__}.audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Remove default handlers to avoid duplicates
        self.audit_logger.handlers.clear()
        
        # Add file handler if configured
        if self.config.log_to_file:
            try:
                import os
                os.makedirs(os.path.dirname(self.config.log_file_path), exist_ok=True)
                
                file_handler = logging.FileHandler(self.config.log_file_path)
                file_handler.setLevel(logging.INFO)
                
                # Use JSON formatter for structured logs
                formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                    '"logger": "%(name)s", "message": %(message)s}'
                )
                file_handler.setFormatter(formatter)
                self.audit_logger.addHandler(file_handler)
                
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
        
        # Add console handler for important events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.audit_logger.addHandler(console_handler)
    
    def _start_background_flush(self) -> None:
        """Start background task for periodic log flushing."""
        self._running = True
        
        async def flush_loop():
            while self._running:
                try:
                    await asyncio.sleep(self.config.flush_interval)
                    await self._flush_buffer()
                except Exception as e:
                    logger.error(f"Error in audit flush loop: {e}")
        
        # Start the flush task
        try:
            loop = asyncio.get_event_loop()
            self._flush_task = loop.create_task(flush_loop())
        except RuntimeError:
            # No event loop running, will flush synchronously
            logger.info("No event loop found, using synchronous audit logging")
    
    async def log_operation(
        self,
        operation: OperationType,
        collection: str,
        document_id: Optional[str] = None,
        execution_time_ms: float = 0.0,
        records_affected: int = 0,
        success: bool = True,
        error: Optional[Exception] = None,
        user_context: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> str:
        """
        Log a database operation.
        
        Args:
            operation: Type of operation
            collection: Collection/table name
            document_id: Document ID if applicable
            execution_time_ms: Operation execution time
            records_affected: Number of records affected
            success: Whether operation succeeded
            error: Exception if operation failed
            user_context: User security context
            metadata: Additional operation metadata
            trace_id: Optional trace ID (will generate if not provided)
            
        Returns:
            Generated trace ID for the operation
        """
        # Generate trace and operation IDs
        trace_id = trace_id or str(uuid.uuid4())
        operation_id = str(uuid.uuid4())
        
        # Extract user context
        user_context = user_context or {}
        
        # Handle error information
        error_message = None
        error_type = None
        if error:
            error_message = str(error)
            error_type = type(error).__name__
            success = False
        
        # Create audit event
        event = AuditEvent(
            trace_id=trace_id,
            operation_id=operation_id,
            operation=operation,
            collection=collection,
            document_id=document_id,
            execution_time_ms=execution_time_ms,
            records_affected=records_affected,
            success=success,
            error_message=error_message,
            error_type=error_type,
            user_id=user_context.get('user_id'),
            session_id=user_context.get('session_id'),
            ip_address=user_context.get('ip_address'),
            user_agent=user_context.get('user_agent'),
            metadata=self._sanitize_metadata(metadata or {})
        )
        
        # Log event based on configuration
        await self._log_event(event)
        
        return trace_id
    
    async def log_query(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
        result_count: int = 0,
        execution_time_ms: float = 0.0,
        success: bool = True,
        error: Optional[Exception] = None,
        user_context: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None
    ) -> str:
        """
        Log a database query operation.
        
        Args:
            collection: Collection/table name
            filters: Query filters (will be sanitized)
            result_count: Number of results returned
            execution_time_ms: Query execution time
            success: Whether query succeeded
            error: Exception if query failed
            user_context: User security context
            trace_id: Optional trace ID
            
        Returns:
            Generated trace ID for the query
        """
        metadata = {}
        
        if self.config.log_query_details and filters:
            # Sanitize filters to remove sensitive data
            metadata['query_filters'] = self._sanitize_query_filters(filters)
        
        return await self.log_operation(
            operation=OperationType.QUERY,
            collection=collection,
            execution_time_ms=execution_time_ms,
            records_affected=result_count,
            success=success,
            error=error,
            user_context=user_context,
            metadata=metadata,
            trace_id=trace_id
        )
    
    async def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        user_context: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a security-related event.
        
        Args:
            event_type: Type of security event
            description: Event description
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            user_context: User security context
            metadata: Additional event metadata
            
        Returns:
            Generated trace ID for the event
        """
        security_metadata = {
            'security_event': True,
            'event_type': event_type,
            'description': description,
            'severity': severity,
            **(metadata or {})
        }
        
        # Log as a system operation
        trace_id = await self.log_operation(
            operation=OperationType.READ,  # Using READ as placeholder
            collection="security_events",
            user_context=user_context,
            metadata=security_metadata
        )
        
        # Also log to security logger if severity is high
        if severity in ['ERROR', 'CRITICAL']:
            self.audit_logger.error(
                json.dumps({
                    'trace_id': trace_id,
                    'security_event': event_type,
                    'description': description,
                    'severity': severity,
                    'user_context': user_context,
                    'metadata': metadata
                })
            )
        
        return trace_id
    
    async def _log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event based on configuration.
        
        Args:
            event: Audit event to log
        """
        # Check if we should log this event based on level
        if not self._should_log_event(event):
            return
        
        if self.config.async_logging:
            # Add to buffer for batch processing
            with self._lock:
                self._event_buffer.append(event)
        else:
            # Log immediately
            await self._write_event(event)
    
    def _should_log_event(self, event: AuditEvent) -> bool:
        """
        Determine if event should be logged based on configuration.
        
        Args:
            event: Audit event
            
        Returns:
            True if event should be logged
        """
        if self.config.audit_level == AuditLevel.MINIMAL:
            # Only log errors and security events
            return not event.success or event.metadata.get('security_event', False)
        
        # All other levels log all events
        return True
    
    async def _write_event(self, event: AuditEvent) -> None:
        """
        Write audit event to configured destinations.
        
        Args:
            event: Audit event to write
        """
        try:
            event_dict = event.to_dict()
            
            # Log to structured logger
            self.audit_logger.info(json.dumps(event_dict))
            
            # Log to database if configured
            if self.config.log_to_database and self.config.audit_database_config:
                # TODO: Implement database logging
                pass
                
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
    
    async def _flush_buffer(self) -> None:
        """Flush buffered audit events."""
        if not self._event_buffer:
            return
        
        # Get events from buffer
        events_to_flush = []
        with self._lock:
            while self._event_buffer and len(events_to_flush) < self.config.batch_size:
                events_to_flush.append(self._event_buffer.popleft())
        
        # Write events
        for event in events_to_flush:
            await self._write_event(event)
        
        if events_to_flush:
            logger.debug(f"Flushed {len(events_to_flush)} audit events")
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to remove sensitive information.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Sanitized metadata
        """
        if not self.config.mask_sensitive_fields:
            return metadata
        
        sanitized = {}
        sensitive_keys = {
            'password', 'secret', 'token', 'key', 'auth',
            'ssn', 'social_security_number', 'credit_card'
        }
        
        for key, value in metadata.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[MASKED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_metadata(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_query_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize query filters for logging.
        
        Args:
            filters: Original query filters
            
        Returns:
            Sanitized filters
        """
        # For now, just mask the values to show structure without data
        if not self.config.log_sensitive_data:
            sanitized = {}
            for key, value in filters.items():
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_query_filters(value)
                else:
                    sanitized[key] = f"[{type(value).__name__}]"
            return sanitized
        
        return filters
    
    async def get_audit_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary of audit events for analysis.
        
        Args:
            start_time: Start time for summary
            end_time: End time for summary
            collection: Filter by collection
            
        Returns:
            Audit summary statistics
        """
        # TODO: Implement audit summary from logs/database
        # This would typically query stored audit logs
        
        return {
            'summary_period': {
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None
            },
            'collection_filter': collection,
            'message': 'Audit summary not yet implemented'
        }
    
    async def close(self) -> None:
        """Close audit logger and cleanup resources."""
        self._running = False
        
        # Flush remaining events
        await self._flush_buffer()
        
        # Cancel background task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Audit logger closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        return {
            'config': {
                'audit_level': self.config.audit_level.value,
                'async_logging': self.config.async_logging,
                'buffer_size': self.config.buffer_size,
                'log_to_file': self.config.log_to_file,
                'log_to_database': self.config.log_to_database
            },
            'runtime': {
                'buffered_events': len(self._event_buffer),
                'background_flush_running': self._running
            }
        }
