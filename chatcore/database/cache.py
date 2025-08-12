"""
Cache Manager for Database Layer
===============================

Provides Redis-based caching layer for database operations to improve performance.
Supports multiple caching strategies and cache invalidation patterns.

Features:
- Automatic cache key generation
- TTL (Time To Live) management
- Cache warming and preloading
- Pattern-based invalidation
- Metrics and monitoring
- Fallback handling when Redis is unavailable

Usage:
    cache_manager = CacheManager(redis_config)
    
    # Cache database results
    result = await cache_manager.get_or_set(
        key="user:123",
        fetch_func=lambda: database.get("users", "123"),
        ttl=300
    )
"""

import asyncio
import json
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from .base import DatabaseRecord, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    health_check_interval: int = 30
    decode_responses: bool = True
    
    # Cache behavior
    default_ttl: int = 300  # 5 minutes
    max_ttl: int = 3600     # 1 hour
    key_prefix: str = "chatcore:db:"
    
    # Performance settings
    pipeline_size: int = 100
    compression_threshold: int = 1024  # bytes


@dataclass 
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'errors': self.errors,
            'total_size': self.total_size,
            'hit_rate': self.hit_rate
        }


class CacheManager:
    """
    Redis-based cache manager for database operations.
    
    Provides intelligent caching with automatic key generation,
    TTL management, and fallback handling.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._redis_pool = None
        self._redis_client = None
        self._metrics = CacheMetrics()
        self._connected = False
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - caching disabled")
    
    async def connect(self) -> None:
        """
        Connect to Redis server.
        
        Raises:
            ConnectionError: If connection fails
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - skipping connection")
            return
        
        try:
            # Create connection pool
            self._redis_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses
            )
            
            # Create Redis client
            self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            
            # Test connection
            await self._redis_client.ping()
            self._connected = True
            
            logger.info("Connected to Redis cache server")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            # Don't raise exception - allow graceful fallback
    
    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup resources."""
        if self._redis_client:
            await self._redis_client.close()
        if self._redis_pool:
            await self._redis_pool.disconnect()
        self._connected = False
        logger.info("Disconnected from Redis cache server")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self._connected:
            self._metrics.misses += 1
            return None
        
        try:
            full_key = self._build_key(key)
            cached_data = await self._redis_client.get(full_key)
            
            if cached_data is None:
                self._metrics.misses += 1
                return None
            
            # Deserialize data
            value = self._deserialize(cached_data)
            self._metrics.hits += 1
            
            logger.debug(f"Cache hit for key: {key}")
            return value
            
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            self._metrics.errors += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if set successfully
        """
        if not self._connected:
            return False
        
        try:
            full_key = self._build_key(key)
            serialized_value = self._serialize(value)
            
            # Use default TTL if not specified
            cache_ttl = ttl or self.config.default_ttl
            cache_ttl = min(cache_ttl, self.config.max_ttl)
            
            await self._redis_client.setex(full_key, cache_ttl, serialized_value)
            self._metrics.sets += 1
            
            logger.debug(f"Cache set for key: {key}, TTL: {cache_ttl}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            self._metrics.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted successfully
        """
        if not self._connected:
            return False
        
        try:
            full_key = self._build_key(key)
            result = await self._redis_client.delete(full_key)
            self._metrics.deletes += 1
            
            logger.debug(f"Cache delete for key: {key}")
            return result > 0
            
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            self._metrics.errors += 1
            return False
    
    async def get_or_set(
        self,
        key: str,
        fetch_func: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get value from cache or fetch and cache it.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data if not in cache
            ttl: Time to live in seconds
            
        Returns:
            Cached or fetched value
        """
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Fetch from source
        try:
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func()
            else:
                value = fetch_func()
            
            # Cache the result
            if value is not None:
                await self.set(key, value, ttl)
            
            return value
            
        except Exception as e:
            logger.error(f"Error fetching data for cache key {key}: {e}")
            return None
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache keys matching a pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            return 0
        
        try:
            full_pattern = self._build_key(pattern)
            keys = await self._redis_client.keys(full_pattern)
            
            if keys:
                deleted = await self._redis_client.delete(*keys)
                self._metrics.deletes += deleted
                logger.info(f"Invalidated {deleted} cache keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Cache pattern invalidation error for {pattern}: {e}")
            self._metrics.errors += 1
            return 0
    
    def build_query_key(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> str:
        """
        Build cache key for database query.
        
        Args:
            collection: Collection name
            filters: Query filters
            sort_by: Sort field
            limit: Record limit
            offset: Record offset
            
        Returns:
            Cache key for the query
        """
        # Create a deterministic key from query parameters
        key_parts = [f"query:{collection}"]
        
        if filters:
            # Sort filters for consistent keys
            sorted_filters = json.dumps(filters, sort_keys=True, default=str)
            filters_hash = hashlib.md5(sorted_filters.encode()).hexdigest()[:8]
            key_parts.append(f"filters:{filters_hash}")
        
        if sort_by:
            key_parts.append(f"sort:{sort_by}")
        
        if limit:
            key_parts.append(f"limit:{limit}")
        
        if offset > 0:
            key_parts.append(f"offset:{offset}")
        
        return ":".join(key_parts)
    
    def build_record_key(self, collection: str, document_id: str) -> str:
        """
        Build cache key for individual record.
        
        Args:
            collection: Collection name
            document_id: Document ID
            
        Returns:
            Cache key for the record
        """
        return f"record:{collection}:{document_id}"
    
    async def cache_query_result(
        self,
        collection: str,
        result: QueryResult,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache query result.
        
        Args:
            collection: Collection name
            result: Query result to cache
            filters: Query filters
            sort_by: Sort field
            limit: Record limit
            offset: Record offset
            ttl: Time to live
        """
        query_key = self.build_query_key(collection, filters, sort_by, limit, offset)
        await self.set(query_key, result, ttl)
        
        # Also cache individual records
        for record in result.data:
            if '_id' in record or 'id' in record:
                doc_id = record.get('_id') or record.get('id')
                record_key = self.build_record_key(collection, str(doc_id))
                await self.set(record_key, record, ttl)
    
    async def invalidate_collection(self, collection: str) -> int:
        """
        Invalidate all cache entries for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Number of keys deleted
        """
        patterns = [
            f"query:{collection}:*",
            f"record:{collection}:*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = await self.invalidate_pattern(pattern)
            total_deleted += deleted
        
        return total_deleted
    
    async def warm_cache(
        self,
        collection: str,
        records: List[DatabaseRecord],
        ttl: Optional[int] = None
    ) -> None:
        """
        Pre-warm cache with records.
        
        Args:
            collection: Collection name
            records: Records to cache
            ttl: Time to live
        """
        tasks = []
        for record in records:
            if '_id' in record or 'id' in record:
                doc_id = record.get('_id') or record.get('id')
                record_key = self.build_record_key(collection, str(doc_id))
                task = self.set(record_key, record, ttl)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Warmed cache with {len(tasks)} records for collection: {collection}")
    
    def _build_key(self, key: str) -> str:
        """
        Build full cache key with prefix.
        
        Args:
            key: Base key
            
        Returns:
            Full cache key
        """
        return f"{self.config.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> str:
        """
        Serialize value for storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized string
        """
        if isinstance(value, QueryResult):
            # Special handling for QueryResult objects
            serialized = {
                '__type__': 'QueryResult',
                'data': value.data,
                'total_count': value.total_count,
                'page': value.page,
                'page_size': value.page_size,
                'has_more': value.has_more,
                'execution_time_ms': value.execution_time_ms,
                'query_id': value.query_id,
                'metadata': value.metadata
            }
            return json.dumps(serialized, default=str)
        
        return json.dumps(value, default=str)
    
    def _deserialize(self, data: str) -> Any:
        """
        Deserialize value from storage.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized value
        """
        try:
            parsed = json.loads(data)
            
            # Special handling for QueryResult objects
            if isinstance(parsed, dict) and parsed.get('__type__') == 'QueryResult':
                return QueryResult(
                    data=parsed['data'],
                    total_count=parsed['total_count'],
                    page=parsed['page'],
                    page_size=parsed['page_size'],
                    has_more=parsed['has_more'],
                    execution_time_ms=parsed['execution_time_ms'],
                    query_id=parsed['query_id'],
                    metadata=parsed['metadata']
                )
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to deserialize cache data: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.
        
        Returns:
            Health status and metrics
        """
        if not self._connected:
            return {
                'status': 'disconnected',
                'redis_available': REDIS_AVAILABLE,
                'metrics': self._metrics.to_dict()
            }
        
        try:
            # Test Redis connection
            await self._redis_client.ping()
            
            # Get Redis info
            info = await self._redis_client.info()
            
            return {
                'status': 'healthy',
                'redis_available': True,
                'connected': True,
                'metrics': self._metrics.to_dict(),
                'redis_info': {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_commands_processed': info.get('total_commands_processed')
                }
            }
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self._metrics.to_dict()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self._metrics.to_dict()
    
    @property
    def is_connected(self) -> bool:
        """Check if cache is connected."""
        return self._connected
