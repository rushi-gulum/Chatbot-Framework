"""
Analytics Data Store
==================

Persistent storage for analytics data with multiple backend support.

PHASE3-REFACTOR: Enterprise analytics storage with time-series optimization.

Features:
- Multiple storage backends
- Time-series optimization
- Automatic data retention
- Batch operations
- Query optimization
- Data compression
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends."""
    MEMORY = "memory"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    INFLUXDB = "influxdb"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"


class RetentionPolicy(Enum):
    """Data retention policies."""
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1m"
    QUARTER = "3m"
    YEAR = "1y"
    FOREVER = "forever"


@dataclass
class QueryFilter:
    """Query filter for analytics data."""
    metric_names: Optional[List[str]] = None
    tenant_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dimensions: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class AggregationQuery:
    """Aggregation query parameters."""
    metric_name: str
    aggregation: str = "avg"  # avg, sum, min, max, count, p50, p95, p99
    group_by: Optional[List[str]] = None  # Fields to group by
    time_bucket: Optional[str] = None  # 1m, 5m, 1h, 1d, etc.
    filters: Optional[QueryFilter] = None


class IAnalyticsStore(ABC):
    """Interface for analytics storage backends."""
    
    @abstractmethod
    async def store_metrics(self, metrics: List[Any]) -> bool:
        """Store list of metrics."""
        pass
    
    @abstractmethod
    async def query_metrics(self, filters: QueryFilter) -> List[Dict[str, Any]]:
        """Query metrics with filters."""
        pass
    
    @abstractmethod
    async def aggregate_metrics(self, query: AggregationQuery) -> List[Dict[str, Any]]:
        """Aggregate metrics."""
        pass
    
    @abstractmethod
    async def get_time_series(self, metric_name: str, 
                            start_time: datetime, end_time: datetime,
                            tenant_id: Optional[str] = None,
                            group_by: Optional[List[str]] = None,
                            time_bucket: str = "1h") -> List[Dict[str, Any]]:
        """Get time series data."""
        pass
    
    @abstractmethod
    async def cleanup_old_data(self, retention_policy: RetentionPolicy) -> int:
        """Clean up old data based on retention policy."""
        pass
    
    @abstractmethod
    async def get_store_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


class MemoryAnalyticsStore(IAnalyticsStore):
    """In-memory analytics store for testing/development."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.max_size = 10000
    
    async def store_metrics(self, metrics: List[Any]) -> bool:
        """Store metrics in memory."""
        try:
            for metric in metrics:
                if hasattr(metric, 'to_dict'):
                    self.metrics.append(metric.to_dict())
                else:
                    self.metrics.append(metric)
            
            # Trim if too large
            if len(self.metrics) > self.max_size:
                self.metrics = self.metrics[-self.max_size:]
            
            return True
        except Exception as e:
            logger.error(f"Memory store error: {e}")
            return False
    
    async def query_metrics(self, filters: QueryFilter) -> List[Dict[str, Any]]:
        """Query metrics from memory."""
        results = self.metrics.copy()
        
        # Apply filters
        if filters.metric_names:
            results = [m for m in results if m.get("name") in filters.metric_names]
        
        if filters.tenant_ids:
            results = [m for m in results if m.get("tenant_id") in filters.tenant_ids]
        
        if filters.start_time:
            results = [m for m in results 
                      if datetime.fromisoformat(m["timestamp"]) >= filters.start_time]
        
        if filters.end_time:
            results = [m for m in results 
                      if datetime.fromisoformat(m["timestamp"]) <= filters.end_time]
        
        # Apply limit/offset
        if filters.offset:
            results = results[filters.offset:]
        
        if filters.limit:
            results = results[:filters.limit]
        
        return results
    
    async def aggregate_metrics(self, query: AggregationQuery) -> List[Dict[str, Any]]:
        """Aggregate metrics in memory."""
        # Simple aggregation for memory store
        filters = query.filters or QueryFilter()
        filters.metric_names = [query.metric_name]
        
        metrics = await self.query_metrics(filters)
        
        if not metrics:
            return []
        
        values = [m["value"] for m in metrics]
        
        if query.aggregation == "sum":
            result_value = sum(values)
        elif query.aggregation == "avg":
            result_value = sum(values) / len(values)
        elif query.aggregation == "min":
            result_value = min(values)
        elif query.aggregation == "max":
            result_value = max(values)
        elif query.aggregation == "count":
            result_value = len(values)
        else:
            result_value = sum(values) / len(values)
        
        return [{
            "metric_name": query.metric_name,
            "aggregation": query.aggregation,
            "value": result_value,
            "count": len(values),
            "timestamp": datetime.utcnow().isoformat()
        }]
    
    async def get_time_series(self, metric_name: str, 
                            start_time: datetime, end_time: datetime,
                            tenant_id: Optional[str] = None,
                            group_by: Optional[List[str]] = None,
                            time_bucket: str = "1h") -> List[Dict[str, Any]]:
        """Get time series from memory."""
        filters = QueryFilter(
            metric_names=[metric_name],
            tenant_ids=[tenant_id] if tenant_id else None,
            start_time=start_time,
            end_time=end_time
        )
        
        metrics = await self.query_metrics(filters)
        
        # Simple time bucketing
        buckets = {}
        for metric in metrics:
            timestamp = datetime.fromisoformat(metric["timestamp"])
            
            # Round to bucket
            if time_bucket == "1h":
                bucket_time = timestamp.replace(minute=0, second=0, microsecond=0)
            elif time_bucket == "1d":
                bucket_time = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket_time = timestamp.replace(minute=0, second=0, microsecond=0)
            
            bucket_key = bucket_time.isoformat()
            
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(metric["value"])
        
        # Aggregate buckets
        results = []
        for bucket_time, values in buckets.items():
            results.append({
                "timestamp": bucket_time,
                "value": sum(values) / len(values),
                "count": len(values)
            })
        
        return sorted(results, key=lambda x: x["timestamp"])
    
    async def cleanup_old_data(self, retention_policy: RetentionPolicy) -> int:
        """Cleanup old data from memory."""
        if retention_policy == RetentionPolicy.FOREVER:
            return 0
        
        # Calculate cutoff time
        now = datetime.utcnow()
        if retention_policy == RetentionPolicy.HOUR:
            cutoff = now - timedelta(hours=1)
        elif retention_policy == RetentionPolicy.DAY:
            cutoff = now - timedelta(days=1)
        elif retention_policy == RetentionPolicy.WEEK:
            cutoff = now - timedelta(weeks=1)
        elif retention_policy == RetentionPolicy.MONTH:
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=1)
        
        original_count = len(self.metrics)
        self.metrics = [m for m in self.metrics 
                       if datetime.fromisoformat(m["timestamp"]) >= cutoff]
        
        return original_count - len(self.metrics)
    
    async def get_store_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            "backend": "memory",
            "total_metrics": len(self.metrics),
            "max_size": self.max_size,
            "memory_usage_mb": len(json.dumps(self.metrics)) / (1024 * 1024)
        }


class PostgreSQLAnalyticsStore(IAnalyticsStore):
    """PostgreSQL analytics store with time-series optimization."""
    
    def __init__(self, connection_string: str, table_name: str = "analytics_metrics"):
        self.connection_string = connection_string
        self.table_name = table_name
        self.pool = None
    
    async def _get_pool(self):
        """Get connection pool."""
        if self.pool is None:
            try:
                import asyncpg
                self.pool = await asyncpg.create_pool(self.connection_string)
                await self._create_tables()
            except ImportError:
                logger.error("asyncpg not installed. Install with: pip install asyncpg")
                raise
        return self.pool
    
    async def _create_tables(self):
        """Create analytics tables if they don't exist."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Main metrics table with time-series optimization
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    value FLOAT NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    unit VARCHAR(50) DEFAULT 'count',
                    timestamp TIMESTAMPTZ NOT NULL,
                    tenant_id VARCHAR(255),
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    channel VARCHAR(100),
                    dimensions JSONB,
                    tags TEXT[],
                    metadata JSONB
                );
            """)
            
            # Indexes for performance
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
                ON {self.table_name} (timestamp);
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_name_tenant 
                ON {self.table_name} (name, tenant_id);
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tenant_timestamp 
                ON {self.table_name} (tenant_id, timestamp);
            """)
    
    async def store_metrics(self, metrics: List[Any]) -> bool:
        """Store metrics in PostgreSQL."""
        try:
            pool = await self._get_pool()
            
            async with pool.acquire() as conn:
                # Prepare batch insert
                values = []
                for metric in metrics:
                    if hasattr(metric, 'to_dict'):
                        data = metric.to_dict()
                    else:
                        data = metric
                    
                    values.append((
                        data["name"],
                        data["value"],
                        data["type"],
                        data.get("unit", "count"),
                        datetime.fromisoformat(data["timestamp"]),
                        data.get("tenant_id"),
                        data.get("user_id"),
                        data.get("session_id"),
                        data.get("channel"),
                        json.dumps(data.get("dimensions", {})),
                        data.get("tags", []),
                        json.dumps(data.get("metadata", {}))
                    ))
                
                # Batch insert
                await conn.executemany(f"""
                    INSERT INTO {self.table_name} 
                    (name, value, type, unit, timestamp, tenant_id, user_id, 
                     session_id, channel, dimensions, tags, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, values)
            
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL store error: {e}")
            return False
    
    async def query_metrics(self, filters: QueryFilter) -> List[Dict[str, Any]]:
        """Query metrics from PostgreSQL."""
        pool = await self._get_pool()
        
        # Build query
        where_clauses = []
        params = []
        param_count = 0
        
        if filters.metric_names:
            param_count += 1
            where_clauses.append(f"name = ANY(${param_count})")
            params.append(filters.metric_names)
        
        if filters.tenant_ids:
            param_count += 1
            where_clauses.append(f"tenant_id = ANY(${param_count})")
            params.append(filters.tenant_ids)
        
        if filters.start_time:
            param_count += 1
            where_clauses.append(f"timestamp >= ${param_count}")
            params.append(filters.start_time)
        
        if filters.end_time:
            param_count += 1
            where_clauses.append(f"timestamp <= ${param_count}")
            params.append(filters.end_time)
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        limit_clause = ""
        if filters.limit:
            param_count += 1
            limit_clause = f"LIMIT ${param_count}"
            params.append(filters.limit)
        
        if filters.offset:
            param_count += 1
            limit_clause += f" OFFSET ${param_count}"
            params.append(filters.offset)
        
        query = f"""
            SELECT name, value, type, unit, timestamp, tenant_id, user_id,
                   session_id, channel, dimensions, tags, metadata
            FROM {self.table_name}
            {where_clause}
            ORDER BY timestamp DESC
            {limit_clause}
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                results.append({
                    "name": row["name"],
                    "value": row["value"],
                    "type": row["type"],
                    "unit": row["unit"],
                    "timestamp": row["timestamp"].isoformat(),
                    "tenant_id": row["tenant_id"],
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "channel": row["channel"],
                    "dimensions": json.loads(row["dimensions"] or "{}"),
                    "tags": row["tags"] or [],
                    "metadata": json.loads(row["metadata"] or "{}")
                })
            
            return results
    
    async def aggregate_metrics(self, query: AggregationQuery) -> List[Dict[str, Any]]:
        """Aggregate metrics in PostgreSQL."""
        pool = await self._get_pool()
        
        # Build aggregation query
        agg_func = query.aggregation.upper()
        if agg_func not in ["AVG", "SUM", "MIN", "MAX", "COUNT"]:
            agg_func = "AVG"
        
        where_clauses = [f"name = $1"]
        params = [query.metric_name]
        param_count = 1
        
        if query.filters:
            if query.filters.tenant_ids:
                param_count += 1
                where_clauses.append(f"tenant_id = ANY(${param_count})")
                params.append(query.filters.tenant_ids)
            
            if query.filters.start_time:
                param_count += 1
                where_clauses.append(f"timestamp >= ${param_count}")
                params.append(query.filters.start_time)
            
            if query.filters.end_time:
                param_count += 1
                where_clauses.append(f"timestamp <= ${param_count}")
                params.append(query.filters.end_time)
        
        where_clause = "WHERE " + " AND ".join(where_clauses)
        
        sql_query = f"""
            SELECT {agg_func}(value) as value, COUNT(*) as count
            FROM {self.table_name}
            {where_clause}
        """
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql_query, *params)
            
            return [{
                "metric_name": query.metric_name,
                "aggregation": query.aggregation,
                "value": float(row["value"]) if row["value"] is not None else 0,
                "count": row["count"],
                "timestamp": datetime.utcnow().isoformat()
            }]
    
    async def get_time_series(self, metric_name: str, 
                            start_time: datetime, end_time: datetime,
                            tenant_id: Optional[str] = None,
                            group_by: Optional[List[str]] = None,
                            time_bucket: str = "1h") -> List[Dict[str, Any]]:
        """Get time series from PostgreSQL."""
        pool = await self._get_pool()
        
        # Time bucket mapping
        bucket_sql = {
            "1m": "date_trunc('minute', timestamp)",
            "5m": "date_trunc('hour', timestamp) + INTERVAL '5 min' * FLOOR(date_part('minute', timestamp) / 5)",
            "1h": "date_trunc('hour', timestamp)",
            "1d": "date_trunc('day', timestamp)"
        }.get(time_bucket, "date_trunc('hour', timestamp)")
        
        where_clauses = ["name = $1", "timestamp >= $2", "timestamp <= $3"]
        params = [metric_name, start_time, end_time]
        param_count = 3
        
        if tenant_id:
            param_count += 1
            where_clauses.append(f"tenant_id = ${param_count}")
            params.append(tenant_id)
        
        where_clause = "WHERE " + " AND ".join(where_clauses)
        
        query = f"""
            SELECT {bucket_sql} as bucket,
                   AVG(value) as value,
                   COUNT(*) as count
            FROM {self.table_name}
            {where_clause}
            GROUP BY bucket
            ORDER BY bucket
        """
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return [{
                "timestamp": row["bucket"].isoformat(),
                "value": float(row["value"]),
                "count": row["count"]
            } for row in rows]
    
    async def cleanup_old_data(self, retention_policy: RetentionPolicy) -> int:
        """Clean up old data from PostgreSQL."""
        if retention_policy == RetentionPolicy.FOREVER:
            return 0
        
        pool = await self._get_pool()
        
        # Calculate cutoff time
        now = datetime.utcnow()
        if retention_policy == RetentionPolicy.HOUR:
            cutoff = now - timedelta(hours=1)
        elif retention_policy == RetentionPolicy.DAY:
            cutoff = now - timedelta(days=1)
        elif retention_policy == RetentionPolicy.WEEK:
            cutoff = now - timedelta(weeks=1)
        elif retention_policy == RetentionPolicy.MONTH:
            cutoff = now - timedelta(days=30)
        elif retention_policy == RetentionPolicy.QUARTER:
            cutoff = now - timedelta(days=90)
        elif retention_policy == RetentionPolicy.YEAR:
            cutoff = now - timedelta(days=365)
        else:
            cutoff = now - timedelta(days=1)
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE timestamp < $1",
                cutoff
            )
            
            # Extract number of deleted rows
            deleted_count = int(result.split()[-1]) if result.split() else 0
            return deleted_count
    
    async def get_store_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL store statistics."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Get total count
            total_row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {self.table_name}")
            total_count = total_row["count"]
            
            # Get oldest and newest timestamps
            time_range_row = await conn.fetchrow(f"""
                SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest 
                FROM {self.table_name}
            """)
            
            return {
                "backend": "postgresql",
                "total_metrics": total_count,
                "oldest_metric": time_range_row["oldest"].isoformat() if time_range_row["oldest"] else None,
                "newest_metric": time_range_row["newest"].isoformat() if time_range_row["newest"] else None,
                "table_name": self.table_name
            }


class AnalyticsStoreFactory:
    """Factory for creating analytics stores."""
    
    @staticmethod
    def create_store(backend: StorageBackend, **kwargs) -> IAnalyticsStore:
        """Create analytics store instance."""
        if backend == StorageBackend.MEMORY:
            return MemoryAnalyticsStore()
        elif backend == StorageBackend.POSTGRESQL:
            connection_string = kwargs.get("connection_string")
            if not connection_string:
                raise ValueError("PostgreSQL connection_string required")
            return PostgreSQLAnalyticsStore(connection_string, kwargs.get("table_name", "analytics_metrics"))
        else:
            raise NotImplementedError(f"Backend {backend.value} not implemented yet")


# Global store instance
_analytics_store: Optional[IAnalyticsStore] = None


def get_analytics_store() -> Optional[IAnalyticsStore]:
    """Get global analytics store."""
    return _analytics_store


async def initialize_analytics_store(backend: StorageBackend, **kwargs):
    """Initialize global analytics store."""
    global _analytics_store
    _analytics_store = AnalyticsStoreFactory.create_store(backend, **kwargs)
    logger.info(f"Analytics store initialized with backend: {backend.value}")


def set_analytics_store(store: IAnalyticsStore):
    """Set global analytics store."""
    global _analytics_store
    _analytics_store = store
