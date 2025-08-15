"""
Metrics Collection System
========================

Enterprise metrics collection with structured logging and time-series support.

PHASE3-REFACTOR: Comprehensive metrics collection for analytics dashboard.

Features:
- Structured metric collection
- Time-series data support
- Real-time aggregation
- Multi-dimensional metrics
- Tenant-aware collection
- Performance monitoring
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"          # Monotonically increasing
    GAUGE = "gauge"             # Point-in-time value
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"            # Duration measurements
    RATE = "rate"              # Events per time unit


class MetricUnit(Enum):
    """Metric units for proper display."""
    COUNT = "count"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    BYTES = "bytes"
    PERCENT = "percent"
    REQUESTS_PER_SECOND = "req/s"
    ERRORS_PER_MINUTE = "errors/min"


@dataclass
class MetricDimension:
    """Metric dimension for grouping and filtering."""
    name: str
    value: str


@dataclass
class Metric:
    """
    Individual metric data point.
    
    PHASE3-REFACTOR: Structured metric with dimensions and metadata.
    """
    name: str
    value: Union[int, float]
    metric_type: MetricType
    unit: MetricUnit = MetricUnit.COUNT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Dimensions for grouping/filtering
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    channel: Optional[str] = None
    dimensions: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for storage."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "unit": self.unit.value,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "channel": self.channel,
            "dimensions": self.dimensions,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """Create metric from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            metric_type=MetricType(data["type"]),
            unit=MetricUnit(data.get("unit", "count")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tenant_id=data.get("tenant_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            channel=data.get("channel"),
            dimensions=data.get("dimensions", {}),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class TimeSeriesMetric:
    """Time series aggregated metric data."""
    name: str
    timestamps: List[datetime]
    values: List[float]
    dimensions: Dict[str, str] = field(default_factory=dict)
    aggregation: str = "avg"  # avg, sum, min, max, count
    
    def get_latest_value(self) -> Optional[float]:
        """Get latest metric value."""
        return self.values[-1] if self.values else None
    
    def get_trend(self, periods: int = 5) -> str:
        """Get trend direction over last N periods."""
        if len(self.values) < periods:
            return "insufficient_data"
        
        recent_values = self.values[-periods:]
        if len(recent_values) < 2:
            return "stable"
        
        # Simple linear trend
        slope = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


class MetricAggregator:
    """Real-time metric aggregation."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes default
        self.window_size = window_size
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_cleanup = time.time()
    
    def add_metric(self, metric: Metric):
        """Add metric to aggregation window."""
        key = f"{metric.name}:{metric.tenant_id or 'global'}"
        self.metric_windows[key].append((metric.timestamp, metric.value))
        
        # Periodic cleanup
        current_time = time.time()
        if current_time - self.last_cleanup > 60:  # Cleanup every minute
            self._cleanup_old_metrics()
            self.last_cleanup = current_time
    
    def _cleanup_old_metrics(self):
        """Remove metrics outside window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_size)
        
        for key, window in self.metric_windows.items():
            # Remove old entries
            while window and window[0][0] < cutoff_time:
                window.popleft()
    
    def get_aggregated_value(self, metric_name: str, tenant_id: Optional[str] = None,
                           aggregation: str = "avg") -> Optional[float]:
        """Get aggregated value for metric."""
        key = f"{metric_name}:{tenant_id or 'global'}"
        window = self.metric_windows.get(key, deque())
        
        if not window:
            return None
        
        values = [value for _, value in window]
        
        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        else:
            return statistics.mean(values)


class MetricsCollector:
    """
    Central metrics collection service.
    
    PHASE3-REFACTOR: Enterprise metrics collection with real-time aggregation.
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metric_buffer: deque = deque(maxlen=buffer_size)
        self.aggregator = MetricAggregator()
        
        # Metric stores (will be injected)
        self.stores: List[Any] = []  # IAnalyticsStore instances
        
        # Background processing
        self.flush_task: Optional[asyncio.Task] = None
        self.flush_interval = 10  # seconds
        
        # Statistics
        self.metrics_collected = 0
        self.metrics_flushed = 0
        self.collection_errors = 0
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background metric processing."""
        self.flush_task = asyncio.create_task(self._flush_loop())
    
    def add_store(self, store):
        """Add analytics store for persistence."""
        self.stores.append(store)
    
    def collect_metric(self, name: str, value: Union[int, float], 
                      metric_type: MetricType = MetricType.GAUGE,
                      unit: MetricUnit = MetricUnit.COUNT,
                      tenant_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      channel: Optional[str] = None,
                      dimensions: Optional[Dict[str, str]] = None,
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Collect individual metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                unit=unit,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                channel=channel,
                dimensions=dimensions or {},
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.metric_buffer.append(metric)
            
            # Add to real-time aggregator
            self.aggregator.add_metric(metric)
            
            self.metrics_collected += 1
            
        except Exception as e:
            self.collection_errors += 1
            logger.error(f"Metric collection error: {e}")
    
    def increment_counter(self, name: str, value: int = 1, **kwargs):
        """Increment counter metric."""
        self.collect_metric(name, value, MetricType.COUNTER, **kwargs)
    
    def set_gauge(self, name: str, value: Union[int, float], **kwargs):
        """Set gauge metric value."""
        self.collect_metric(name, value, MetricType.GAUGE, **kwargs)
    
    def record_timer(self, name: str, duration_ms: float, **kwargs):
        """Record timer metric."""
        self.collect_metric(name, duration_ms, MetricType.TIMER, MetricUnit.MILLISECONDS, **kwargs)
    
    def record_histogram(self, name: str, value: float, **kwargs):
        """Record histogram metric."""
        self.collect_metric(name, value, MetricType.HISTOGRAM, **kwargs)
    
    async def flush_metrics(self) -> int:
        """Flush buffered metrics to stores."""
        if not self.metric_buffer or not self.stores:
            return 0
        
        # Get metrics to flush
        metrics_to_flush = list(self.metric_buffer)
        self.metric_buffer.clear()
        
        flushed_count = 0
        
        # Flush to all stores
        for store in self.stores:
            try:
                success = await store.store_metrics(metrics_to_flush)
                if success:
                    flushed_count = len(metrics_to_flush)
            except Exception as e:
                logger.error(f"Error flushing metrics to store: {e}")
        
        self.metrics_flushed += flushed_count
        return flushed_count
    
    async def _flush_loop(self):
        """Background metric flushing loop."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_metrics()
            except Exception as e:
                logger.error(f"Metrics flush loop error: {e}")
    
    def get_real_time_value(self, metric_name: str, tenant_id: Optional[str] = None,
                          aggregation: str = "avg") -> Optional[float]:
        """Get real-time aggregated metric value."""
        return self.aggregator.get_aggregated_value(metric_name, tenant_id, aggregation)
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "metrics_collected": self.metrics_collected,
            "metrics_flushed": self.metrics_flushed,
            "collection_errors": self.collection_errors,
            "buffer_size": len(self.metric_buffer),
            "buffer_capacity": self.buffer_size,
            "connected_stores": len(self.stores)
        }
    
    async def shutdown(self):
        """Shutdown metrics collector."""
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_metrics()
        
        logger.info("Metrics collector shutdown complete")


# Context managers for timing operations
class MetricTimer:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, **kwargs):
        self.collector = collector
        self.metric_name = metric_name
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.metric_name, duration_ms, **self.kwargs)


class AsyncMetricTimer:
    """Async context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, **kwargs):
        self.collector = collector
        self.metric_name = metric_name
        self.kwargs = kwargs
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.metric_name, duration_ms, **self.kwargs)


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


async def initialize_metrics_collector(buffer_size: int = 1000):
    """Initialize global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(buffer_size)
    logger.info("Global metrics collector initialized")


async def shutdown_metrics_collector():
    """Shutdown global metrics collector."""
    global _metrics_collector
    if _metrics_collector:
        await _metrics_collector.shutdown()
        _metrics_collector = None
    logger.info("Global metrics collector shutdown")


# Convenience functions
def collect_metric(name: str, value: Union[int, float], **kwargs):
    """Collect metric using global collector."""
    collector = get_metrics_collector()
    collector.collect_metric(name, value, **kwargs)


def increment_counter(name: str, value: int = 1, **kwargs):
    """Increment counter using global collector."""
    collector = get_metrics_collector()
    collector.increment_counter(name, value, **kwargs)


def set_gauge(name: str, value: Union[int, float], **kwargs):
    """Set gauge using global collector."""
    collector = get_metrics_collector()
    collector.set_gauge(name, value, **kwargs)


def record_timer(name: str, duration_ms: float, **kwargs):
    """Record timer using global collector."""
    collector = get_metrics_collector()
    collector.record_timer(name, duration_ms, **kwargs)


def timer(metric_name: str, **kwargs):
    """Timer context manager."""
    return MetricTimer(get_metrics_collector(), metric_name, **kwargs)


def async_timer(metric_name: str, **kwargs):
    """Async timer context manager."""
    return AsyncMetricTimer(get_metrics_collector(), metric_name, **kwargs)


# Decorators for automatic metric collection
def measure_time(metric_name: str, **metric_kwargs):
    """Decorator to measure function execution time."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with async_timer(metric_name, **metric_kwargs):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with timer(metric_name, **metric_kwargs):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def count_calls(metric_name: str, **metric_kwargs):
    """Decorator to count function calls."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                increment_counter(metric_name, **metric_kwargs)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                increment_counter(metric_name, **metric_kwargs)
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator
