"""
Analytics Module
===============

Enterprise analytics system with real-time monitoring and dashboard capabilities.

PHASE3-REFACTOR: Complete analytics infrastructure for enterprise monitoring.

Components:
- MetricsCollector: Real-time metric collection and aggregation
- AnalyticsStore: Persistent storage with multiple backend support
- DashboardAPI: RESTful APIs for dashboard management
- RealTimeMonitor: WebSocket-based real-time monitoring with alerts

Features:
- Multi-tenant metric isolation
- Real-time data streaming
- Configurable dashboards
- Alert system with rules
- Multiple storage backends
- Export capabilities
"""

# Import main components
from .metrics_collector import (
    MetricsCollector,
    Metric,
    MetricType,
    MetricUnit,
    TimeSeriesMetric,
    MetricAggregator,
    get_metrics_collector,
    initialize_metrics_collector,
    shutdown_metrics_collector,
    collect_metric,
    increment_counter,
    set_gauge,
    record_timer,
    timer,
    async_timer,
    measure_time,
    count_calls
)

from .analytics_store import (
    IAnalyticsStore,
    MemoryAnalyticsStore,
    PostgreSQLAnalyticsStore,
    AnalyticsStoreFactory,
    StorageBackend,
    RetentionPolicy,
    QueryFilter,
    AggregationQuery,
    get_analytics_store,
    initialize_analytics_store,
    set_analytics_store
)

from .dashboard_api import (
    DashboardAPI,
    DashboardConfig,
    DashboardWidget,
    ChartType,
    TimeRange,
    get_dashboard_api,
    initialize_dashboard_api,
    shutdown_dashboard_api
)

from .real_time_monitor import (
    RealTimeMonitor,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertType,
    MetricEvent,
    MetricBuffer,
    get_realtime_monitor,
    initialize_realtime_monitor,
    shutdown_realtime_monitor,
    process_metric_event,
    add_alert_rule
)

# Version info
__version__ = "3.0.0"
__phase__ = "PHASE3"

# Export all for easy imports
__all__ = [
    # Metrics Collection
    "MetricsCollector",
    "Metric", 
    "MetricType",
    "MetricUnit",
    "TimeSeriesMetric",
    "MetricAggregator",
    "get_metrics_collector",
    "initialize_metrics_collector", 
    "shutdown_metrics_collector",
    "collect_metric",
    "increment_counter",
    "set_gauge", 
    "record_timer",
    "timer",
    "async_timer",
    "measure_time",
    "count_calls",
    
    # Analytics Storage
    "IAnalyticsStore",
    "MemoryAnalyticsStore",
    "PostgreSQLAnalyticsStore", 
    "AnalyticsStoreFactory",
    "StorageBackend",
    "RetentionPolicy",
    "QueryFilter",
    "AggregationQuery",
    "get_analytics_store",
    "initialize_analytics_store",
    "set_analytics_store",
    
    # Dashboard API
    "DashboardAPI",
    "DashboardConfig",
    "DashboardWidget",
    "ChartType", 
    "TimeRange",
    "get_dashboard_api",
    "initialize_dashboard_api",
    "shutdown_dashboard_api",
    
    # Real-time Monitoring
    "RealTimeMonitor",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertType", 
    "MetricEvent",
    "MetricBuffer",
    "get_realtime_monitor",
    "initialize_realtime_monitor",
    "shutdown_realtime_monitor",
    "process_metric_event",
    "add_alert_rule"
]
