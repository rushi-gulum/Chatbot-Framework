"""
Analytics Dashboard API
======================

RESTful API for analytics dashboard with real-time capabilities.

PHASE3-REFACTOR: Enterprise dashboard APIs with WebSocket support.

Features:
- RESTful metrics API
- Real-time WebSocket updates
- Dashboard configuration
- Custom visualization support
- Export capabilities
- Multi-tenant isolation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Chart types for visualization."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    GAUGE = "gauge"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    TABLE = "table"


class TimeRange(Enum):
    """Predefined time ranges."""
    LAST_HOUR = "1h"
    LAST_DAY = "1d" 
    LAST_WEEK = "1w"
    LAST_MONTH = "1m"
    LAST_QUARTER = "3m"
    LAST_YEAR = "1y"
    CUSTOM = "custom"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    id: str
    title: str
    chart_type: ChartType
    metric_name: str
    aggregation: str = "avg"
    time_range: TimeRange = TimeRange.LAST_DAY
    time_bucket: str = "1h"
    filters: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, int]] = None  # x, y, width, height
    options: Optional[Dict[str, Any]] = None  # Chart-specific options
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "chart_type": self.chart_type.value,
            "metric_name": self.metric_name,
            "aggregation": self.aggregation,
            "time_range": self.time_range.value,
            "time_bucket": self.time_bucket,
            "filters": self.filters or {},
            "position": self.position or {},
            "options": self.options or {}
        }


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    id: str
    name: str
    description: str
    tenant_id: Optional[str] = None
    widgets: Optional[List[DashboardWidget]] = None
    refresh_interval: int = 30  # seconds
    auto_refresh: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.widgets is None:
            self.widgets = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "widgets": [w.to_dict() for w in (self.widgets or [])],
            "refresh_interval": self.refresh_interval,
            "auto_refresh": self.auto_refresh,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DashboardAPI:
    """
    Analytics Dashboard API.
    
    PHASE3-REFACTOR: RESTful API for dashboard management and data access.
    """
    
    def __init__(self, analytics_store, metrics_collector):
        self.analytics_store = analytics_store
        self.metrics_collector = metrics_collector
        self.dashboards: Dict[str, DashboardConfig] = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, List[Any]] = {}  # tenant_id -> [websockets]
        
        # Background task for real-time updates
        self.realtime_task: Optional[asyncio.Task] = None
        self._start_realtime_updates()
    
    def _start_realtime_updates(self):
        """Start real-time update task."""
        self.realtime_task = asyncio.create_task(self._realtime_update_loop())
    
    async def _realtime_update_loop(self):
        """Background loop for real-time updates."""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                await self._send_realtime_updates()
            except Exception as e:
                logger.error(f"Real-time update error: {e}")
    
    async def _send_realtime_updates(self):
        """Send real-time updates to connected clients."""
        for tenant_id, connections in self.websocket_connections.items():
            if not connections:
                continue
            
            # Get real-time metrics for tenant
            updates = await self._get_realtime_metrics(tenant_id)
            
            if updates:
                message = {
                    "type": "metrics_update",
                    "data": updates,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Send to all connections
                disconnected = []
                for ws in connections:
                    try:
                        await ws.send_text(json.dumps(message))
                    except Exception:
                        disconnected.append(ws)
                
                # Remove disconnected clients
                for ws in disconnected:
                    connections.remove(ws)
    
    async def _get_realtime_metrics(self, tenant_id: Optional[str]) -> List[Dict[str, Any]]:
        """Get real-time metrics for tenant."""
        try:
            # Get common metrics
            metrics = [
                "chatbot.messages.total",
                "chatbot.response_time.avg",
                "chatbot.satisfaction.avg",
                "system.cpu.usage",
                "system.memory.usage"
            ]
            
            results = []
            for metric_name in metrics:
                value = self.metrics_collector.get_real_time_value(metric_name, tenant_id)
                if value is not None:
                    results.append({
                        "metric": metric_name,
                        "value": value,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return []
    
    # Dashboard Management
    async def create_dashboard(self, config: DashboardConfig) -> Dict[str, Any]:
        """Create new dashboard."""
        try:
            self.dashboards[config.id] = config
            
            return {
                "success": True,
                "dashboard_id": config.id,
                "dashboard": config.to_dict()
            }
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_dashboard(self, dashboard_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get dashboard configuration."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            return {
                "success": True,
                "dashboard": dashboard.to_dict()
            }
        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any], 
                             tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Update dashboard configuration."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            # Apply updates
            if "name" in updates:
                dashboard.name = updates["name"]
            if "description" in updates:
                dashboard.description = updates["description"]
            if "refresh_interval" in updates:
                dashboard.refresh_interval = updates["refresh_interval"]
            if "auto_refresh" in updates:
                dashboard.auto_refresh = updates["auto_refresh"]
            
            dashboard.updated_at = datetime.utcnow()
            
            return {
                "success": True,
                "dashboard": dashboard.to_dict()
            }
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_dashboard(self, dashboard_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete dashboard."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            del self.dashboards[dashboard_id]
            
            return {"success": True}
        except Exception as e:
            logger.error(f"Error deleting dashboard: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_dashboards(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """List dashboards for tenant."""
        try:
            dashboards = []
            for dashboard in self.dashboards.values():
                # Filter by tenant
                if tenant_id and dashboard.tenant_id != tenant_id:
                    continue
                
                dashboards.append(dashboard.to_dict())
            
            return {
                "success": True,
                "dashboards": dashboards
            }
        except Exception as e:
            logger.error(f"Error listing dashboards: {e}")
            return {"success": False, "error": str(e)}
    
    # Widget Management
    async def add_widget(self, dashboard_id: str, widget: DashboardWidget, 
                        tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Add widget to dashboard."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            if dashboard.widgets is None:
                dashboard.widgets = []
            
            dashboard.widgets.append(widget)
            dashboard.updated_at = datetime.utcnow()
            
            return {
                "success": True,
                "widget": widget.to_dict()
            }
        except Exception as e:
            logger.error(f"Error adding widget: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_widget(self, dashboard_id: str, widget_id: str, 
                           updates: Dict[str, Any], tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Update dashboard widget."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            # Find widget
            widget = None
            if dashboard.widgets:
                for w in dashboard.widgets:
                    if w.id == widget_id:
                        widget = w
                        break
            
            if not widget:
                return {"success": False, "error": "Widget not found"}
            
            # Apply updates
            if "title" in updates:
                widget.title = updates["title"]
            if "metric_name" in updates:
                widget.metric_name = updates["metric_name"]
            if "aggregation" in updates:
                widget.aggregation = updates["aggregation"]
            if "time_range" in updates:
                widget.time_range = TimeRange(updates["time_range"])
            if "position" in updates:
                widget.position = updates["position"]
            
            dashboard.updated_at = datetime.utcnow()
            
            return {
                "success": True,
                "widget": widget.to_dict()
            }
        except Exception as e:
            logger.error(f"Error updating widget: {e}")
            return {"success": False, "error": str(e)}
    
    async def remove_widget(self, dashboard_id: str, widget_id: str, 
                           tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Remove widget from dashboard."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            # Remove widget
            if dashboard.widgets:
                dashboard.widgets = [w for w in dashboard.widgets if w.id != widget_id]
            dashboard.updated_at = datetime.utcnow()
            
            return {"success": True}
        except Exception as e:
            logger.error(f"Error removing widget: {e}")
            return {"success": False, "error": str(e)}
    
    # Data APIs
    async def get_widget_data(self, dashboard_id: str, widget_id: str,
                            tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data for specific widget."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            # Find widget
            widget = None
            if dashboard.widgets:
                for w in dashboard.widgets:
                    if w.id == widget_id:
                        widget = w
                        break
            
            if not widget:
                return {"success": False, "error": "Widget not found"}
            
            # Get data based on widget configuration
            data = await self._get_widget_data(widget, tenant_id)
            
            return {
                "success": True,
                "widget_id": widget_id,
                "data": data
            }
        except Exception as e:
            logger.error(f"Error getting widget data: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_widget_data(self, widget: DashboardWidget, 
                              tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data for widget based on configuration."""
        # Calculate time range
        end_time = datetime.utcnow()
        
        if widget.time_range == TimeRange.LAST_HOUR:
            start_time = end_time - timedelta(hours=1)
        elif widget.time_range == TimeRange.LAST_DAY:
            start_time = end_time - timedelta(days=1)
        elif widget.time_range == TimeRange.LAST_WEEK:
            start_time = end_time - timedelta(weeks=1)
        elif widget.time_range == TimeRange.LAST_MONTH:
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)
        
        if widget.chart_type in [ChartType.LINE, ChartType.BAR]:
            # Time series data
            if self.analytics_store:
                data = await self.analytics_store.get_time_series(
                    widget.metric_name,
                    start_time,
                    end_time,
                    tenant_id,
                    time_bucket=widget.time_bucket
                )
            else:
                data = []
            
            return {
                "type": "time_series",
                "data": data,
                "chart_type": widget.chart_type.value
            }
        
        elif widget.chart_type == ChartType.GAUGE:
            # Single value
            value = self.metrics_collector.get_real_time_value(
                widget.metric_name, tenant_id, widget.aggregation
            )
            
            return {
                "type": "single_value", 
                "value": value or 0,
                "chart_type": "gauge"
            }
        
        else:
            # Default to current value
            value = self.metrics_collector.get_real_time_value(
                widget.metric_name, tenant_id, widget.aggregation
            )
            
            return {
                "type": "single_value",
                "value": value or 0,
                "chart_type": widget.chart_type.value
            }
    
    async def get_dashboard_data(self, dashboard_id: str, 
                               tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data for all widgets in dashboard."""
        try:
            dashboard = self.dashboards.get(dashboard_id)
            if not dashboard:
                return {"success": False, "error": "Dashboard not found"}
            
            # Check tenant access
            if tenant_id and dashboard.tenant_id != tenant_id:
                return {"success": False, "error": "Access denied"}
            
            # Get data for all widgets
            widget_data = {}
            if dashboard.widgets:
                for widget in dashboard.widgets:
                    try:
                        data = await self._get_widget_data(widget, tenant_id)
                        widget_data[widget.id] = data
                    except Exception as e:
                        logger.error(f"Error getting data for widget {widget.id}: {e}")
                        widget_data[widget.id] = {"error": str(e)}
            
            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "widget_data": widget_data
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"success": False, "error": str(e)}
    
    # Metrics APIs
    async def get_available_metrics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get list of available metrics."""
        try:
            # Common metrics available
            metrics = [
                {
                    "name": "chatbot.messages.total",
                    "description": "Total messages processed",
                    "type": "counter",
                    "unit": "count"
                },
                {
                    "name": "chatbot.response_time.avg",
                    "description": "Average response time",
                    "type": "gauge", 
                    "unit": "milliseconds"
                },
                {
                    "name": "chatbot.satisfaction.avg",
                    "description": "Average satisfaction score",
                    "type": "gauge",
                    "unit": "percent"
                },
                {
                    "name": "chatbot.errors.rate",
                    "description": "Error rate",
                    "type": "rate",
                    "unit": "errors/min"
                },
                {
                    "name": "system.cpu.usage",
                    "description": "CPU usage",
                    "type": "gauge",
                    "unit": "percent"
                },
                {
                    "name": "system.memory.usage",
                    "description": "Memory usage",
                    "type": "gauge",
                    "unit": "percent"
                }
            ]
            
            return {
                "success": True,
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Error getting available metrics: {e}")
            return {"success": False, "error": str(e)}
    
    async def export_dashboard_data(self, dashboard_id: str, format: str = "json",
                                  tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Export dashboard data."""
        try:
            data = await self.get_dashboard_data(dashboard_id, tenant_id)
            
            if not data.get("success"):
                return data
            
            if format == "json":
                return {
                    "success": True,
                    "format": "json",
                    "data": data
                }
            elif format == "csv":
                # Convert to CSV format (simplified)
                csv_data = "metric,timestamp,value\n"
                for widget_id, widget_data in data["widget_data"].items():
                    if "data" in widget_data:
                        for point in widget_data["data"]:
                            csv_data += f"{widget_id},{point.get('timestamp','')},{point.get('value','')}\n"
                
                return {
                    "success": True,
                    "format": "csv",
                    "data": csv_data
                }
            else:
                return {"success": False, "error": "Unsupported format"}
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return {"success": False, "error": str(e)}
    
    # WebSocket Management
    async def add_websocket_connection(self, websocket, tenant_id: Optional[str] = None):
        """Add WebSocket connection for real-time updates."""
        tenant_key = tenant_id or "global"
        if tenant_key not in self.websocket_connections:
            self.websocket_connections[tenant_key] = []
        
        self.websocket_connections[tenant_key].append(websocket)
    
    async def remove_websocket_connection(self, websocket, tenant_id: Optional[str] = None):
        """Remove WebSocket connection."""
        tenant_key = tenant_id or "global"
        if tenant_key in self.websocket_connections:
            if websocket in self.websocket_connections[tenant_key]:
                self.websocket_connections[tenant_key].remove(websocket)
    
    async def shutdown(self):
        """Shutdown dashboard API."""
        if self.realtime_task:
            self.realtime_task.cancel()
            try:
                await self.realtime_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections
        for connections in self.websocket_connections.values():
            for ws in connections:
                try:
                    await ws.close()
                except Exception:
                    pass
        
        logger.info("Dashboard API shutdown complete")


# Global dashboard API instance
_dashboard_api: Optional[DashboardAPI] = None


def get_dashboard_api() -> Optional[DashboardAPI]:
    """Get global dashboard API."""
    return _dashboard_api


async def initialize_dashboard_api(analytics_store, metrics_collector):
    """Initialize global dashboard API."""
    global _dashboard_api
    _dashboard_api = DashboardAPI(analytics_store, metrics_collector)
    logger.info("Dashboard API initialized")


async def shutdown_dashboard_api():
    """Shutdown global dashboard API."""
    global _dashboard_api
    if _dashboard_api:
        await _dashboard_api.shutdown()
        _dashboard_api = None
