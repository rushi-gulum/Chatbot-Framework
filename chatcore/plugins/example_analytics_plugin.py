"""
Example Analytics Plugin
=======================

Sample plugin demonstrating analytics and metrics integration.

This plugin shows:
- Metrics collection hooks
- Custom analytics processing
- Dashboard widget creation
- Real-time monitoring
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from datetime import datetime

from chatcore.plugins import IPlugin, PluginMetadata, PluginType, HookType, hook

logger = logging.getLogger(__name__)


class AnalyticsPlugin(IPlugin):
    """Example analytics plugin with custom metrics."""
    
    def __init__(self):
        super().__init__()
        self.metrics_collector = None
        self.dashboard_api = None
        self.session_data = {}  # Track session information
        self.response_times = []  # Track response times
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="analytics_plugin",
            version="1.0.0",
            description="Advanced analytics and metrics collection",
            author="ChatBot Framework",
            plugin_type=PluginType.ANALYTICS,
            hooks=[
                HookType.MESSAGE_RECEIVED,
                HookType.RESPONSE_GENERATED,
                HookType.SESSION_STARTED,
                HookType.SESSION_ENDED,
                HookType.ERROR_OCCURRED
            ],
            provides=["custom_analytics", "session_tracking", "performance_metrics"],
            requires=["metrics_collector"],
            default_config={
                "enabled": True,
                "track_sessions": True,
                "track_performance": True,
                "track_errors": True,
                "dashboard_widgets": True
            }
        )
    
    async def initialize(self, config: Dict[str, Any], plugin_manager):
        """Initialize plugin."""
        await super().initialize(config, plugin_manager)
        
        # Get required services
        try:
            from chatcore.analytics import get_metrics_collector, get_dashboard_api
            self.metrics_collector = get_metrics_collector()
            self.dashboard_api = get_dashboard_api()
        except ImportError:
            self.logger.warning("Analytics services not available")
    
    async def start(self):
        """Start plugin."""
        await super().start()
        
        if self.config.get("dashboard_widgets", True) and self.dashboard_api:
            await self._create_dashboard_widgets()
    
    async def _create_dashboard_widgets(self):
        """Create custom dashboard widgets."""
        try:
            from chatcore.analytics import DashboardWidget, DashboardConfig, ChartType, TimeRange
            
            # Create custom dashboard
            dashboard = DashboardConfig(
                id="analytics_plugin_dashboard",
                name="Advanced Analytics",
                description="Custom analytics dashboard from plugin",
                widgets=[
                    DashboardWidget(
                        id="session_duration",
                        title="Average Session Duration",
                        chart_type=ChartType.GAUGE,
                        metric_name="plugin.session.duration.avg",
                        time_range=TimeRange.LAST_DAY
                    ),
                    DashboardWidget(
                        id="response_time_trend",
                        title="Response Time Trend",
                        chart_type=ChartType.LINE,
                        metric_name="plugin.response.time",
                        time_range=TimeRange.LAST_DAY,
                        time_bucket="1h"
                    ),
                    DashboardWidget(
                        id="error_rate",
                        title="Error Rate",
                        chart_type=ChartType.BAR,
                        metric_name="plugin.errors.rate",
                        time_range=TimeRange.LAST_DAY,
                        time_bucket="1h"
                    )
                ]
            )
            
            if self.dashboard_api:
                await self.dashboard_api.create_dashboard(dashboard)
                self.logger.info("Created custom analytics dashboard")
            else:
                self.logger.warning("Dashboard API not available")
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard widgets: {e}")
    
    @hook(HookType.SESSION_STARTED)
    async def on_session_started(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Track session start."""
        if not self.config.get("track_sessions", True):
            return context
        
        session_id = context.get("session_id")
        user_id = context.get("user_id")
        
        if session_id:
            self.session_data[session_id] = {
                "user_id": user_id,
                "start_time": datetime.utcnow(),
                "message_count": 0,
                "total_response_time": 0
            }
            
            # Collect metric
            if self.metrics_collector:
                self.metrics_collector.increment_counter(
                    "plugin.sessions.started",
                    tenant_id=context.get("tenant_id"),
                    dimensions={"user_id": user_id}
                )
            
            self.logger.debug(f"Session started: {session_id}")
        
        return context
    
    @hook(HookType.SESSION_ENDED)
    async def on_session_ended(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Track session end and calculate duration."""
        if not self.config.get("track_sessions", True):
            return context
        
        session_id = context.get("session_id")
        
        if session_id and session_id in self.session_data:
            session_info = self.session_data[session_id]
            session_duration = (datetime.utcnow() - session_info["start_time"]).total_seconds()
            
            # Collect metrics
            if self.metrics_collector:
                self.metrics_collector.set_gauge(
                    "plugin.session.duration.avg",
                    session_duration,
                    tenant_id=context.get("tenant_id"),
                    dimensions={
                        "user_id": session_info["user_id"],
                        "message_count": str(session_info["message_count"])
                    }
                )
                
                self.metrics_collector.increment_counter(
                    "plugin.sessions.ended",
                    tenant_id=context.get("tenant_id")
                )
            
            self.logger.debug(f"Session ended: {session_id}, duration: {session_duration}s")
            del self.session_data[session_id]
        
        return context
    
    @hook(HookType.MESSAGE_RECEIVED)
    async def on_message_received(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Track message received and start response timing."""
        message = context.get("message", {})
        session_id = context.get("session_id")
        
        # Track session message count
        if session_id and session_id in self.session_data:
            self.session_data[session_id]["message_count"] += 1
        
        # Start response time tracking
        if self.config.get("track_performance", True):
            context["_analytics_start_time"] = time.time()
        
        # Collect metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "plugin.messages.received",
                tenant_id=context.get("tenant_id"),
                dimensions={
                    "channel": message.get("channel", "unknown"),
                    "message_type": message.get("type", "text")
                }
            )
        
        return context
    
    @hook(HookType.RESPONSE_GENERATED)
    async def on_response_generated(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Track response generation and calculate response time."""
        response = context.get("response", {})
        session_id = context.get("session_id")
        
        # Calculate response time
        if self.config.get("track_performance", True) and "_analytics_start_time" in context:
            response_time = (time.time() - context["_analytics_start_time"]) * 1000  # ms
            self.response_times.append(response_time)
            
            # Keep only last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            # Update session data
            if session_id and session_id in self.session_data:
                self.session_data[session_id]["total_response_time"] += response_time
            
            # Collect metrics
            if self.metrics_collector:
                self.metrics_collector.record_timer(
                    "plugin.response.time",
                    response_time,
                    tenant_id=context.get("tenant_id"),
                    dimensions={
                        "response_type": response.get("type", "text"),
                        "session_id": session_id
                    }
                )
                
                # Update average response time
                avg_response_time = sum(self.response_times) / len(self.response_times)
                self.metrics_collector.set_gauge(
                    "plugin.response.time.avg",
                    avg_response_time,
                    tenant_id=context.get("tenant_id")
                )
        
        # Collect response metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "plugin.responses.generated",
                tenant_id=context.get("tenant_id"),
                dimensions={
                    "response_type": response.get("type", "text"),
                    "success": str(response.get("success", True))
                }
            )
        
        return context
    
    @hook(HookType.ERROR_OCCURRED)
    async def on_error_occurred(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Track errors and error rates."""
        if not self.config.get("track_errors", True):
            return context
        
        error = context.get("error", {})
        error_type = error.get("type", "unknown")
        error_message = error.get("message", "")
        
        # Collect error metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "plugin.errors.total",
                tenant_id=context.get("tenant_id"),
                dimensions={
                    "error_type": error_type,
                    "error_category": self._categorize_error(error_type)
                }
            )
            
            # Calculate error rate (errors per minute)
            current_time = time.time()
            recent_errors = [t for t in getattr(self, '_error_timestamps', []) 
                           if current_time - t < 60]  # Last minute
            error_rate = len(recent_errors) + 1  # Include current error
            
            self.metrics_collector.set_gauge(
                "plugin.errors.rate",
                error_rate,
                tenant_id=context.get("tenant_id")
            )
            
            # Store error timestamp
            if not hasattr(self, '_error_timestamps'):
                self._error_timestamps = []
            self._error_timestamps.append(current_time)
            
            # Keep only last 100 error timestamps
            if len(self._error_timestamps) > 100:
                self._error_timestamps = self._error_timestamps[-100:]
        
        self.logger.warning(f"Error tracked: {error_type} - {error_message}")
        
        return context
    
    def _categorize_error(self, error_type: str) -> str:
        """Categorize error type for better analytics."""
        error_type_lower = error_type.lower()
        
        if "timeout" in error_type_lower or "network" in error_type_lower:
            return "network"
        elif "auth" in error_type_lower or "permission" in error_type_lower:
            return "authorization"
        elif "validation" in error_type_lower or "format" in error_type_lower:
            return "validation"
        elif "llm" in error_type_lower or "model" in error_type_lower:
            return "llm"
        elif "database" in error_type_lower or "storage" in error_type_lower:
            return "storage"
        else:
            return "other"
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary for this plugin."""
        active_sessions = len(self.session_data)
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "active_sessions": active_sessions,
            "total_response_times_tracked": len(self.response_times),
            "average_response_time_ms": avg_response_time,
            "session_details": {
                session_id: {
                    "user_id": data["user_id"],
                    "duration_seconds": (datetime.utcnow() - data["start_time"]).total_seconds(),
                    "message_count": data["message_count"],
                    "avg_response_time_ms": data["total_response_time"] / max(data["message_count"], 1)
                }
                for session_id, data in self.session_data.items()
            }
        }
    
    async def stop(self):
        """Stop plugin and cleanup."""
        # Log final analytics
        summary = self.get_analytics_summary()
        self.logger.info(f"Analytics plugin stopping. Final summary: {summary}")
        
        await super().stop()
