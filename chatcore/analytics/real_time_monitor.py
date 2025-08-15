"""
Real-time Analytics Monitor
==========================

Real-time monitoring system for live analytics updates.

PHASE3-REFACTOR: WebSocket-based real-time monitoring with event streams.

Features:
- WebSocket connections
- Real-time metric streams
- Alert system
- Event-driven updates
- Performance monitoring
- Anomaly detection
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class AlertType(Enum):
    """Types of alerts."""
    THRESHOLD = "threshold"           # Value crosses threshold
    ANOMALY = "anomaly"              # Statistical anomaly
    RATE_CHANGE = "rate_change"      # Rapid rate change
    MISSING_DATA = "missing_data"    # Data not received
    SYSTEM_HEALTH = "system_health"  # System health issues


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    metric_name: str
    alert_type: AlertType
    severity: AlertSeverity
    
    # Threshold-based
    threshold_value: Optional[float] = None
    threshold_operator: str = ">"  # >, <, >=, <=, ==, !=
    
    # Rate change
    rate_change_threshold: Optional[float] = None  # % change
    rate_change_window: int = 300  # seconds
    
    # Missing data
    missing_data_threshold: int = 300  # seconds
    
    # Conditions
    tenant_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Alert settings
    cooldown_seconds: int = 300  # Prevent spam
    enabled: bool = True
    
    def evaluate_threshold(self, value: float) -> bool:
        """Evaluate threshold condition."""
        if self.threshold_value is None:
            return False
        
        if self.threshold_operator == ">":
            return value > self.threshold_value
        elif self.threshold_operator == "<":
            return value < self.threshold_value
        elif self.threshold_operator == ">=":
            return value >= self.threshold_value
        elif self.threshold_operator == "<=":
            return value <= self.threshold_value
        elif self.threshold_operator == "==":
            return value == self.threshold_value
        elif self.threshold_operator == "!=":
            return value != self.threshold_value
        
        return False


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_id: str
    rule_name: str
    metric_name: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    value: float
    threshold: Optional[float]
    tenant_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


@dataclass
class MetricEvent:
    """Real-time metric event."""
    metric_name: str
    value: float
    timestamp: datetime
    tenant_id: Optional[str] = None
    dimensions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "dimensions": self.dimensions
        }


class MetricBuffer:
    """Buffer for metric values with time window."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.values: deque = deque(maxlen=1000)
        self.last_cleanup = time.time()
    
    def add_value(self, value: float, timestamp: datetime):
        """Add metric value."""
        self.values.append((timestamp, value))
        
        # Periodic cleanup
        current_time = time.time()
        if current_time - self.last_cleanup > 60:  # Every minute
            self._cleanup_old_values()
            self.last_cleanup = current_time
    
    def _cleanup_old_values(self):
        """Remove values outside window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_size)
        
        while self.values and self.values[0][0] < cutoff_time:
            self.values.popleft()
    
    def get_recent_values(self, seconds: Optional[int] = None) -> List[float]:
        """Get recent values within time window."""
        if seconds is None:
            seconds = self.window_size
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)
        return [value for timestamp, value in self.values if timestamp >= cutoff_time]
    
    def get_latest_value(self) -> Optional[float]:
        """Get latest value."""
        return self.values[-1][1] if self.values else None
    
    def calculate_rate_change(self, window_seconds: int = 300) -> Optional[float]:
        """Calculate rate of change over window."""
        values = self.get_recent_values(window_seconds)
        
        if len(values) < 2:
            return None
        
        # Simple linear trend
        old_avg = statistics.mean(values[:len(values)//2])
        new_avg = statistics.mean(values[len(values)//2:])
        
        if old_avg == 0:
            return None
        
        return ((new_avg - old_avg) / old_avg) * 100
    
    def detect_anomaly(self, threshold_std: float = 2.0) -> bool:
        """Detect statistical anomaly."""
        values = self.get_recent_values()
        
        if len(values) < 10:  # Need sufficient data
            return False
        
        latest_value = values[-1]
        mean_value = statistics.mean(values[:-1])  # Exclude latest
        
        try:
            std_dev = statistics.stdev(values[:-1])
            if std_dev == 0:
                return False
            
            z_score = abs((latest_value - mean_value) / std_dev)
            return z_score > threshold_std
        except statistics.StatisticsError:
            return False


class RealTimeMonitor:
    """
    Real-time analytics monitoring system.
    
    PHASE3-REFACTOR: WebSocket-based real-time monitoring with alerts.
    """
    
    def __init__(self):
        # WebSocket connections by tenant
        self.websocket_connections: Dict[str, Set[Any]] = defaultdict(set)
        
        # Metric buffers
        self.metric_buffers: Dict[str, MetricBuffer] = {}
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Event handlers
        self.event_handlers: List[Callable] = []
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.events_processed = 0
        self.alerts_triggered = 0
        self.connections_count = 0
        
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.alert_task = asyncio.create_task(self._alert_loop())
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                await self._process_monitoring()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    async def _alert_loop(self):
        """Background alert processing loop."""
        while True:
            try:
                await asyncio.sleep(5)  # Check alerts every 5 seconds
                await self._process_alerts()
            except Exception as e:
                logger.error(f"Alert loop error: {e}")
    
    async def _process_monitoring(self):
        """Process monitoring checks."""
        # Check for missing data alerts
        current_time = datetime.utcnow()
        
        for rule in self.alert_rules.values():
            if (rule.alert_type == AlertType.MISSING_DATA and 
                rule.enabled and 
                self._is_rule_applicable(rule)):
                
                buffer_key = f"{rule.metric_name}:{rule.tenant_id or 'global'}"
                buffer = self.metric_buffers.get(buffer_key)
                
                if buffer:
                    latest_value = buffer.get_latest_value()
                    if latest_value is not None:
                        # Check if data is missing
                        time_since_data = (current_time - buffer.values[-1][0]).total_seconds()
                        
                        if time_since_data > rule.missing_data_threshold:
                            await self._trigger_alert(rule, 0, f"No data received for {time_since_data:.0f} seconds")
    
    def _is_rule_applicable(self, rule: AlertRule) -> bool:
        """Check if rule is applicable based on conditions."""
        # Check cooldown
        cooldown_key = f"{rule.id}"
        if cooldown_key in self.alert_cooldowns:
            if datetime.utcnow() < self.alert_cooldowns[cooldown_key]:
                return False
        
        return True
    
    async def _process_alerts(self):
        """Process alert rules."""
        for rule in self.alert_rules.values():
            if not rule.enabled or not self._is_rule_applicable(rule):
                continue
            
            buffer_key = f"{rule.metric_name}:{rule.tenant_id or 'global'}"
            buffer = self.metric_buffers.get(buffer_key)
            
            if not buffer:
                continue
            
            latest_value = buffer.get_latest_value()
            if latest_value is None:
                continue
            
            # Check different alert types
            if rule.alert_type == AlertType.THRESHOLD:
                if rule.evaluate_threshold(latest_value):
                    message = f"Metric {rule.metric_name} = {latest_value} crossed threshold {rule.threshold_value}"
                    await self._trigger_alert(rule, latest_value, message)
            
            elif rule.alert_type == AlertType.ANOMALY:
                if buffer.detect_anomaly():
                    message = f"Anomaly detected in {rule.metric_name}: {latest_value}"
                    await self._trigger_alert(rule, latest_value, message)
            
            elif rule.alert_type == AlertType.RATE_CHANGE:
                rate_change = buffer.calculate_rate_change(rule.rate_change_window)
                if (rate_change is not None and 
                    rule.rate_change_threshold is not None and
                    abs(rate_change) > rule.rate_change_threshold):
                    message = f"Rapid change in {rule.metric_name}: {rate_change:.1f}% in {rule.rate_change_window}s"
                    await self._trigger_alert(rule, latest_value, message)
    
    async def _trigger_alert(self, rule: AlertRule, value: float, message: str):
        """Trigger alert for rule."""
        alert_id = f"{rule.id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            metric_name=rule.metric_name,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=message,
            value=value,
            threshold=rule.threshold_value,
            tenant_id=rule.tenant_id
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        
        # Set cooldown
        cooldown_until = datetime.utcnow() + timedelta(seconds=rule.cooldown_seconds)
        self.alert_cooldowns[rule.id] = cooldown_until
        
        # Send alert
        await self._send_alert_notification(alert)
        
        self.alerts_triggered += 1
        logger.warning(f"Alert triggered: {alert.message}")
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification to connected clients."""
        tenant_key = alert.tenant_id or "global"
        connections = self.websocket_connections.get(tenant_key, set())
        
        alert_message = {
            "type": "alert",
            "alert": alert.to_dict()
        }
        
        disconnected = set()
        for ws in connections:
            try:
                await ws.send_text(json.dumps(alert_message))
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            connections.discard(ws)
    
    # Public API
    async def add_websocket_connection(self, websocket, tenant_id: Optional[str] = None):
        """Add WebSocket connection for real-time updates."""
        tenant_key = tenant_id or "global"
        self.websocket_connections[tenant_key].add(websocket)
        self.connections_count = sum(len(conns) for conns in self.websocket_connections.values())
        
        # Send current status
        await self._send_current_status(websocket, tenant_id)
    
    async def remove_websocket_connection(self, websocket, tenant_id: Optional[str] = None):
        """Remove WebSocket connection."""
        tenant_key = tenant_id or "global"
        self.websocket_connections[tenant_key].discard(websocket)
        self.connections_count = sum(len(conns) for conns in self.websocket_connections.values())
    
    async def _send_current_status(self, websocket, tenant_id: Optional[str] = None):
        """Send current status to new connection."""
        try:
            # Send active alerts
            active_alerts = [alert.to_dict() for alert in self.active_alerts.values()
                           if not tenant_id or alert.tenant_id == tenant_id]
            
            status_message = {
                "type": "status",
                "active_alerts": active_alerts,
                "monitor_stats": self.get_monitor_stats()
            }
            
            await websocket.send_text(json.dumps(status_message))
        except Exception as e:
            logger.error(f"Error sending status: {e}")
    
    async def process_metric_event(self, event: MetricEvent):
        """Process incoming metric event."""
        try:
            buffer_key = f"{event.metric_name}:{event.tenant_id or 'global'}"
            
            # Get or create buffer
            if buffer_key not in self.metric_buffers:
                self.metric_buffers[buffer_key] = MetricBuffer()
            
            buffer = self.metric_buffers[buffer_key]
            buffer.add_value(event.value, event.timestamp)
            
            # Send real-time update
            await self._send_metric_update(event)
            
            # Call event handlers
            for handler in self.event_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing metric event: {e}")
    
    async def _send_metric_update(self, event: MetricEvent):
        """Send metric update to connected clients."""
        tenant_key = event.tenant_id or "global"
        connections = self.websocket_connections.get(tenant_key, set())
        
        update_message = {
            "type": "metric_update",
            "metric": event.to_dict()
        }
        
        disconnected = set()
        for ws in connections:
            try:
                await ws.send_text(json.dumps(update_message))
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            connections.discard(ws)
    
    # Alert Rule Management
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def get_alert_rules(self, tenant_id: Optional[str] = None) -> List[AlertRule]:
        """Get alert rules for tenant."""
        rules = []
        for rule in self.alert_rules.values():
            if not tenant_id or rule.tenant_id == tenant_id:
                rules.append(rule)
        return rules
    
    # Alert Management
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge alert."""
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
            await self._send_alert_update(alert)
            return True
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert."""
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.resolved = True
            await self._send_alert_update(alert)
            return True
        return False
    
    async def _send_alert_update(self, alert: Alert):
        """Send alert update to clients."""
        tenant_key = alert.tenant_id or "global"
        connections = self.websocket_connections.get(tenant_key, set())
        
        update_message = {
            "type": "alert_update",
            "alert": alert.to_dict()
        }
        
        disconnected = set()
        for ws in connections:
            try:
                await ws.send_text(json.dumps(update_message))
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            connections.discard(ws)
    
    def get_active_alerts(self, tenant_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts for tenant."""
        alerts = []
        for alert in self.active_alerts.values():
            if not tenant_id or alert.tenant_id == tenant_id:
                alerts.append(alert)
        return alerts
    
    # Event Handler Management
    def add_event_handler(self, handler: Callable):
        """Add event handler function."""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable):
        """Remove event handler function."""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    # Statistics
    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "events_processed": self.events_processed,
            "alerts_triggered": self.alerts_triggered,
            "active_alerts_count": len(self.active_alerts),
            "alert_rules_count": len(self.alert_rules),
            "connections_count": self.connections_count,
            "metric_buffers_count": len(self.metric_buffers)
        }
    
    async def shutdown(self):
        """Shutdown real-time monitor."""
        # Cancel background tasks
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections
        for connections in self.websocket_connections.values():
            for ws in connections:
                try:
                    await ws.close()
                except Exception:
                    pass
        
        logger.info("Real-time monitor shutdown complete")


# Global monitor instance
_realtime_monitor: Optional[RealTimeMonitor] = None


def get_realtime_monitor() -> RealTimeMonitor:
    """Get global real-time monitor."""
    global _realtime_monitor
    if _realtime_monitor is None:
        _realtime_monitor = RealTimeMonitor()
    return _realtime_monitor


async def initialize_realtime_monitor():
    """Initialize global real-time monitor."""
    global _realtime_monitor
    _realtime_monitor = RealTimeMonitor()
    logger.info("Real-time monitor initialized")


async def shutdown_realtime_monitor():
    """Shutdown global real-time monitor."""
    global _realtime_monitor
    if _realtime_monitor:
        await _realtime_monitor.shutdown()
        _realtime_monitor = None


# Convenience functions
async def process_metric_event(metric_name: str, value: float, 
                             tenant_id: Optional[str] = None,
                             dimensions: Optional[Dict[str, str]] = None):
    """Process metric event using global monitor."""
    monitor = get_realtime_monitor()
    event = MetricEvent(
        metric_name=metric_name,
        value=value,
        timestamp=datetime.utcnow(),
        tenant_id=tenant_id,
        dimensions=dimensions or {}
    )
    await monitor.process_metric_event(event)


def add_alert_rule(rule: AlertRule):
    """Add alert rule using global monitor."""
    monitor = get_realtime_monitor()
    monitor.add_alert_rule(rule)
