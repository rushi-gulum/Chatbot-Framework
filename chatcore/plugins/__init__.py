"""
Plugin System
============

Dynamic plugin loading and management system for extensible chatbot capabilities.

PHASE3-REFACTOR: Enterprise plugin architecture with hot-reloading and lifecycle management.

Features:
- Dynamic plugin loading
- Hot-reloading capabilities
- Plugin lifecycle management
- Hook system for extensibility
- Plugin isolation
- Dependency management
- Configuration injection
- Event system integration
"""

import asyncio
import importlib
import importlib.util
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Callable, Union
from enum import Enum
import json

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    STARTED = "started"
    STOPPED = "stopped"
    ERROR = "error"


class PluginType(Enum):
    """Types of plugins."""
    CHANNEL = "channel"           # Communication channels
    MIDDLEWARE = "middleware"     # Request/response middleware
    PROCESSOR = "processor"       # Message processors
    INTEGRATION = "integration"   # External integrations
    ANALYTICS = "analytics"       # Analytics extensions
    SECURITY = "security"         # Security enhancements
    TOOL = "tool"                # Chatbot tools
    STORAGE = "storage"          # Storage backends
    CUSTOM = "custom"            # Custom functionality


class HookType(Enum):
    """Available hook types for plugin integration."""
    PRE_MESSAGE = "pre_message"
    POST_MESSAGE = "post_message"
    PRE_RESPONSE = "pre_response"
    POST_RESPONSE = "post_response"
    MESSAGE_RECEIVED = "message_received"
    RESPONSE_GENERATED = "response_generated"
    USER_AUTHENTICATED = "user_authenticated"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    ERROR_OCCURRED = "error_occurred"
    METRICS_COLLECTED = "metrics_collected"
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_UNLOADED = "plugin_unloaded"


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    
    # Requirements
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    framework_version: str = ">=3.0.0"
    
    # Configuration
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None
    
    # Capabilities
    hooks: List[HookType] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)  # Services/features provided
    requires: List[str] = field(default_factory=list)  # Required services/features
    
    # Loading options
    auto_load: bool = True
    load_priority: int = 100  # Lower number = higher priority
    hot_reload: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "python_version": self.python_version,
            "framework_version": self.framework_version,
            "config_schema": self.config_schema,
            "default_config": self.default_config,
            "hooks": [hook.value for hook in self.hooks],
            "provides": self.provides,
            "requires": self.requires,
            "auto_load": self.auto_load,
            "load_priority": self.load_priority,
            "hot_reload": self.hot_reload
        }


class IPlugin(ABC):
    """
    Base interface for all plugins.
    
    PHASE3-REFACTOR: Standard plugin interface with lifecycle management.
    """
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.config: Dict[str, Any] = {}
        self.state: PluginState = PluginState.UNLOADED
        self.plugin_manager: Optional['PluginManager'] = None
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    async def initialize(self, config: Dict[str, Any], plugin_manager: 'PluginManager'):
        """Initialize plugin with configuration."""
        self.config = config
        self.plugin_manager = plugin_manager
        self.metadata = self.get_metadata()
        self.state = PluginState.INITIALIZED
        self.logger.info(f"Plugin {self.metadata.name} initialized")
    
    async def start(self):
        """Start plugin operation."""
        self.state = PluginState.STARTED
        plugin_name = self.metadata.name if self.metadata else self.__class__.__name__
        self.logger.info(f"Plugin {plugin_name} started")
    
    async def stop(self):
        """Stop plugin operation."""
        self.state = PluginState.STOPPED
        plugin_name = self.metadata.name if self.metadata else self.__class__.__name__
        self.logger.info(f"Plugin {plugin_name} stopped")
    
    async def reload(self):
        """Reload plugin (hot-reload)."""
        await self.stop()
        await self.start()
        plugin_name = self.metadata.name if self.metadata else self.__class__.__name__
        self.logger.info(f"Plugin {plugin_name} reloaded")
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "name": self.metadata.name if self.metadata else "Unknown",
            "state": self.state.value,
            "config": self.config
        }


class PluginHook:
    """Hook registration and execution system."""
    
    def __init__(self):
        self.hooks: Dict[HookType, List[Callable]] = {hook: [] for hook in HookType}
    
    def register_hook(self, hook_type: HookType, callback: Callable, plugin_name: str):
        """Register a hook callback."""
        # Add plugin info to callback
        callback._plugin_name = plugin_name
        self.hooks[hook_type].append(callback)
        logger.debug(f"Registered {hook_type.value} hook for plugin {plugin_name}")
    
    def unregister_hooks(self, plugin_name: str):
        """Unregister all hooks for a plugin."""
        removed_count = 0
        for hook_type, callbacks in self.hooks.items():
            self.hooks[hook_type] = [
                cb for cb in callbacks 
                if getattr(cb, '_plugin_name', None) != plugin_name
            ]
            removed_count += len(callbacks) - len(self.hooks[hook_type])
        
        logger.debug(f"Unregistered {removed_count} hooks for plugin {plugin_name}")
    
    async def execute_hooks(self, hook_type: HookType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all registered hooks for a type."""
        callbacks = self.hooks.get(hook_type, [])
        
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    context = await callback(context) or context
                else:
                    context = callback(context) or context
            except Exception as e:
                plugin_name = getattr(callback, '_plugin_name', 'Unknown')
                logger.error(f"Hook execution error in plugin {plugin_name}: {e}")
        
        return context
    
    def get_hook_count(self, hook_type: HookType) -> int:
        """Get number of registered hooks for a type."""
        return len(self.hooks.get(hook_type, []))


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    instance: IPlugin
    metadata: PluginMetadata
    module: Any
    file_path: Path
    load_time: datetime
    state: PluginState
    config: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PluginManager:
    """
    Central plugin management system.
    
    PHASE3-REFACTOR: Enterprise plugin manager with lifecycle and dependency management.
    """
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        self.plugin_directories = plugin_directories or []
        self.plugins: Dict[str, PluginInfo] = {}
        self.hook_system = PluginHook()
        
        # Plugin loading
        self.auto_load_enabled = True
        self.hot_reload_enabled = True
        
        # File watching for hot reload
        self.file_watchers = {}
        self.watch_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.global_config: Dict[str, Any] = {}
        
        # Statistics
        self.plugins_loaded = 0
        self.plugins_failed = 0
        self.reload_count = 0
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        if self.hot_reload_enabled:
            self._start_file_watching()
    
    def _start_file_watching(self):
        """Start file watching for hot reload."""
        self.watch_task = asyncio.create_task(self._file_watch_loop())
    
    async def _file_watch_loop(self):
        """Background task for file watching."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                await self._check_file_changes()
            except Exception as e:
                logger.error(f"File watch error: {e}")
    
    async def _check_file_changes(self):
        """Check for file changes and trigger reloads."""
        # Simple timestamp-based change detection
        for plugin_name, plugin_info in self.plugins.items():
            if not plugin_info.metadata.hot_reload:
                continue
            
            try:
                current_mtime = plugin_info.file_path.stat().st_mtime
                last_mtime = self.file_watchers.get(plugin_name, 0)
                
                if current_mtime > last_mtime:
                    self.file_watchers[plugin_name] = current_mtime
                    if last_mtime > 0:  # Skip initial load
                        await self.reload_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Error checking file changes for {plugin_name}: {e}")
    
    # Plugin Discovery
    def discover_plugins(self) -> List[Path]:
        """Discover plugin files in configured directories."""
        plugin_files = []
        
        for directory in self.plugin_directories:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                # Look for Python files
                for file_path in dir_path.rglob("*.py"):
                    if file_path.name.startswith("plugin_") or file_path.name.endswith("_plugin.py"):
                        plugin_files.append(file_path)
                
                # Look for plugin.json descriptors
                for json_file in dir_path.rglob("plugin.json"):
                    plugin_files.append(json_file.parent / "__init__.py")
        
        return plugin_files
    
    def _load_plugin_metadata_from_file(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from JSON file."""
        metadata_file = plugin_file.parent / "plugin.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                return PluginMetadata(
                    name=data["name"],
                    version=data["version"],
                    description=data["description"],
                    author=data["author"],
                    plugin_type=PluginType(data.get("plugin_type", "custom")),
                    dependencies=data.get("dependencies", []),
                    python_version=data.get("python_version", ">=3.8"),
                    framework_version=data.get("framework_version", ">=3.0.0"),
                    config_schema=data.get("config_schema"),
                    default_config=data.get("default_config"),
                    hooks=[HookType(h) for h in data.get("hooks", [])],
                    provides=data.get("provides", []),
                    requires=data.get("requires", []),
                    auto_load=data.get("auto_load", True),
                    load_priority=data.get("load_priority", 100),
                    hot_reload=data.get("hot_reload", True)
                )
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_file}: {e}")
        
        return None
    
    # Plugin Loading
    async def load_plugin(self, plugin_path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a single plugin."""
        try:
            plugin_path = Path(plugin_path)
            
            # Load module
            spec = importlib.util.spec_from_file_location(f"plugin_{plugin_path.stem}", plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"Cannot load plugin spec from {plugin_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, IPlugin) and obj != IPlugin:
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                logger.error(f"No plugin class found in {plugin_path}")
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()
            
            # Check if already loaded
            if metadata.name in self.plugins:
                logger.warning(f"Plugin {metadata.name} already loaded")
                return False
            
            # Load configuration
            plugin_config = config or {}
            
            # Try to load from JSON metadata
            json_metadata = self._load_plugin_metadata_from_file(plugin_path)
            if json_metadata and json_metadata.default_config:
                plugin_config = {**json_metadata.default_config, **plugin_config}
            
            # Merge with global config
            if metadata.name in self.global_config:
                plugin_config = {**plugin_config, **self.global_config[metadata.name]}
            
            # Initialize plugin
            await plugin_instance.initialize(plugin_config, self)
            
            # Create plugin info
            plugin_info = PluginInfo(
                instance=plugin_instance,
                metadata=metadata,
                module=module,
                file_path=plugin_path,
                load_time=datetime.utcnow(),
                state=PluginState.LOADED,
                config=plugin_config
            )
            
            self.plugins[metadata.name] = plugin_info
            self.file_watchers[metadata.name] = plugin_path.stat().st_mtime
            
            # Register hooks
            await self._register_plugin_hooks(plugin_instance)
            
            # Emit event
            await self._emit_event("plugin_loaded", {"plugin_name": metadata.name})
            
            self.plugins_loaded += 1
            logger.info(f"Plugin {metadata.name} loaded successfully")
            
            return True
            
        except Exception as e:
            self.plugins_failed += 1
            logger.error(f"Error loading plugin from {plugin_path}: {e}")
            return False
    
    async def _register_plugin_hooks(self, plugin: IPlugin):
        """Register plugin hooks with the hook system."""
        if not plugin.metadata:
            return
        
        for method_name in dir(plugin):
            method = getattr(plugin, method_name)
            
            if callable(method) and hasattr(method, '_hook_type'):
                hook_type = method._hook_type
                self.hook_system.register_hook(hook_type, method, plugin.metadata.name)
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return False
        
        try:
            plugin_info = self.plugins[plugin_name]
            
            # Stop plugin
            await plugin_info.instance.stop()
            
            # Unregister hooks
            self.hook_system.unregister_hooks(plugin_name)
            
            # Remove from modules
            if plugin_info.module and hasattr(plugin_info.module, '__name__'):
                if plugin_info.module.__name__ in sys.modules:
                    del sys.modules[plugin_info.module.__name__]
            
            # Remove plugin info
            del self.plugins[plugin_name]
            if plugin_name in self.file_watchers:
                del self.file_watchers[plugin_name]
            
            # Emit event
            await self._emit_event("plugin_unloaded", {"plugin_name": plugin_name})
            
            logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (hot-reload)."""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin_info = self.plugins[plugin_name]
            plugin_config = plugin_info.config.copy()
            plugin_path = plugin_info.file_path
            
            # Unload current plugin
            await self.unload_plugin(plugin_name)
            
            # Load again
            success = await self.load_plugin(plugin_path, plugin_config)
            
            if success:
                # Start the reloaded plugin
                await self.start_plugin(plugin_name)
                self.reload_count += 1
                logger.info(f"Plugin {plugin_name} reloaded successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False
    
    # Plugin Lifecycle
    async def start_plugin(self, plugin_name: str) -> bool:
        """Start a loaded plugin."""
        if plugin_name not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_name]
        
        try:
            await plugin_info.instance.start()
            plugin_info.state = PluginState.STARTED
            return True
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            logger.error(f"Error starting plugin {plugin_name}: {e}")
            return False
    
    async def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a running plugin."""
        if plugin_name not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_name]
        
        try:
            await plugin_info.instance.stop()
            plugin_info.state = PluginState.STOPPED
            return True
        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            logger.error(f"Error stopping plugin {plugin_name}: {e}")
            return False
    
    # Bulk Operations
    async def load_all_plugins(self, auto_start: bool = True) -> Dict[str, bool]:
        """Load all discovered plugins."""
        results = {}
        plugin_files = self.discover_plugins()
        
        # Sort by priority (if metadata available)
        sorted_files = []
        for plugin_file in plugin_files:
            metadata = self._load_plugin_metadata_from_file(plugin_file)
            priority = metadata.load_priority if metadata else 100
            sorted_files.append((priority, plugin_file))
        
        sorted_files.sort(key=lambda x: x[0])
        
        # Load plugins in priority order
        for priority, plugin_file in sorted_files:
            plugin_name = plugin_file.stem
            success = await self.load_plugin(plugin_file)
            results[plugin_name] = success
            
            if success and auto_start:
                await self.start_plugin(plugin_name)
        
        return results
    
    async def start_all_plugins(self) -> Dict[str, bool]:
        """Start all loaded plugins."""
        results = {}
        for plugin_name in self.plugins:
            results[plugin_name] = await self.start_plugin(plugin_name)
        return results
    
    async def stop_all_plugins(self) -> Dict[str, bool]:
        """Stop all running plugins."""
        results = {}
        for plugin_name in self.plugins:
            results[plugin_name] = await self.stop_plugin(plugin_name)
        return results
    
    # Configuration
    def set_global_config(self, config: Dict[str, Any]):
        """Set global plugin configuration."""
        self.global_config = config
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].config
        return {}
    
    async def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        if plugin_name not in self.plugins:
            return False
        
        self.plugins[plugin_name].config.update(config)
        
        # Reload plugin with new config
        if self.plugins[plugin_name].metadata.hot_reload:
            return await self.reload_plugin(plugin_name)
        
        return True
    
    # Hook System Access
    async def execute_hooks(self, hook_type: HookType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hooks of a specific type."""
        return await self.hook_system.execute_hooks(hook_type, context)
    
    # Information and Debugging
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        if plugin_name not in self.plugins:
            return None
        
        plugin_info = self.plugins[plugin_name]
        
        return {
            "name": plugin_info.metadata.name,
            "version": plugin_info.metadata.version,
            "description": plugin_info.metadata.description,
            "author": plugin_info.metadata.author,
            "type": plugin_info.metadata.plugin_type.value,
            "state": plugin_info.state.value,
            "load_time": plugin_info.load_time.isoformat(),
            "file_path": str(plugin_info.file_path),
            "config": plugin_info.config,
            "error": plugin_info.error,
            "hooks": [hook.value for hook in plugin_info.metadata.hooks],
            "provides": plugin_info.metadata.provides,
            "requires": plugin_info.metadata.requires
        }
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins."""
        plugins = []
        for name in self.plugins.keys():
            plugin_info = self.get_plugin_info(name)
            if plugin_info:
                plugins.append(plugin_info)
        return plugins
    
    def get_plugin_states(self) -> Dict[str, str]:
        """Get states of all plugins."""
        return {name: info.state.value for name, info in self.plugins.items()}
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            "total_plugins": len(self.plugins),
            "plugins_loaded": self.plugins_loaded,
            "plugins_failed": self.plugins_failed,
            "reload_count": self.reload_count,
            "auto_load_enabled": self.auto_load_enabled,
            "hot_reload_enabled": self.hot_reload_enabled,
            "plugin_directories": self.plugin_directories,
            "total_hooks": sum(self.hook_system.get_hook_count(hook) for hook in HookType)
        }
    
    # Event System
    async def _emit_event(self, event_name: str, data: Dict[str, Any]):
        """Emit an event to registered callbacks."""
        callbacks = self.event_callbacks.get(event_name, [])
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def add_event_callback(self, event_name: str, callback: Callable):
        """Add event callback."""
        if event_name not in self.event_callbacks:
            self.event_callbacks[event_name] = []
        self.event_callbacks[event_name].append(callback)
    
    def remove_event_callback(self, event_name: str, callback: Callable):
        """Remove event callback."""
        if event_name in self.event_callbacks:
            self.event_callbacks[event_name] = [
                cb for cb in self.event_callbacks[event_name] if cb != callback
            ]
    
    # Cleanup
    async def shutdown(self):
        """Shutdown plugin manager."""
        # Stop file watching
        if self.watch_task:
            self.watch_task.cancel()
            try:
                await self.watch_task
            except asyncio.CancelledError:
                pass
        
        # Stop all plugins
        await self.stop_all_plugins()
        
        # Unload all plugins
        for plugin_name in list(self.plugins.keys()):
            await self.unload_plugin(plugin_name)
        
        logger.info("Plugin manager shutdown complete")


# Hook decorator for easy hook registration
def hook(hook_type: HookType):
    """Decorator to register plugin methods as hooks."""
    def decorator(func):
        func._hook_type = hook_type
        return func
    return decorator


# Global plugin manager
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


async def initialize_plugin_manager(plugin_directories: Optional[List[str]] = None):
    """Initialize global plugin manager."""
    global _plugin_manager
    _plugin_manager = PluginManager(plugin_directories)
    logger.info("Plugin manager initialized")


async def shutdown_plugin_manager():
    """Shutdown global plugin manager."""
    global _plugin_manager
    if _plugin_manager:
        await _plugin_manager.shutdown()
        _plugin_manager = None
