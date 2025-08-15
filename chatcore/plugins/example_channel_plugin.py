"""
Example Channel Plugin
====================

Sample plugin demonstrating how to create a custom communication channel.

This plugin shows:
- Plugin interface implementation
- Hook registration
- Configuration handling
- Channel functionality
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from chatcore.plugins import IPlugin, PluginMetadata, PluginType, HookType, hook
from chatcore.channels.base_channel import BaseChannel

logger = logging.getLogger(__name__)


class CustomChannelPlugin(IPlugin):
    """Example custom channel plugin."""
    
    def __init__(self):
        super().__init__()
        self.channel_instance: Optional[CustomChannel] = None
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="custom_channel",
            version="1.0.0",
            description="Example custom communication channel",
            author="ChatBot Framework",
            plugin_type=PluginType.CHANNEL,
            hooks=[HookType.MESSAGE_RECEIVED, HookType.RESPONSE_GENERATED],
            provides=["custom_channel"],
            default_config={
                "enabled": True,
                "port": 8080,
                "webhook_path": "/webhook/custom",
                "auth_token": "your-auth-token"
            }
        )
    
    async def initialize(self, config: Dict[str, Any], plugin_manager):
        """Initialize plugin."""
        await super().initialize(config, plugin_manager)
        
        # Create channel instance
        self.channel_instance = CustomChannel(
            port=config.get("port", 8080),
            webhook_path=config.get("webhook_path", "/webhook/custom"),
            auth_token=config.get("auth_token", "")
        )
    
    async def start(self):
        """Start plugin."""
        await super().start()
        
        if self.channel_instance and self.config.get("enabled", True):
            await self.channel_instance.start()
            self.logger.info("Custom channel started")
    
    async def stop(self):
        """Stop plugin."""
        if self.channel_instance:
            await self.channel_instance.stop()
            self.logger.info("Custom channel stopped")
        
        await super().stop()
    
    @hook(HookType.MESSAGE_RECEIVED)
    async def on_message_received(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for message received events."""
        message = context.get("message", {})
        
        # Add custom processing
        if message.get("channel") == "custom":
            self.logger.info(f"Custom channel processing message: {message.get('text', '')}")
            
            # Add custom metadata
            context["custom_processed"] = True
            context["processing_time"] = asyncio.get_event_loop().time()
        
        return context
    
    @hook(HookType.RESPONSE_GENERATED)
    async def on_response_generated(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for response generated events."""
        response = context.get("response", {})
        
        # Add custom response formatting for this channel
        if context.get("channel") == "custom":
            if "text" in response:
                response["text"] = f"🤖 {response['text']}"
                context["custom_formatted"] = True
        
        return context


class CustomChannel(BaseChannel):
    """Example custom channel implementation."""
    
    def __init__(self, port: int = 8080, webhook_path: str = "/webhook/custom", auth_token: str = ""):
        channel_config = {
            "name": "custom",
            "port": port,
            "webhook_path": webhook_path,
            "auth_token": auth_token
        }
        super().__init__(channel_config)
        self.port = port
        self.webhook_path = webhook_path
        self.auth_token = auth_token
        self.server = None
        self.running = False
        self.logger = logging.getLogger(f"channel.custom")
    
    def _get_channel_type(self) -> str:
        """Get channel type identifier."""
        return "custom"
    
    async def initialize(self):
        """Initialize the channel."""
        self.logger.info("Initializing custom channel")
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate incoming message format."""
        required_fields = ["user_id", "text"]
        return all(field in message for field in required_fields)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get channel capabilities."""
        return {
            "supports_text": True,
            "supports_images": False,
            "supports_audio": False,
            "supports_video": False,
            "supports_files": False,
            "max_message_length": 4000,
            "supports_buttons": True,
            "supports_quick_replies": True
        }
    
    async def start(self):
        """Start the custom channel server."""
        # In a real implementation, you would start a web server here
        self.running = True
        self.logger.info(f"Custom channel listening on port {self.port}")
    
    async def stop(self):
        """Stop the custom channel server."""
        self.running = False
        if self.server:
            # Stop server
            pass
        self.logger.info("Custom channel stopped")
    
    async def send_message(self, message: Dict[str, Any], user_id: str, channel_data: Optional[Dict[str, Any]] = None) -> bool:
        """Send message through custom channel."""
        try:
            # In a real implementation, you would send the message via your custom protocol
            self.logger.info(f"Sending message to user {user_id}: {message.get('text', '')}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
    
    async def receive_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Receive and process incoming message."""
        try:
            # Transform raw message to standard format
            return {
                "user_id": raw_message.get("user_id"),
                "text": raw_message.get("text"),
                "channel": "custom",
                "timestamp": raw_message.get("timestamp"),
                "metadata": raw_message.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error processing incoming message: {e}")
            return None
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get channel information."""
        return {
            "name": "custom",
            "type": "webhook",
            "status": "running" if self.running else "stopped",
            "port": self.port,
            "webhook_path": self.webhook_path
        }
