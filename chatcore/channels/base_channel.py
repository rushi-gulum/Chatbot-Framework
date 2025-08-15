"""
Base Channel Module
==================

Base classes and interfaces for communication channels.
Provides abstract base classes for implementing different communication platforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class MessageType(Enum):
    """Types of messages supported by channels."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    QUICK_REPLY = "quick_reply"
    BUTTON = "button"
    CAROUSEL = "carousel"


class ChannelType(Enum):
    """Types of communication channels."""
    WEB = "web"
    MOBILE = "mobile"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    VOICE = "voice"
    SMS = "sms"
    EMAIL = "email"


@dataclass
class ChannelUser:
    """User information for channels."""
    user_id: str
    channel_user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MessageAttachment:
    """File attachment for messages."""
    filename: str
    content_type: str
    size: int
    url: Optional[str] = None
    data: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChannelMessage:
    """Message received from a channel."""
    message_id: str
    user: ChannelUser
    text: str
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attachments: List[MessageAttachment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None


@dataclass
class ChannelResponse:
    """Response to send through a channel."""
    text: str
    message_type: MessageType = MessageType.TEXT
    attachments: List[MessageAttachment] = field(default_factory=list)
    quick_replies: List[str] = field(default_factory=list)
    buttons: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    delay_seconds: float = 0.0


class ChannelError(Exception):
    """Base exception for channel-related errors."""
    
    def __init__(self, message: str, channel: str = "", error_code: str = ""):
        super().__init__(message)
        self.channel = channel
        self.error_code = error_code


class BaseChannel(ABC):
    """
    Abstract base class for all communication channels.
    
    Defines the interface that all channels must implement for sending and receiving messages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the channel.
        
        Args:
            config: Channel configuration dictionary
        """
        self.config = config
        self.channel_type = ChannelType(config.get('type', 'web'))
        self.is_connected = False
        self.is_running = False
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name identifier."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the channel and establish connections.
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """
        Start the channel for receiving messages.
        
        Returns:
            bool: True if started successfully
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop the channel and close connections.
        
        Returns:
            bool: True if stopped successfully
        """
        pass
    
    @abstractmethod
    async def send_message(self, message: ChannelResponse, user: ChannelUser) -> bool:
        """
        Send a message through this channel.
        
        Args:
            message: Response message to send
            user: Target user
            
        Returns:
            bool: True if sent successfully
        """
        pass
    
    @abstractmethod
    async def receive_messages(self) -> AsyncGenerator[ChannelMessage, None]:
        """
        Receive messages from this channel.
        
        Yields:
            ChannelMessage: Incoming messages
        """
        pass
    
    async def validate_message(self, message: ChannelMessage) -> bool:
        """
        Validate an incoming message.
        
        Args:
            message: Message to validate
            
        Returns:
            bool: True if message is valid
        """
        if not message.text and not message.attachments:
            return False
        
        if not message.user or not message.user.user_id:
            return False
        
        return True
    
    async def format_response(self, response: ChannelResponse) -> ChannelResponse:
        """
        Format a response for this channel's requirements.
        
        Args:
            response: Original response
            
        Returns:
            ChannelResponse: Formatted response
        """
        # Default implementation - override in subclasses for channel-specific formatting
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get channel status information.
        
        Returns:
            Dict containing status data
        """
        return {
            'name': self.name,
            'type': self.channel_type.value,
            'connected': self.is_connected,
            'running': self.is_running,
            'config': {k: v for k, v in self.config.items() if not k.startswith('_')}
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get channel configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update channel configuration.
        
        Args:
            new_config: New configuration data
        """
        self.config.update(new_config)


class ChannelManager:
    """
    Manager for multiple communication channels.
    
    Handles registration, routing, and coordination of different channels.
    """
    
    def __init__(self):
        """Initialize the channel manager."""
        self.channels: Dict[str, BaseChannel] = {}
        self.is_running = False
    
    async def register_channel(self, channel: BaseChannel) -> bool:
        """
        Register a channel with the manager.
        
        Args:
            channel: Channel instance to register
            
        Returns:
            bool: True if registered successfully
        """
        try:
            channel_name = channel.name
            if channel_name in self.channels:
                raise ChannelError(f"Channel {channel_name} already registered")
            
            await channel.initialize()
            self.channels[channel_name] = channel
            return True
            
        except Exception as e:
            raise ChannelError(f"Failed to register channel: {e}")
    
    async def unregister_channel(self, channel_name: str) -> bool:
        """
        Unregister a channel from the manager.
        
        Args:
            channel_name: Name of channel to unregister
            
        Returns:
            bool: True if unregistered successfully
        """
        if channel_name not in self.channels:
            return False
        
        try:
            channel = self.channels[channel_name]
            await channel.stop()
            del self.channels[channel_name]
            return True
            
        except Exception:
            return False
    
    async def start_all(self) -> bool:
        """
        Start all registered channels.
        
        Returns:
            bool: True if all started successfully
        """
        try:
            for channel in self.channels.values():
                await channel.start()
            
            self.is_running = True
            return True
            
        except Exception:
            return False
    
    async def stop_all(self) -> bool:
        """
        Stop all registered channels.
        
        Returns:
            bool: True if all stopped successfully
        """
        try:
            for channel in self.channels.values():
                await channel.stop()
            
            self.is_running = False
            return True
            
        except Exception:
            return False
    
    async def send_message(self, channel_name: str, message: ChannelResponse, user: ChannelUser) -> bool:
        """
        Send message through specific channel.
        
        Args:
            channel_name: Name of channel to use
            message: Message to send
            user: Target user
            
        Returns:
            bool: True if sent successfully
        """
        if channel_name not in self.channels:
            return False
        
        channel = self.channels[channel_name]
        return await channel.send_message(message, user)
    
    async def broadcast_message(self, message: ChannelResponse, user: ChannelUser) -> int:
        """
        Broadcast message to all channels.
        
        Args:
            message: Message to broadcast
            user: Target user
            
        Returns:
            int: Number of channels that successfully sent the message
        """
        success_count = 0
        
        for channel in self.channels.values():
            try:
                if await channel.send_message(message, user):
                    success_count += 1
            except Exception:
                continue
        
        return success_count
    
    def get_channels(self) -> List[str]:
        """Get list of registered channel names."""
        return list(self.channels.keys())
    
    def get_channel(self, channel_name: str) -> Optional[BaseChannel]:
        """
        Get specific channel instance.
        
        Args:
            channel_name: Name of channel
            
        Returns:
            BaseChannel instance or None if not found
        """
        return self.channels.get(channel_name)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all channels.
        
        Returns:
            Dict containing status information
        """
        return {
            'running': self.is_running,
            'channel_count': len(self.channels),
            'channels': {name: channel.get_status() for name, channel in self.channels.items()}
        }
