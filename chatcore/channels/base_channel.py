"""
Base Channel Implementation for Universal Chatbot Framework

This module provides the base classes and interfaces for all channel implementations
in the universal chatbot framework.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Enumeration of supported channel types."""
    WEB = "web"
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    MOBILE = "mobile"
    TELEGRAM = "telegram"
    SLACK = "slack"
    SMS = "sms"
    EMAIL = "email"


class MessageType(Enum):
    """Enumeration of message types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    LOCATION = "location"
    CONTACT = "contact"
    STICKER = "sticker"
    QUICK_REPLY = "quick_reply"
    INTERACTIVE = "interactive"


class ChannelError(Exception):
    """Base exception for channel-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, channel_type: Optional[ChannelType] = None):
        super().__init__(message)
        self.error_code = error_code
        self.channel_type = channel_type
        self.timestamp = datetime.utcnow()


class MessageAttachment:
    """Represents a message attachment (file, image, etc.)."""
    
    def __init__(
        self,
        attachment_type: MessageType,
        url: Optional[str] = None,
        data: Optional[bytes] = None,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        size: Optional[int] = None
    ):
        self.attachment_id = str(uuid.uuid4())
        self.attachment_type = attachment_type
        self.url = url
        self.data = data
        self.filename = filename
        self.mime_type = mime_type
        self.size = size
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert attachment to dictionary."""
        return {
            "attachment_id": self.attachment_id,
            "type": self.attachment_type.value,
            "url": self.url,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "created_at": self.created_at.isoformat()
        }


class ChannelUser:
    """Represents a user in a channel."""
    
    def __init__(
        self,
        user_id: str,
        channel_user_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.channel_user_id = channel_user_id
        self.name = name
        self.email = email
        self.phone = phone
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "user_id": self.user_id,
            "channel_user_id": self.channel_user_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class ChannelMessage:
    """Represents a message received from a channel."""
    
    def __init__(
        self,
        message_id: str,
        content: str,
        message_type: MessageType,
        user: ChannelUser,
        channel_type: ChannelType,
        session_id: Optional[str] = None,
        attachments: Optional[List[MessageAttachment]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.message_id = message_id
        self.content = content
        self.message_type = message_type
        self.user = user
        self.channel_type = channel_type
        self.session_id = session_id or str(uuid.uuid4())
        self.attachments = attachments or []
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.utcnow()
    
    @classmethod
    def create(
        cls,
        content: str,
        user_id: str,
        channel_type: ChannelType,
        channel_user_id: Optional[str] = None,
        message_type: MessageType = MessageType.TEXT,
        session_id: Optional[str] = None,
        attachments: Optional[List[MessageAttachment]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ChannelMessage":
        """Create a new channel message."""
        user = ChannelUser(
            user_id=user_id,
            channel_user_id=channel_user_id or user_id
        )
        
        return cls(
            message_id=str(uuid.uuid4()),
            content=content,
            message_type=message_type,
            user=user,
            channel_type=channel_type,
            session_id=session_id,
            attachments=attachments,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "content": self.content,
            "message_type": self.message_type.value,
            "user": self.user.to_dict(),
            "channel_type": self.channel_type.value,
            "session_id": self.session_id,
            "attachments": [att.to_dict() for att in self.attachments],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ChannelResponse:
    """Represents a response to be sent through a channel."""
    
    def __init__(
        self,
        response_id: str,
        content: str,
        channel_type: ChannelType,
        message_type: MessageType = MessageType.TEXT,
        quick_replies: Optional[List[str]] = None,
        suggested_actions: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[MessageAttachment]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        requires_human_handoff: bool = False,
        confidence_score: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.response_id = response_id
        self.content = content
        self.channel_type = channel_type
        self.message_type = message_type
        self.quick_replies = quick_replies or []
        self.suggested_actions = suggested_actions or []
        self.attachments = attachments or []
        self.metadata = metadata or {}
        self.requires_human_handoff = requires_human_handoff
        self.confidence_score = confidence_score
        self.timestamp = timestamp or datetime.utcnow()
    
    @classmethod
    def create(
        cls,
        content: str,
        channel_type: ChannelType,
        message_type: MessageType = MessageType.TEXT,
        quick_replies: Optional[List[str]] = None,
        suggested_actions: Optional[List[Dict[str, Any]]] = None,
        requires_human_handoff: bool = False,
        confidence_score: Optional[float] = None
    ) -> "ChannelResponse":
        """Create a new channel response."""
        return cls(
            response_id=str(uuid.uuid4()),
            content=content,
            channel_type=channel_type,
            message_type=message_type,
            quick_replies=quick_replies,
            suggested_actions=suggested_actions,
            requires_human_handoff=requires_human_handoff,
            confidence_score=confidence_score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "response_id": self.response_id,
            "content": self.content,
            "channel_type": self.channel_type.value,
            "message_type": self.message_type.value,
            "quick_replies": self.quick_replies,
            "suggested_actions": self.suggested_actions,
            "attachments": [att.to_dict() for att in self.attachments],
            "metadata": self.metadata,
            "requires_human_handoff": self.requires_human_handoff,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }


class BaseChannel(ABC):
    """
    Abstract base class for all channel implementations.
    
    Defines the common interface that all channels must implement
    for sending and receiving messages.
    """
    
    def __init__(self, channel_config: Dict[str, Any]):
        """
        Initialize the base channel.
        
        Args:
            channel_config: Configuration dictionary for the channel
        """
        self.config = channel_config
        self.is_active = False
        self.channel_type = self._get_channel_type()
        self.created_at = datetime.utcnow()
        
        # Common configuration
        self.max_retries = channel_config.get("max_retries", 3)
        self.retry_delay = channel_config.get("retry_delay", 1.0)
        self.timeout = channel_config.get("timeout", 30.0)
        
        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.errors_count = 0
    
    @abstractmethod
    def _get_channel_type(self) -> ChannelType:
        """Return the specific channel type."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the channel for operation.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def send_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """
        Send a message through the channel.
        
        Args:
            response: The response to send
            recipient_id: The recipient identifier
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[ChannelMessage]:
        """
        Receive a message from the channel.
        
        Returns:
            ChannelMessage: The received message, or None if no message available
        """
        pass
    
    @abstractmethod
    async def validate_message(self, message: ChannelMessage) -> bool:
        """
        Validate an incoming message.
        
        Args:
            message: The message to validate
            
        Returns:
            bool: True if message is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities supported by this channel.
        
        Returns:
            List[str]: List of capability names
        """
        pass
    
    def format_response_for_channel(self, response: ChannelResponse) -> ChannelResponse:
        """
        Format a response for this specific channel.
        
        Args:
            response: The response to format
            
        Returns:
            ChannelResponse: The formatted response
        """
        # Default implementation - override in subclasses for channel-specific formatting
        return response
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the channel.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        return {
            "channel_type": self.channel_type.value,
            "is_active": self.is_active,
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "errors_count": self.errors_count,
            "status": "healthy" if self.is_active else "inactive"
        }
    
    async def close(self) -> bool:
        """
        Close the channel and cleanup resources.
        
        Returns:
            bool: True if closed successfully, False otherwise
        """
        self.is_active = False
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get channel metrics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "errors_count": self.errors_count,
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds()
        }
    
    def _increment_sent(self):
        """Increment sent message counter."""
        self.messages_sent += 1
    
    def _increment_received(self):
        """Increment received message counter."""
        self.messages_received += 1
    
    def _increment_errors(self):
        """Increment error counter."""
        self.errors_count += 1
