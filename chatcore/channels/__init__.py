"""
Channels module for Universal Chatbot Framework

This module contains all channel implementations for different communication platforms.
"""

from .base_channel import (
    BaseChannel, 
    ChannelMessage, 
    ChannelResponse, 
    ChannelType, 
    MessageType, 
    ChannelUser, 
    MessageAttachment, 
    ChannelError
)
from .web import WebChannel
from .whatsapp import WhatsAppChannel
from .voice import VoiceChannel
from .mobile import MobileChannel

__all__ = [
    "BaseChannel",
    "ChannelMessage", 
    "ChannelResponse", 
    "ChannelType", 
    "MessageType", 
    "ChannelUser", 
    "MessageAttachment", 
    "ChannelError",
    "WebChannel",
    "WhatsAppChannel", 
    "VoiceChannel",
    "MobileChannel"
]
