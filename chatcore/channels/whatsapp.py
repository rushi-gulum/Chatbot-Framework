"""
WhatsApp Channel Implementation for Universal Chatbot Framework

This module provides WhatsApp Business API integration for handling WhatsApp messages
through the official WhatsApp Business Platform API.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import base64

import aiohttp
from twilio.rest import Client as TwilioClient
import requests

from .base_channel import (
    BaseChannel, ChannelMessage, ChannelResponse, ChannelType, 
    MessageType, ChannelUser, MessageAttachment, ChannelError
)

# Configure logging
logger = logging.getLogger(__name__)


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp channel implementation using WhatsApp Business API.
    
    Supports both Meta's WhatsApp Business Platform API and Twilio's WhatsApp API
    for sending and receiving WhatsApp messages.
    """
    
    def __init__(self, channel_config: Dict[str, Any]):
        """
        Initialize the WhatsApp channel.
        
        Args:
            channel_config: Configuration including API credentials, webhook settings, etc.
        """
        super().__init__(channel_config)
        
        # API Configuration
        self.provider = channel_config.get("provider", "meta")  # "meta" or "twilio"
        self.verify_token = channel_config.get("verify_token")
        self.webhook_url = channel_config.get("webhook_url")
        
        # Meta WhatsApp Business API
        if self.provider == "meta":
            self.access_token = channel_config.get("access_token")
            self.phone_number_id = channel_config.get("phone_number_id")
            self.business_account_id = channel_config.get("business_account_id")
            self.app_secret = channel_config.get("app_secret")
            self.base_url = "https://graph.facebook.com/v18.0"
        
        # Twilio WhatsApp API
        elif self.provider == "twilio":
            self.account_sid = channel_config.get("account_sid")
            self.auth_token = channel_config.get("auth_token")
            self.twilio_phone_number = channel_config.get("twilio_phone_number")
            self.twilio_client = None
        
        # Message queue
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def _get_channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.WHATSAPP
    
    async def initialize(self) -> bool:
        """Initialize the WhatsApp channel."""
        try:
            if self.provider == "twilio" and self.account_sid and self.auth_token:
                self.twilio_client = TwilioClient(self.account_sid, self.auth_token)
                
                # Test Twilio connection
                try:
                    account = self.twilio_client.api.accounts(self.account_sid).fetch()
                    logger.info(f"Twilio WhatsApp channel initialized for account: {account.friendly_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize Twilio client: {e}")
                    return False
            
            elif self.provider == "meta" and self.access_token and self.phone_number_id:
                # Test Meta API connection
                try:
                    await self._test_meta_connection()
                    logger.info("Meta WhatsApp Business API channel initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Meta WhatsApp API: {e}")
                    return False
            
            else:
                logger.error("Invalid WhatsApp configuration")
                return False
            
            self.is_active = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WhatsApp channel: {e}")
            return False
    
    async def _test_meta_connection(self):
        """Test Meta WhatsApp Business API connection."""
        url = f"{self.base_url}/{self.phone_number_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise ChannelError(
                        f"Meta API test failed with status {response.status}",
                        error_code="META_API_ERROR",
                        channel_type=ChannelType.WHATSAPP
                    )
    
    async def send_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """
        Send a response message through WhatsApp.
        
        Args:
            response: The response to send
            recipient_id: WhatsApp phone number (with country code)
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            # Format response for WhatsApp
            formatted_response = self.format_response_for_channel(response)
            
            if self.provider == "meta":
                return await self._send_meta_message(formatted_response, recipient_id)
            elif self.provider == "twilio":
                return await self._send_twilio_message(formatted_response, recipient_id)
            else:
                logger.error(f"Unknown WhatsApp provider: {self.provider}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return False
    
    async def _send_meta_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """Send message using Meta WhatsApp Business API."""
        try:
            url = f"{self.base_url}/{self.phone_number_id}/messages"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Prepare message payload
            payload = {
                "messaging_product": "whatsapp",
                "to": recipient_id,
                "type": "text",
                "text": {
                    "body": response.content
                }
            }
            
            # Add quick replies if available
            if response.quick_replies:
                payload["type"] = "interactive"
                payload["interactive"] = {
                    "type": "button",
                    "body": {
                        "text": response.content
                    },
                    "action": {
                        "buttons": [
                            {
                                "type": "reply",
                                "reply": {
                                    "id": f"quick_reply_{i}",
                                    "title": reply[:20]  # WhatsApp button title limit
                                }
                            }
                            for i, reply in enumerate(response.quick_replies[:3])  # Max 3 buttons
                        ]
                    }
                }
                del payload["text"]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        message_id = result.get("messages", [{}])[0].get("id")
                        logger.info(f"Meta WhatsApp message sent successfully: {message_id}")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.error(f"Meta WhatsApp API error {resp.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Meta WhatsApp message: {e}")
            return False
    
    async def _send_twilio_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """Send message using Twilio WhatsApp API."""
        try:
            if not self.twilio_client:
                logger.error("Twilio client not initialized")
                return False
                
            # Format phone numbers
            from_number = f"whatsapp:{self.twilio_phone_number}"
            to_number = f"whatsapp:{recipient_id}"
            
            message = self.twilio_client.messages.create(
                body=response.content,
                from_=from_number,
                to=to_number
            )
            
            logger.info(f"Twilio WhatsApp message sent successfully: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Twilio WhatsApp message: {e}")
            return False
    
    async def receive_message(self) -> Optional[ChannelMessage]:
        """Receive a message from the message queue."""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving WhatsApp message: {e}")
            return None
    
    async def process_webhook(self, webhook_data: Dict[str, Any]) -> Optional[ChannelMessage]:
        """
        Process incoming webhook data from WhatsApp.
        
        Args:
            webhook_data: Raw webhook data from WhatsApp
            
        Returns:
            Optional[ChannelMessage]: Parsed message or None
        """
        try:
            if self.provider == "meta":
                return await self._process_meta_webhook(webhook_data)
            elif self.provider == "twilio":
                return await self._process_twilio_webhook(webhook_data)
            else:
                logger.error(f"Unknown provider for webhook processing: {self.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing WhatsApp webhook: {e}")
            return None
    
    async def _process_meta_webhook(self, webhook_data: Dict[str, Any]) -> Optional[ChannelMessage]:
        """Process Meta WhatsApp Business API webhook."""
        try:
            entry = webhook_data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            
            messages = value.get("messages", [])
            if not messages:
                return None
            
            message_data = messages[0]
            
            # Extract message information
            whatsapp_message_id = message_data.get("id")
            sender_phone = message_data.get("from")
            timestamp = datetime.fromtimestamp(int(message_data.get("timestamp", 0)))
            
            # Get message content
            message_type = message_data.get("type", "text")
            content = ""
            attachments = []
            
            if message_type == "text":
                content = message_data.get("text", {}).get("body", "")
            elif message_type == "image":
                image_data = message_data.get("image", {})
                content = image_data.get("caption", "[Image]")
                attachment = MessageAttachment(
                    attachment_type=MessageType.IMAGE,
                    url=image_data.get("url", ""),
                    mime_type=image_data.get("mime_type")
                )
                attachments.append(attachment)
            elif message_type == "audio":
                audio_data = message_data.get("audio", {})
                content = "[Audio Message]"
                attachment = MessageAttachment(
                    attachment_type=MessageType.AUDIO,
                    url=audio_data.get("url", ""),
                    mime_type=audio_data.get("mime_type")
                )
                attachments.append(attachment)
            
            # Create channel user
            user = ChannelUser(
                user_id=sender_phone,
                channel_user_id=sender_phone,
                name=value.get("contacts", [{}])[0].get("profile", {}).get("name"),
                metadata={
                    "channel_type": ChannelType.WHATSAPP.value,
                    "provider": "meta"
                }
            )
            
            # Create channel message
            message = ChannelMessage(
                message_id=whatsapp_message_id,
                user=user,
                content=content,
                message_type=MessageType(message_type) if message_type in MessageType.__members__.values() else MessageType.TEXT,
                channel_type=ChannelType.WHATSAPP,
                timestamp=timestamp,
                attachments=attachments,
                metadata={
                    "whatsapp_message_id": whatsapp_message_id,
                    "provider": "meta"
                }
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing Meta webhook: {e}")
            return None
    
    async def _process_twilio_webhook(self, webhook_data: Dict[str, Any]) -> Optional[ChannelMessage]:
        """Process Twilio WhatsApp webhook."""
        try:
            # Extract Twilio webhook data
            message_sid = webhook_data.get("MessageSid")
            sender_phone = webhook_data.get("From", "").replace("whatsapp:", "")
            content = webhook_data.get("Body", "")
            num_media = int(webhook_data.get("NumMedia", 0))
            
            # Handle media attachments
            attachments = []
            if num_media > 0:
                for i in range(num_media):
                    media_url = webhook_data.get(f"MediaUrl{i}")
                    media_type = webhook_data.get(f"MediaContentType{i}")
                    
                    if media_url:
                        # Map media type to MessageType
                        if media_type and media_type.startswith("image"):
                            attachment_type = MessageType.IMAGE
                        elif media_type and media_type.startswith("audio"):
                            attachment_type = MessageType.AUDIO
                        elif media_type and media_type.startswith("video"):
                            attachment_type = MessageType.VIDEO
                        else:
                            attachment_type = MessageType.DOCUMENT
                            
                        attachment = MessageAttachment(
                            attachment_type=attachment_type,
                            url=media_url,
                            mime_type=media_type
                        )
                        attachments.append(attachment)
            
            # Create channel user
            user = ChannelUser(
                user_id=sender_phone,
                channel_user_id=sender_phone,
                name=webhook_data.get("ProfileName"),
                metadata={
                    "channel_type": ChannelType.WHATSAPP.value,
                    "provider": "twilio"
                }
            )
            
            # Create channel message
            message = ChannelMessage(
                message_id=message_sid or str(uuid.uuid4()),
                user=user,
                content=content,
                message_type=MessageType.IMAGE if attachments else MessageType.TEXT,
                channel_type=ChannelType.WHATSAPP,
                timestamp=datetime.utcnow(),
                attachments=attachments,
                metadata={
                    "twilio_message_sid": message_sid,
                    "provider": "twilio"
                }
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing Twilio webhook: {e}")
            return None
    
    async def validate_message(self, message: ChannelMessage) -> bool:
        """Validate an incoming WhatsApp message."""
        try:
            # Basic validation
            if not message.user.channel_user_id:
                logger.warning("Missing WhatsApp phone number")
                return False
            
            # Validate phone number format (basic check)
            phone = message.user.channel_user_id.replace("+", "")
            if not phone.isdigit() or len(phone) < 8:
                logger.warning(f"Invalid phone number format: {message.user.channel_user_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating WhatsApp message: {e}")
            return False
    
    async def get_user_info(self, channel_user_id: str) -> Optional[Dict[str, Any]]:
        """Get WhatsApp user profile information."""
        try:
            if self.provider == "meta":
                # Meta doesn't provide user profile API for WhatsApp Business
                return {
                    "phone_number": channel_user_id,
                    "platform": "whatsapp_business"
                }
            elif self.provider == "twilio":
                # Twilio doesn't provide detailed user profiles
                return {
                    "phone_number": channel_user_id,
                    "platform": "twilio_whatsapp"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting WhatsApp user info: {e}")
            return None
    
    def get_capabilities(self) -> List[str]:
        """Get WhatsApp channel capabilities."""
        capabilities = [
            "text_messages",
            "image_attachments",
            "audio_attachments",
            "document_attachments",
            "quick_replies",
            "delivery_receipts",
            "read_receipts"
        ]
        
        if self.provider == "meta":
            capabilities.extend([
                "interactive_buttons",
                "interactive_lists",
                "template_messages",
                "location_sharing"
            ])
        
        return capabilities
    
    async def mark_as_read(self, message_id: str) -> bool:
        """Mark a WhatsApp message as read."""
        try:
            if self.provider == "meta":
                url = f"{self.base_url}/{self.phone_number_id}/messages"
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "messaging_product": "whatsapp",
                    "status": "read",
                    "message_id": message_id
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        return resp.status == 200
            
            # Twilio doesn't support read receipts
            return True
            
        except Exception as e:
            logger.error(f"Error marking WhatsApp message as read: {e}")
            return False
    
    def format_response_for_channel(self, response: ChannelResponse) -> ChannelResponse:
        """Format response for WhatsApp requirements."""
        # WhatsApp message length limit
        if len(response.content) > 4096:
            response.content = response.content[:4093] + "..."
        
        # Remove unsupported formatting
        response.content = response.content.replace("**", "*").replace("__", "_")
        
        response.channel_type = ChannelType.WHATSAPP
        return response
    
    async def verify_webhook(self, verify_token: str, challenge: str) -> Optional[str]:
        """
        Verify webhook for Meta WhatsApp Business API.
        
        Args:
            verify_token: Token from webhook verification request
            challenge: Challenge string to return
            
        Returns:
            Optional[str]: Challenge string if verification successful
        """
        if verify_token == self.verify_token:
            logger.info("WhatsApp webhook verified successfully")
            return challenge
        else:
            logger.warning("WhatsApp webhook verification failed")
            return None