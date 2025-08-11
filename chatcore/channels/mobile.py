"""
Mobile Channel Implementation for Universal Chatbot Framework

This module provides mobile channel implementation for handling mobile app interactions,
push notifications, and mobile-specific features through various mobile platforms.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid

from .base_channel import (
    BaseChannel, ChannelMessage, ChannelResponse, ChannelType, 
    MessageType, ChannelUser, MessageAttachment, ChannelError
)

# Configure logging
logger = logging.getLogger(__name__)


class MobileSession:
    """Manages a mobile app session."""
    
    def __init__(self, session_id: str, user_id: str, device_info: Dict[str, Any]):
        self.session_id = session_id
        self.user_id = user_id
        self.device_info = device_info
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        self.push_token = device_info.get("push_token")
        self.app_version = device_info.get("app_version")
        self.platform = device_info.get("platform", "unknown")
        
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        
    def is_session_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session is expired."""
        duration = (datetime.utcnow() - self.last_activity).total_seconds()
        return duration > (timeout_minutes * 60)


class MobileChannel(BaseChannel):
    """
    Mobile channel implementation for mobile app interactions.
    
    Supports in-app messaging, push notifications, mobile-specific features,
    and integration with mobile SDKs and push notification services.
    """
    
    def __init__(self, channel_config: Dict[str, Any]):
        """
        Initialize the mobile channel.
        
        Args:
            channel_config: Configuration including push notification settings, SDK config, etc.
        """
        super().__init__(channel_config)
        
        # Push Notification Configuration
        self.push_provider = channel_config.get("push_provider", "fcm")  # fcm, apns, expo
        self.fcm_server_key = channel_config.get("fcm_server_key")
        self.fcm_sender_id = channel_config.get("fcm_sender_id")
        self.apns_key_id = channel_config.get("apns_key_id")
        self.apns_team_id = channel_config.get("apns_team_id")
        self.apns_bundle_id = channel_config.get("apns_bundle_id")
        self.expo_access_token = channel_config.get("expo_access_token")
        
        # Mobile App Configuration
        self.app_id = channel_config.get("app_id")
        self.api_key = channel_config.get("api_key")
        self.webhook_url = channel_config.get("webhook_url")
        self.deep_link_scheme = channel_config.get("deep_link_scheme")
        
        # Session Configuration
        self.session_timeout = channel_config.get("session_timeout_minutes", 30)
        self.max_message_length = channel_config.get("max_message_length", 2000)
        
        # Session management
        self.active_sessions: Dict[str, MobileSession] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Background tasks
        self._cleanup_task = None
        
    def _get_channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.MOBILE
    
    async def initialize(self) -> bool:
        """Initialize the mobile channel."""
        try:
            # Test push notification service
            if self.push_provider == "fcm" and self.fcm_server_key:
                test_result = await self._test_fcm_connection()
                if not test_result:
                    logger.warning("FCM connection test failed")
            
            # Start session cleanup task
            self._cleanup_task = asyncio.create_task(self._session_cleanup_loop())
            
            self.is_active = True
            logger.info("Mobile channel initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize mobile channel: {e}")
            return False
    
    async def _test_fcm_connection(self) -> bool:
        """Test Firebase Cloud Messaging connection."""
        try:
            import aiohttp
            
            url = "https://fcm.googleapis.com/fcm/send"
            headers = {
                "Authorization": f"key={self.fcm_server_key}",
                "Content-Type": "application/json"
            }
            
            # Test payload (won't be delivered due to invalid token)
            payload = {
                "to": "test_token",
                "data": {"test": "connection"}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    # We expect this to fail with invalid token, but valid auth
                    result = await response.json()
                    if "error" in result and "InvalidRegistration" in str(result):
                        logger.info("FCM connection test successful")
                        return True
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"FCM connection test error: {e}")
            return False
    
    async def send_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """
        Send a response message through the mobile channel.
        
        Args:
            response: The response to send
            recipient_id: Session ID or user ID
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            # Format response for mobile
            formatted_response = self.format_response_for_channel(response)
            
            # Get session
            session = self.active_sessions.get(recipient_id)
            if not session:
                logger.warning(f"No active session found for recipient: {recipient_id}")
                # Try to send as push notification
                return await self._send_push_notification(formatted_response, recipient_id)
            
            session.update_activity()
            
            # Send in-app message if session is active
            if session.is_active and not session.is_session_expired(self.session_timeout):
                success = await self._send_in_app_message(formatted_response, session)
                if success:
                    return True
            
            # Fallback to push notification
            return await self._send_push_notification(formatted_response, recipient_id, session)
            
        except Exception as e:
            logger.error(f"Error sending mobile message: {e}")
            return False
    
    async def _send_in_app_message(self, response: ChannelResponse, session: MobileSession) -> bool:
        """Send in-app message to active session."""
        try:
            message_data = {
                "type": "chat_message",
                "content": response.content,
                "response_id": response.response_id,
                "timestamp": response.timestamp.isoformat(),
                "message_type": response.message_type.value,
                "quick_replies": response.quick_replies,
                "suggested_actions": response.suggested_actions,
                "requires_human_handoff": response.requires_human_handoff,
                "confidence_score": response.confidence_score,
                "attachments": [
                    {
                        "id": att.attachment_id,
                        "type": att.attachment_type.value,
                        "url": att.url,
                        "filename": att.filename,
                        "mime_type": att.mime_type,
                        "size": att.size
                    }
                    for att in response.attachments
                ]
            }
            
            # Send via WebSocket, Server-Sent Events, or HTTP polling
            # For this implementation, we'll use a webhook callback
            if self.webhook_url and self.webhook_url is not None:
                return await self._send_webhook_message(message_data, session)
            
            # Store for polling-based retrieval
            await self._store_message_for_polling(message_data, session)
            return True
            
        except Exception as e:
            logger.error(f"Error sending in-app message: {e}")
            return False
    
    async def _send_webhook_message(self, message_data: Dict[str, Any], session: MobileSession) -> bool:
        """Send message via webhook to mobile app."""
        try:
            import aiohttp
            
            if not self.webhook_url:
                logger.error("Webhook URL is not configured")
                return False
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "X-Session-ID": session.session_id
            }
            
            payload = {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "message": message_data
            }
            
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    str(self.webhook_url),
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"In-app message sent successfully to session {session.session_id}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending webhook message: {e}")
            return False
    
    async def _store_message_for_polling(self, message_data: Dict[str, Any], session: MobileSession):
        """Store message for polling-based retrieval."""
        # This would typically store in Redis, database, or in-memory cache
        # For now, just log that message was stored
        logger.info(f"Message stored for polling retrieval: session {session.session_id}")
    
    async def _send_push_notification(self, response: ChannelResponse, recipient_id: str, session: Optional[MobileSession] = None) -> bool:
        """Send push notification to mobile device."""
        try:
            if self.push_provider == "fcm":
                return await self._send_fcm_notification(response, recipient_id, session)
            elif self.push_provider == "apns":
                return await self._send_apns_notification(response, recipient_id, session)
            elif self.push_provider == "expo":
                return await self._send_expo_notification(response, recipient_id, session)
            else:
                logger.error(f"Unsupported push provider: {self.push_provider}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False
    
    async def _send_fcm_notification(self, response: ChannelResponse, recipient_id: str, session: Optional[MobileSession] = None) -> bool:
        """Send Firebase Cloud Messaging notification."""
        try:
            import aiohttp
            
            if not session or not session.push_token:
                logger.warning(f"No push token available for recipient: {recipient_id}")
                return False
            
            url = "https://fcm.googleapis.com/fcm/send"
            headers = {
                "Authorization": f"key={self.fcm_server_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare notification payload
            notification_data = {
                "title": "New Message",
                "body": response.content[:100] + "..." if len(response.content) > 100 else response.content,
                "icon": "ic_notification",
                "sound": "default",
                "click_action": "FLUTTER_NOTIFICATION_CLICK"
            }
            
            # Prepare data payload
            data_payload = {
                "response_id": response.response_id,
                "timestamp": response.timestamp.isoformat(),
                "message_type": response.message_type.value,
                "session_id": session.session_id if session else "",
                "requires_human_handoff": str(response.requires_human_handoff)
            }
            
            # Add deep link if configured
            if self.deep_link_scheme:
                data_payload["deep_link"] = f"{self.deep_link_scheme}://chat/{session.session_id if session else recipient_id}"
            
            payload = {
                "to": session.push_token,
                "notification": notification_data,
                "data": data_payload,
                "priority": "high"
            }
            
            async with aiohttp.ClientSession() as client:
                async with client.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success") == 1:
                            logger.info(f"FCM notification sent successfully to {recipient_id}")
                            return True
                        else:
                            logger.error(f"FCM notification failed: {result}")
                            return False
                    else:
                        logger.error(f"FCM request failed with status {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending FCM notification: {e}")
            return False
    
    async def _send_apns_notification(self, response: ChannelResponse, recipient_id: str, session: Optional[MobileSession] = None) -> bool:
        """Send Apple Push Notification Service notification."""
        # Implementation would use aioapns or similar library
        logger.info(f"APNS notification would be sent to {recipient_id}")
        return True
    
    async def _send_expo_notification(self, response: ChannelResponse, recipient_id: str, session: Optional[MobileSession] = None) -> bool:
        """Send Expo push notification."""
        # Implementation would use Expo push API
        logger.info(f"Expo notification would be sent to {recipient_id}")
        return True
    
    async def receive_message(self) -> Optional[ChannelMessage]:
        """Receive a message from the message queue."""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving mobile message: {e}")
            return None
    
    async def process_mobile_message(self, message_data: Dict[str, Any]) -> Optional[ChannelMessage]:
        """
        Process incoming message from mobile app.
        
        Args:
            message_data: Raw message data from mobile app
            
        Returns:
            Optional[ChannelMessage]: Parsed message or None
        """
        try:
            # Extract message information
            user_id = message_data.get("user_id")
            session_id = message_data.get("session_id")
            content = message_data.get("content", "")
            message_type = message_data.get("message_type", "text")
            device_info = message_data.get("device_info", {})
            attachments_data = message_data.get("attachments", [])
            
            if not user_id:
                logger.warning("Missing user_id in mobile message")
                return None
            
            # Get or create session
            if session_id:
                session = self.active_sessions.get(session_id)
                if not session:
                    session = MobileSession(session_id, user_id, device_info)
                    self.active_sessions[session_id] = session
                session.update_activity()
            else:
                session_id = str(uuid.uuid4())
                session = MobileSession(session_id, user_id, device_info)
                self.active_sessions[session_id] = session
            
            # Process attachments
            attachments = []
            for att_data in attachments_data:
                attachment = MessageAttachment(
                    attachment_type=MessageType(att_data.get("type", "document")) if att_data.get("type") in MessageType.__members__.values() else MessageType.DOCUMENT,
                    url=att_data.get("url", ""),
                    filename=att_data.get("filename"),
                    mime_type=att_data.get("mime_type"),
                    size=att_data.get("size")
                )
                attachments.append(attachment)
            
            # Create channel user
            user = ChannelUser(
                user_id=user_id,
                channel_user_id=user_id,
                name=device_info.get("user_name"),
                metadata={
                    "device_id": device_info.get("device_id"),
                    "platform": device_info.get("platform"),
                    "app_version": device_info.get("app_version"),
                    "push_token": device_info.get("push_token"),
                    "channel_type": ChannelType.MOBILE.value
                }
            )
            
            # Create channel message
            message = ChannelMessage(
                message_id=message_data.get("message_id", str(uuid.uuid4())),
                user=user,
                content=content,
                message_type=MessageType(message_type) if message_type in MessageType.__members__.values() else MessageType.TEXT,
                channel_type=ChannelType.MOBILE,
                session_id=session_id,
                attachments=attachments,
                metadata={
                    "device_info": device_info,
                    "session_info": {
                        "started_at": session.started_at.isoformat(),
                        "platform": session.platform
                    }
                }
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing mobile message: {e}")
            return None
    
    async def validate_message(self, message: ChannelMessage) -> bool:
        """Validate an incoming mobile message."""
        try:
            # Basic validation
            if not message.user.user_id:
                logger.warning("Missing user ID in mobile message")
                return False
            
            if len(message.content) > self.max_message_length:
                logger.warning(f"Message too long: {len(message.content)} > {self.max_message_length}")
                return False
            
            # Validate session if provided
            if message.session_id:
                session = self.active_sessions.get(message.session_id)
                if session and session.user_id != message.user.user_id:
                    logger.warning("Session user mismatch")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating mobile message: {e}")
            return False
    
    async def register_device(self, user_id: str, device_info: Dict[str, Any]) -> str:
        """
        Register a mobile device for push notifications.
        
        Args:
            user_id: User identifier
            device_info: Device information including push token
            
        Returns:
            str: Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            session = MobileSession(session_id, user_id, device_info)
            self.active_sessions[session_id] = session
            
            logger.info(f"Registered mobile device for user {user_id}, session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error registering mobile device: {e}")
            return ""
    
    async def unregister_device(self, session_id: str) -> bool:
        """
        Unregister a mobile device.
        
        Args:
            session_id: Session to unregister
            
        Returns:
            bool: True if unregistered successfully
        """
        try:
            session = self.active_sessions.get(session_id)
            if session:
                session.is_active = False
                del self.active_sessions[session_id]
                logger.info(f"Unregistered mobile device session: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering mobile device: {e}")
            return False
    
    async def _session_cleanup_loop(self):
        """Background task to clean up expired sessions."""
        while self.is_active:
            try:
                expired_sessions = []
                for session_id, session in self.active_sessions.items():
                    if session.is_session_expired(self.session_timeout):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.unregister_device(session_id)
                    logger.info(f"Cleaned up expired session: {session_id}")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def get_capabilities(self) -> List[str]:
        """Get mobile channel capabilities."""
        return [
            "text_messages",
            "image_attachments",
            "file_attachments",
            "push_notifications",
            "in_app_messaging",
            "quick_replies",
            "suggested_actions",
            "deep_linking",
            "session_management",
            "device_registration",
            "offline_messaging"
        ]
    
    def format_response_for_channel(self, response: ChannelResponse) -> ChannelResponse:
        """Format response for mobile channel requirements."""
        # Limit message length
        if len(response.content) > self.max_message_length:
            response.content = response.content[:self.max_message_length-3] + "..."
        
        # Ensure quick replies don't exceed mobile UI limits
        if len(response.quick_replies) > 4:
            response.quick_replies = response.quick_replies[:4]
        
        response.channel_type = ChannelType.MOBILE
        return response
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a mobile session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "device_info": session.device_info,
                "started_at": session.started_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "is_active": session.is_active,
                "platform": session.platform,
                "app_version": session.app_version,
                "has_push_token": bool(session.push_token)
            }
            
        except Exception as e:
            logger.error(f"Error getting mobile session info: {e}")
            return None
    
    async def close(self) -> bool:
        """Close the mobile channel and cleanup resources."""
        try:
            # Stop cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear all sessions
            self.active_sessions.clear()
            
            # Clear message queue
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except:
                    break
            
            self.is_active = False
            logger.info("Mobile channel closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing mobile channel: {e}")
            return False