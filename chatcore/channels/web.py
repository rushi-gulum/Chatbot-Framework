"""
Web Channel Implementation for Universal Chatbot Framework

This module provides the web channel implementation for handling HTTP-based chat interactions
through REST APIs, WebSockets, and web interfaces with enterprise-grade security.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
import aiohttp

from .base_channel import (
    BaseChannel, ChannelMessage, ChannelResponse, ChannelType, 
    MessageType, ChannelUser, ChannelError
)
from .security import (
    SecurityConfig, CorrelationContext, SecureLogger, JWTAuth, 
    RateLimiter, InputSanitizer, WebhookVerifier, CircuitBreaker,
    HTTPSEnforcer, retry_with_backoff
)

# Configure secure logging
secure_logger = SecureLogger(__name__)


class WebChatRequest(BaseModel):
    """Request model for web chat messages with validation."""
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., min_length=1, max_length=50)
    session_id: Optional[str] = Field(None, max_length=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('message')
    def sanitize_message(cls, v):
        return InputSanitizer.sanitize_text(v)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError('User ID cannot be empty')
        return InputSanitizer.sanitize_text(v, max_length=50)


class WebChatResponse(BaseModel):
    """Response model for web chat messages."""
    response: str
    response_id: str
    session_id: str
    timestamp: str
    quick_replies: List[str] = Field(default_factory=list)
    suggested_actions: List[Dict[str, Any]] = Field(default_factory=list)
    requires_human_handoff: bool = False
    confidence_score: Optional[float] = None


class WebSocketConnection:
    """Manages individual WebSocket connections."""
    
    def __init__(self, websocket: WebSocket, user_id: str, session_id: str):
        self.websocket = websocket
        self.user_id = user_id
        self.session_id = session_id
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
    async def send_message(self, message: Dict[str, Any]):
        """Send a message through the WebSocket."""
        try:
            await self.websocket.send_json(message)
            self.last_activity = datetime.utcnow()
        except Exception as e:
            secure_logger.error(f"Error sending WebSocket message: {e}")
            raise
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive a message from the WebSocket."""
        try:
            data = await self.websocket.receive_json()
            self.last_activity = datetime.utcnow()
            return data
        except Exception as e:
            secure_logger.error(f"Error receiving WebSocket message: {e}")
            raise


class WebChannel(BaseChannel):
    """
    Enhanced Web channel implementation for HTTP and WebSocket communication.
    
    Supports both REST API endpoints and real-time WebSocket connections
    with enterprise-grade security features including JWT auth, rate limiting,
    input sanitization, and comprehensive logging.
    """
    
    def __init__(self, channel_config: Dict[str, Any]):
        """
        Initialize the web channel with security enhancements.
        
        Args:
            channel_config: Configuration including host, port, security settings, etc.
        """
        super().__init__(channel_config)
        
        # Configuration
        self.host = channel_config.get("host", "0.0.0.0")
        self.port = channel_config.get("port", 8000)
        self.cors_origins = channel_config.get("cors_origins", ["*"])
        self.webhook_url = channel_config.get("webhook_url")
        self.api_key = channel_config.get("api_key")
        
        # Security configuration
        self.security_config = SecurityConfig(channel_config.get("security", {}))
        self.jwt_auth = JWTAuth(self.security_config)
        self.rate_limiter = RateLimiter(self.security_config)
        self.webhook_verifier = WebhookVerifier(self.security_config.webhook_secret) if self.security_config.webhook_secret else None
        self.circuit_breaker = CircuitBreaker()
        
        # FastAPI app with security middleware
        self.app = FastAPI(
            title="Chatbot Web Channel",
            description="Secure web channel for universal chatbot framework",
            version="1.0.0"
        )
        
        # Add security middleware
        if self.security_config.enforce_https:
            self.app.add_middleware(HTTPSEnforcer, enforce_https=True)
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"]
        )
        
        # WebSocket connections
        self.connections: Dict[str, WebSocketConnection] = {}
        
        # Message queue for async processing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        self._setup_routes()
    
    def _get_channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.WEB
    
    async def initialize(self) -> bool:
        """Initialize the web channel with security components."""
        try:
            # Initialize rate limiter
            await self.rate_limiter.initialize()
            
            self.is_active = True
            secure_logger.info(f"Web channel initialized on {self.host}:{self.port}")
            return True
        except Exception as e:
            secure_logger.error(f"Failed to initialize web channel: {e}")
            return False
    
    def _setup_routes(self):
        """Setup FastAPI routes with security enhancements."""
        
        @self.app.middleware("http")
        async def add_correlation_id(request: Request, call_next):
            """Add correlation ID to each request."""
            correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
            CorrelationContext.set_id(correlation_id)
            
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            """Apply rate limiting."""
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            user_agent = request.headers.get("user-agent", "unknown")
            identifier = f"{client_ip}:{user_agent}"
            
            if not await self.rate_limiter.is_allowed(identifier, "minute"):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            if not await self.rate_limiter.is_allowed(identifier, "hour"):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Hourly rate limit exceeded"
                )
            
            return await call_next(request)
        
        @self.app.post("/chat", response_model=WebChatResponse)
        async def chat_endpoint(
            request: WebChatRequest,
            http_request: Request,
            auth_data: dict = Depends(self.jwt_auth)
        ):
            """Secure REST endpoint for chat messages."""
            try:
                # Log the request (PII will be masked)
                secure_logger.info(f"Chat request from user {request.user_id}")
                
                # Create channel message
                message = ChannelMessage.create(
                    content=request.message,
                    user_id=request.user_id,
                    channel_type=ChannelType.WEB,
                    channel_user_id=request.user_id,
                    session_id=request.session_id,
                    metadata={
                        **request.metadata,
                        "auth_user": auth_data.get("user_id"),
                        "client_ip": getattr(http_request.client, 'host', 'unknown') if http_request.client else 'unknown',
                        "user_agent": http_request.headers.get("user-agent")
                    }
                )
                
                # Validate message
                if not await self.validate_message(message):
                    raise HTTPException(status_code=400, detail="Invalid message")
                
                # Add to message queue
                await self.message_queue.put(message)
                
                # Create response (in real implementation, this would be from chatbot)
                response = ChannelResponse.create(
                    content="Message received and processing...",
                    channel_type=ChannelType.WEB
                )
                
                secure_logger.info(f"Chat response sent to user {request.user_id}")
                
                return WebChatResponse(
                    response=response.content,
                    response_id=response.response_id,
                    session_id=message.session_id,
                    timestamp=response.timestamp.isoformat(),
                    quick_replies=response.quick_replies,
                    suggested_actions=response.suggested_actions,
                    requires_human_handoff=response.requires_human_handoff,
                    confidence_score=response.confidence_score
                )
                
            except HTTPException:
                raise
            except Exception as e:
                secure_logger.error(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            """Secure WebSocket endpoint for real-time chat."""
            await websocket.accept()
            
            session_id = str(uuid.uuid4())
            correlation_id = CorrelationContext.generate_new()
            connection = WebSocketConnection(websocket, user_id, session_id)
            self.connections[session_id] = connection
            
            secure_logger.info(f"WebSocket connection established for user {user_id}")
            
            try:
                # Send welcome message
                await connection.send_message({
                    "type": "connection_established",
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                while True:
                    # Receive message from WebSocket
                    data = await connection.receive_message()
                    
                    # Sanitize input
                    message_content = InputSanitizer.sanitize_text(data.get("message", ""))
                    
                    # Create channel message
                    message = ChannelMessage.create(
                        content=message_content,
                        user_id=user_id,
                        channel_type=ChannelType.WEB,
                        channel_user_id=user_id,
                        session_id=session_id,
                        metadata=data.get("metadata", {})
                    )
                    
                    # Validate and queue message
                    if await self.validate_message(message):
                        await self.message_queue.put(message)
                        
                        # Send acknowledgment
                        await connection.send_message({
                            "type": "message_received",
                            "message_id": message.message_id,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
            except WebSocketDisconnect:
                secure_logger.info(f"WebSocket disconnected for user {user_id}")
            except Exception as e:
                secure_logger.error(f"Error in WebSocket connection: {e}")
            finally:
                # Clean up connection
                if session_id in self.connections:
                    del self.connections[session_id]
        
        @self.app.post("/webhook")
        async def webhook_endpoint(request: Request):
            """Secure webhook endpoint with signature verification."""
            if not self.webhook_verifier:
                raise HTTPException(status_code=501, detail="Webhook verification not configured")
            
            # Get raw body for signature verification
            body = await request.body()
            signature = request.headers.get(self.security_config.webhook_signature_header)
            
            if not signature:
                raise HTTPException(status_code=400, detail="Missing signature header")
            
            # Verify signature
            if not self.webhook_verifier.verify_signature(body, signature):
                secure_logger.warning("Webhook signature verification failed")
                raise HTTPException(status_code=401, detail="Invalid signature")
            
            try:
                data = json.loads(body)
                secure_logger.info("Webhook received and verified")
                
                # Process webhook data
                # This would be implemented based on specific webhook requirements
                
                return {"status": "success"}
            
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                secure_logger.error(f"Error processing webhook: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return await self.health_check()
        
        @self.app.get("/connections")
        async def get_connections(auth_data: dict = Depends(self.jwt_auth)):
            """Get active WebSocket connections (admin only)."""
            # Check admin permissions
            if "admin" not in auth_data.get("permissions", []):
                raise HTTPException(status_code=403, detail="Admin access required")
            
            return {
                "active_connections": len(self.connections),
                "connections": [
                    {
                        "session_id": conn.session_id,
                        "user_id": conn.user_id,
                        "connected_at": conn.connected_at.isoformat(),
                        "last_activity": conn.last_activity.isoformat()
                    }
                    for conn in self.connections.values()
                ]
            }
    
    @retry_with_backoff(max_retries=3, backoff_factor=1.0)
    async def send_message(self, response: ChannelResponse, recipient_id: str) -> bool:
        """
        Send a response message through the web channel with retry logic.
        
        Args:
            response: The response to send
            recipient_id: Session ID for WebSocket or user ID for webhook
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            # Format response for web channel
            formatted_response = self.format_response_for_channel(response)
            
            # Try WebSocket first
            if recipient_id in self.connections:
                connection = self.connections[recipient_id]
                await connection.send_message({
                    "type": "bot_response",
                    "content": formatted_response.content,
                    "response_id": formatted_response.response_id,
                    "timestamp": formatted_response.timestamp.isoformat(),
                    "quick_replies": formatted_response.quick_replies,
                    "suggested_actions": formatted_response.suggested_actions,
                    "requires_human_handoff": formatted_response.requires_human_handoff,
                    "confidence_score": formatted_response.confidence_score
                })
                self._increment_sent()
                return True
            
            # Try webhook if configured
            elif self.webhook_url:
                return await self.circuit_breaker.call(
                    self._send_webhook, formatted_response, recipient_id
                )
            
            else:
                secure_logger.warning(f"No delivery method found for recipient: {recipient_id}")
                return False
                
        except Exception as e:
            secure_logger.error(f"Error sending message via web channel: {e}")
            self._increment_errors()
            return False
    
    async def _send_webhook(self, response: ChannelResponse, recipient_id: str) -> bool:
        """Send response via webhook with enhanced security."""
        try:
            if not self.webhook_url:
                return False
                
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "recipient_id": recipient_id,
                "response": {
                    "content": response.content,
                    "response_id": response.response_id,
                    "timestamp": response.timestamp.isoformat(),
                    "quick_replies": response.quick_replies,
                    "suggested_actions": response.suggested_actions,
                    "requires_human_handoff": response.requires_human_handoff,
                    "confidence_score": response.confidence_score
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=True  # Enforce SSL
                ) as resp:
                    if resp.status == 200:
                        secure_logger.info(f"Webhook sent successfully to {recipient_id}")
                        self._increment_sent()
                        return True
                    else:
                        secure_logger.error(f"Webhook failed with status {resp.status}")
                        self._increment_errors()
                        return False
                        
        except Exception as e:
            secure_logger.error(f"Error sending webhook: {e}")
            self._increment_errors()
            return False
    
    async def receive_message(self) -> Optional[ChannelMessage]:
        """Receive a message from the message queue."""
        try:
            # Wait for message with timeout
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            self._increment_received()
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            secure_logger.error(f"Error receiving message: {e}")
            self._increment_errors()
            return None
    
    async def validate_message(self, message: ChannelMessage) -> bool:
        """Validate an incoming web message with enhanced security checks."""
        try:
            # Basic validation
            if not message.content or not message.content.strip():
                secure_logger.warning("Empty message content")
                return False
            
            if not message.user.user_id:
                secure_logger.warning("Missing user ID")
                return False
            
            if len(message.content) > 4000:
                secure_logger.warning("Message too long")
                return False
            
            # Additional security validation
            sanitized_content = InputSanitizer.sanitize_text(message.content)
            if sanitized_content != message.content:
                secure_logger.warning("Message content modified during sanitization")
                message.content = sanitized_content
            
            return True
            
        except Exception as e:
            secure_logger.error(f"Error validating message: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get web channel capabilities."""
        return [
            "text_messages",
            "real_time_websocket",
            "quick_replies",
            "suggested_actions",
            "file_attachments",
            "typing_indicator",
            "read_receipts",
            "rich_content",
            "webhook_delivery",
            "jwt_authentication",
            "rate_limiting",
            "input_sanitization",
            "webhook_verification"
        ]
    
    async def typing_indicator(self, recipient_id: str, is_typing: bool = True) -> bool:
        """Send typing indicator via WebSocket."""
        try:
            if recipient_id in self.connections:
                connection = self.connections[recipient_id]
                await connection.send_message({
                    "type": "typing_indicator",
                    "is_typing": is_typing,
                    "timestamp": datetime.utcnow().isoformat()
                })
                return True
            return False
        except Exception as e:
            secure_logger.error(f"Error sending typing indicator: {e}")
            return False
    
    async def close(self) -> bool:
        """Close all WebSocket connections and cleanup."""
        try:
            # Close all WebSocket connections
            for connection in list(self.connections.values()):
                try:
                    await connection.websocket.close()
                except:
                    pass
            
            self.connections.clear()
            
            # Clear message queue
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except:
                    break
            
            self.is_active = False
            secure_logger.info("Web channel closed successfully")
            return True
            
        except Exception as e:
            secure_logger.error(f"Error closing web channel: {e}")
            return False
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app
    
    def create_auth_token(self, user_id: str, permissions: Optional[List[str]] = None) -> str:
        """Create JWT authentication token for a user."""
        return self.jwt_auth.create_token(user_id, permissions or [])
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-related metrics."""
        return {
            "rate_limiter_active": self.rate_limiter.redis_client is not None,
            "webhook_verification_enabled": self.webhook_verifier is not None,
            "https_enforced": self.security_config.enforce_https,
            "cors_origins": self.cors_origins,
            "active_connections": len(self.connections),
            **self.get_metrics()
        }
    
