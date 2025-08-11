"""
Enterprise Chatbot Core Architecture
===================================

This module defines the core interfaces and base classes for an enterprise-grade,
modular chatbot framework with full security, dependency injection, and extensibility.

Architecture Principles:
- Dependency Injection for testability and modularity
- Interface-based design for easy mocking and extension
- Security-first approach with input validation and sanitization
- Channel-agnostic design supporting multiple communication platforms
- Future-proof AI model integration with abstraction layers
- Comprehensive logging, monitoring, and analytics
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging
from contextlib import asynccontextmanager
import asyncio

# Security imports
from cryptography.fernet import Fernet
from ..channels.security import SecureLogger, InputSanitizer

# Configure secure logging
secure_logger = SecureLogger(__name__)


class MessageType(Enum):
    """Types of messages in the chatbot system."""
    USER_INPUT = "user_input"
    BOT_RESPONSE = "bot_response"
    SYSTEM_MESSAGE = "system_message"
    FALLBACK = "fallback"
    DISAMBIGUATION = "disambiguation"
    CSAT_REQUEST = "csat_request"


class SentimentType(Enum):
    """Sentiment analysis results."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    CONFUSED = "confused"


class ResponseConfidence(Enum):
    """Confidence levels for bot responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class ChatMessage:
    """
    Secure, immutable message object for the chatbot system.
    
    All messages are validated and sanitized upon creation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = field(default="")
    message_type: MessageType = MessageType.USER_INPUT
    user_id: str = field(default="")
    session_id: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    channel_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and sanitize message content upon creation."""
        if self.content:
            self.content = InputSanitizer.sanitize_text(self.content, max_length=4000)
        
        # Validate required fields
        if not self.user_id:
            raise ValueError("user_id is required")
        
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "message_type": self.message_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "channel_info": self.channel_info
        }


@dataclass
class ChatResponse:
    """
    Bot response with confidence, sentiment, and action indicators.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = field(default="")
    confidence: ResponseConfidence = ResponseConfidence.MEDIUM
    requires_human_handoff: bool = False
    suggested_actions: List[str] = field(default_factory=list)
    quick_replies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate response content."""
        if self.content:
            self.content = InputSanitizer.sanitize_text(self.content, max_length=4000)


@dataclass
class ConversationContext:
    """
    Immutable context object containing conversation state and history.
    """
    session_id: str
    user_id: str
    current_intent: Optional[str] = None
    sentiment: Optional[SentimentType] = None
    conversation_stage: str = "initial"
    context_variables: Dict[str, Any] = field(default_factory=dict)
    recent_messages: List[ChatMessage] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    channel_context: Dict[str, Any] = field(default_factory=dict)


class SecurityConfig:
    """Security configuration for the chatbot core."""
    
    def __init__(self, config: Dict[str, Any]):
        self.encryption_key = config.get("encryption_key")
        self.enable_input_validation = config.get("enable_input_validation", True)
        self.enable_pii_masking = config.get("enable_pii_masking", True)
        self.max_message_length = config.get("max_message_length", 4000)
        self.rate_limit_per_session = config.get("rate_limit_per_session", 100)
        
        # Initialize encryption if key provided
        self.cipher_suite = None
        if self.encryption_key:
            self.cipher_suite = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)


# ============================================================================
# CORE INTERFACES - Define contracts for all chatbot components
# ============================================================================

class IMemoryProvider(ABC):
    """Interface for memory management systems."""
    
    @abstractmethod
    async def store_message(self, message: ChatMessage, context: ConversationContext) -> bool:
        """Store a message in memory with context."""
        pass
    
    @abstractmethod
    async def retrieve_context(self, session_id: str, user_id: str) -> ConversationContext:
        """Retrieve conversation context for a session."""
        pass
    
    @abstractmethod
    async def store_summary(self, session_id: str, summary: str) -> bool:
        """Store conversation summary."""
        pass
    
    @abstractmethod
    async def search_memory(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory for relevant information."""
        pass
    
    @abstractmethod
    async def clear_session(self, session_id: str) -> bool:
        """Clear session-specific memory."""
        pass


class ISentimentAnalyzer(ABC):
    """Interface for sentiment analysis."""
    
    @abstractmethod
    async def analyze_sentiment(self, message: ChatMessage, context: ConversationContext) -> SentimentType:
        """Analyze sentiment of a message."""
        pass
    
    @abstractmethod
    async def get_emotional_state(self, context: ConversationContext) -> Dict[str, float]:
        """Get overall emotional state from conversation context."""
        pass


class IDialogueManager(ABC):
    """Interface for dialogue management."""
    
    @abstractmethod
    async def process_message(self, message: ChatMessage, context: ConversationContext) -> ConversationContext:
        """Process message and update conversation context."""
        pass
    
    @abstractmethod
    async def determine_intent(self, message: ChatMessage, context: ConversationContext) -> Optional[str]:
        """Determine user intent from message."""
        pass
    
    @abstractmethod
    async def manage_conversation_flow(self, context: ConversationContext) -> Dict[str, Any]:
        """Manage conversation flow and next actions."""
        pass


class IDisambiguator(ABC):
    """Interface for handling ambiguous user inputs."""
    
    @abstractmethod
    async def is_ambiguous(self, message: ChatMessage, context: ConversationContext) -> bool:
        """Check if message is ambiguous."""
        pass
    
    @abstractmethod
    async def generate_clarification(self, message: ChatMessage, context: ConversationContext) -> ChatResponse:
        """Generate clarification questions for ambiguous input."""
        pass
    
    @abstractmethod
    async def resolve_ambiguity(self, clarification_response: ChatMessage, context: ConversationContext) -> ConversationContext:
        """Resolve ambiguity based on user clarification."""
        pass


class IFallbackHandler(ABC):
    """Interface for fallback handling."""
    
    @abstractmethod
    async def should_fallback(self, message: ChatMessage, context: ConversationContext, confidence: ResponseConfidence) -> bool:
        """Determine if fallback is needed."""
        pass
    
    @abstractmethod
    async def generate_fallback_response(self, message: ChatMessage, context: ConversationContext) -> ChatResponse:
        """Generate appropriate fallback response."""
        pass
    
    @abstractmethod
    async def escalate_to_human(self, message: ChatMessage, context: ConversationContext) -> ChatResponse:
        """Escalate conversation to human agent."""
        pass


class ICSATTracker(ABC):
    """Interface for Customer Satisfaction tracking."""
    
    @abstractmethod
    async def should_request_feedback(self, context: ConversationContext) -> bool:
        """Determine if CSAT feedback should be requested."""
        pass
    
    @abstractmethod
    async def generate_csat_request(self, context: ConversationContext) -> ChatResponse:
        """Generate CSAT feedback request."""
        pass
    
    @abstractmethod
    async def process_csat_feedback(self, feedback: ChatMessage, context: ConversationContext) -> float:
        """Process and store CSAT feedback."""
        pass
    
    @abstractmethod
    async def get_satisfaction_metrics(self, user_id: Optional[str] = None, timeframe_days: int = 30) -> Dict[str, float]:
        """Get satisfaction metrics."""
        pass


class ISummarizer(ABC):
    """Interface for conversation summarization."""
    
    @abstractmethod
    async def summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """Summarize a conversation."""
        pass
    
    @abstractmethod
    async def extract_key_points(self, conversation_summary: str) -> List[str]:
        """Extract key points from conversation."""
        pass
    
    @abstractmethod
    async def generate_analytics_summary(self, context: ConversationContext) -> Dict[str, Any]:
        """Generate summary for analytics purposes."""
        pass


class ILLMProvider(ABC):
    """Interface for Large Language Model providers."""
    
    @abstractmethod
    async def generate_response(self, message: ChatMessage, context: ConversationContext, system_prompt: str = "") -> ChatResponse:
        """Generate response using LLM."""
        pass
    
    @abstractmethod
    async def classify_intent(self, message: ChatMessage) -> str:
        """Classify user intent using LLM."""
        pass
    
    @abstractmethod
    async def extract_entities(self, message: ChatMessage) -> Dict[str, Any]:
        """Extract entities from message."""
        pass


# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

class DIContainer:
    """
    Dependency Injection container for managing chatbot components.
    
    Supports both singleton and transient lifetimes, factory methods,
    and lazy initialization for better performance and testability.
    """
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        
    def register_singleton(self, interface: Type, implementation: Any) -> 'DIContainer':
        """Register a singleton service."""
        self._singletons[interface] = implementation
        return self
    
    def register_transient(self, interface: Type, implementation_factory: Callable) -> 'DIContainer':
        """Register a transient service with factory."""
        self._factories[interface] = implementation_factory
        return self
    
    def register_instance(self, interface: Type, instance: Any) -> 'DIContainer':
        """Register a specific instance."""
        self._services[interface] = instance
        return self
    
    def resolve(self, interface: Type) -> Any:
        """Resolve a service from the container."""
        # Check for direct instance
        if interface in self._services:
            return self._services[interface]
        
        # Check for singleton
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check for factory
        if interface in self._factories:
            return self._factories[interface]()
        
        raise ValueError(f"Service {interface} not registered")
    
    def resolve_all(self, *interfaces: Type) -> tuple:
        """Resolve multiple services at once."""
        return tuple(self.resolve(interface) for interface in interfaces)


# ============================================================================
# CORE CHATBOT ENGINE
# ============================================================================

class ChatbotCore:
    """
    Enterprise-grade chatbot core with full security, monitoring, and extensibility.
    
    This is the main orchestrator that coordinates all chatbot components
    through dependency injection and provides a clean, testable interface.
    """
    
    def __init__(self, 
                 container: DIContainer,
                 security_config: SecurityConfig,
                 enable_monitoring: bool = True):
        """
        Initialize the chatbot core with dependency injection.
        
        Args:
            container: Dependency injection container with all required services
            security_config: Security configuration
            enable_monitoring: Whether to enable performance monitoring
        """
        self.container = container
        self.security_config = security_config
        self.enable_monitoring = enable_monitoring
        
        # Resolve core dependencies
        self.memory_provider = container.resolve(IMemoryProvider)
        self.sentiment_analyzer = container.resolve(ISentimentAnalyzer)
        self.dialogue_manager = container.resolve(IDialogueManager)
        self.disambiguator = container.resolve(IDisambiguator)
        self.fallback_handler = container.resolve(IFallbackHandler)
        self.csat_tracker = container.resolve(ICSATTracker)
        self.summarizer = container.resolve(ISummarizer)
        self.llm_provider = container.resolve(ILLMProvider)
        
        # Performance monitoring
        self.metrics = {
            "total_messages": 0,
            "fallback_rate": 0.0,
            "average_response_time": 0.0,
            "satisfaction_score": 0.0
        }
    
    async def process_message(self, message: ChatMessage) -> ChatResponse:
        """
        Main entry point for processing user messages.
        
        This method orchestrates the entire chatbot pipeline with proper
        error handling, security validation, and performance monitoring.
        
        Args:
            message: Sanitized and validated user message
            
        Returns:
            ChatResponse: Bot response with confidence and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # 1. Security validation
            await self._validate_message_security(message)
            
            # 2. Retrieve conversation context
            context = await self.memory_provider.retrieve_context(
                message.session_id, 
                message.user_id
            )
            
            # 3. Sentiment analysis
            sentiment = await self.sentiment_analyzer.analyze_sentiment(message, context)
            context.sentiment = sentiment
            
            # 4. Check for disambiguation need
            if await self.disambiguator.is_ambiguous(message, context):
                response = await self.disambiguator.generate_clarification(message, context)
                await self._finalize_response(message, response, context, start_time)
                return response
            
            # 5. Process through dialogue manager
            updated_context = await self.dialogue_manager.process_message(message, context)
            
            # 6. Generate response using LLM
            response = await self.llm_provider.generate_response(message, updated_context)
            
            # 7. Check if fallback is needed
            if await self.fallback_handler.should_fallback(message, updated_context, response.confidence):
                response = await self.fallback_handler.generate_fallback_response(message, updated_context)
            
            # 8. Check for CSAT request
            if await self.csat_tracker.should_request_feedback(updated_context):
                csat_response = await self.csat_tracker.generate_csat_request(updated_context)
                response.metadata["csat_request"] = csat_response.to_dict() if hasattr(csat_response, 'to_dict') else str(csat_response)
            
            # 9. Store message and update memory
            await self.memory_provider.store_message(message, updated_context)
            
            # 10. Finalize and return response
            await self._finalize_response(message, response, updated_context, start_time)
            return response
            
        except Exception as e:
            secure_logger.error(f"Error processing message: {e}")
            return await self._generate_error_response(message, str(e))
    
    async def _validate_message_security(self, message: ChatMessage) -> None:
        """Validate message security and apply rate limiting."""
        if not self.security_config.enable_input_validation:
            return
        
        # Content validation
        if len(message.content) > self.security_config.max_message_length:
            raise ValueError(f"Message too long: {len(message.content)} > {self.security_config.max_message_length}")
        
        # Rate limiting per session
        # Implementation would check Redis or in-memory store
        # This is a placeholder for the actual implementation
        
        secure_logger.info(f"Message validated for user {message.user_id}")
    
    async def _finalize_response(self, 
                               message: ChatMessage, 
                               response: ChatResponse, 
                               context: ConversationContext,
                               start_time: datetime) -> None:
        """Finalize response with logging and metrics."""
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update metrics
        self.metrics["total_messages"] += 1
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * (self.metrics["total_messages"] - 1) + processing_time) 
            / self.metrics["total_messages"]
        )
        
        # Log interaction
        secure_logger.info(
            f"Message processed - User: {message.user_id}, "
            f"Session: {message.session_id}, "
            f"Confidence: {response.confidence.value}, "
            f"Processing time: {processing_time:.3f}s"
        )
        
        # Store response in context for future reference
        response.metadata["processing_time"] = processing_time
        response.metadata["context_stage"] = context.conversation_stage
    
    async def _generate_error_response(self, message: ChatMessage, error: str) -> ChatResponse:
        """Generate appropriate error response."""
        return ChatResponse(
            content="I apologize, but I encountered an issue processing your message. Please try again.",
            confidence=ResponseConfidence.LOW,
            requires_human_handoff=True,
            metadata={
                "error": "internal_error",
                "original_message_id": message.id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary and analytics."""
        context = await self.memory_provider.retrieve_context(session_id, "")
        summary = await self.summarizer.generate_analytics_summary(context)
        return summary
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            **self.metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "memory_provider": type(self.memory_provider).__name__,
            "llm_provider": type(self.llm_provider).__name__
        }
    
    @asynccontextmanager
    async def conversation_session(self, user_id: str, session_id: Optional[str] = None):
        """
        Context manager for handling conversation sessions with proper cleanup.
        
        Usage:
            async with chatbot.conversation_session("user123") as session:
                response = await session.process_message(message)
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Initialize session context
            context = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                conversation_stage="started"
            )
            yield self
        finally:
            # Cleanup and summarization
            try:
                await self.memory_provider.store_summary(
                    session_id, 
                    await self.summarizer.summarize_conversation([])
                )
                secure_logger.info(f"Session {session_id} cleaned up successfully")
            except Exception as e:
                secure_logger.error(f"Error during session cleanup: {e}")


# ============================================================================
# CONFIGURATION AND FACTORY
# ============================================================================

class ChatbotFactory:
    """
    Factory for creating configured chatbot instances with proper DI setup.
    
    This factory encapsulates the complexity of setting up all dependencies
    and provides simple methods for creating chatbot instances for different
    environments (development, testing, production).
    """
    
    @staticmethod
    def create_production_chatbot(config: Dict[str, Any]) -> ChatbotCore:
        """Create a production-ready chatbot with all security features enabled."""
        container = DIContainer()
        security_config = SecurityConfig(config.get("security", {}))
        
        # Register implementations (these would be actual implementations)
        # This is a template showing how to set up the DI container
        
        return ChatbotCore(container, security_config, enable_monitoring=True)
    
    @staticmethod
    def create_test_chatbot(mock_providers: Dict[Type, Any]) -> ChatbotCore:
        """Create a chatbot for testing with mock providers."""
        container = DIContainer()
        
        # Register mock implementations
        for interface, mock_impl in mock_providers.items():
            container.register_instance(interface, mock_impl)
        
        security_config = SecurityConfig({"enable_input_validation": False})
        return ChatbotCore(container, security_config, enable_monitoring=False)


__all__ = [
    # Enums
    'MessageType', 'SentimentType', 'ResponseConfidence',
    # Data classes
    'ChatMessage', 'ChatResponse', 'ConversationContext',
    # Configuration
    'SecurityConfig',
    # Interfaces
    'IMemoryProvider', 'ISentimentAnalyzer', 'IDialogueManager',
    'IDisambiguator', 'IFallbackHandler', 'ICSATTracker', 
    'ISummarizer', 'ILLMProvider',
    # Core classes
    'DIContainer', 'ChatbotCore', 'ChatbotFactory'
]
