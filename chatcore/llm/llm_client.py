"""
LLM Client Module for Chatcore Framework
=======================================

Enterprise-grade LLM client abstraction supporting multiple providers
with security, performance, and extensibility features.

Key Features:
- Multi-provider support (OpenAI, Anthropic, LLaMA, Self-hosted)
- Security: Input sanitization, rate limiting, secure logging
- Performance: Async I/O, request batching, response caching
- Extensibility: Factory pattern, middleware hooks
- Reliability: Retry logic, structured error handling

Author: Chatbot Framework Team
Version: 1.0.0
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Tuple
import logging
import hashlib
import os
from urllib.parse import urlparse

# Third-party imports (to be installed)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"
    SELF_HOSTED = "self_hosted"
    AZURE_OPENAI = "azure_openai"


class ResponseFormat(Enum):
    """Response format types."""
    TEXT = "text"
    JSON = "json"
    STREAMING = "streaming"


class SecurityLevel(Enum):
    """Security level for content filtering."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRICT = "strict"


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl: int = 3600
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 40000
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    custom_headers: Dict[str, str] = field(default_factory=dict)
    proxy_url: Optional[str] = None


@dataclass
class Message:
    """Chat message structure."""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class LLMRequest:
    """LLM request structure."""
    messages: List[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    response_format: ResponseFormat = ResponseFormat.TEXT
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM response structure."""
    content: str
    finish_reason: str
    model: str
    usage: Dict[str, int]
    response_time: float
    provider: LLMProvider
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StreamingChunk:
    """Streaming response chunk."""
    content: str
    finish_reason: Optional[str] = None
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Base exception for LLM operations."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN", provider: Optional[LLMProvider] = None, **kwargs):
        super().__init__(message)
        self.error_code = error_code
        self.provider = provider
        self.metadata = kwargs


class RateLimitError(LLMError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="RATE_LIMIT", **kwargs)
        self.retry_after = retry_after


class SecurityError(LLMError):
    """Security-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SECURITY_VIOLATION", **kwargs)


class ModelError(LLMError):
    """Model-specific error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)


class ContentFilter:
    """Content filtering and sanitization."""
    
    # Patterns for prompt injection detection
    INJECTION_PATTERNS = [
        r'ignore\s+all\s+previous\s+instructions',
        r'forget\s+everything\s+above',
        r'system\s*:\s*you\s+are\s+now',
        r'new\s+instructions?:',
        r'roleplay\s+as\s+a\s+different',
        r'pretend\s+to\s+be\s+(?:an?\s+)?(?:evil|malicious|harmful)',
        r'execute\s+(?:python|javascript|sql|shell|bash)',
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'eval\s*\(',
        r'subprocess\.',
        r'os\.',
        r'__import__',
    ]
    
    # Sensitive data patterns
    SENSITIVE_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z\d]?){0,16}\b',  # IBAN
        r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*',  # Bearer token
        r'sk-[A-Za-z0-9]{48}',  # OpenAI API key pattern
    ]
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.injection_regex = re.compile('|'.join(self.INJECTION_PATTERNS), re.IGNORECASE)
        self.sensitive_regex = re.compile('|'.join(self.SENSITIVE_PATTERNS), re.IGNORECASE)
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input."""
        if not isinstance(text, str):
            raise SecurityError("Input must be a string")
        
        # Check for prompt injection
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.STRICT]:
            if self.injection_regex.search(text):
                raise SecurityError("Potential prompt injection detected")
        
        # Remove or mask sensitive data
        sanitized = self.sensitive_regex.sub('[REDACTED]', text)
        
        # Additional sanitization based on security level
        if self.security_level == SecurityLevel.STRICT:
            # Remove potential code blocks
            sanitized = re.sub(r'```.*?```', '[CODE_BLOCK_REMOVED]', sanitized, flags=re.DOTALL)
            sanitized = re.sub(r'`[^`]+`', '[CODE_REMOVED]', sanitized)
        
        return sanitized
    
    def sanitize_for_logging(self, text: str) -> str:
        """Sanitize text for logging purposes."""
        # Always remove sensitive data from logs
        sanitized = self.sensitive_regex.sub('[REDACTED]', text)
        
        # Truncate very long content
        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."
        
        return sanitized


class TokenCounter:
    """Token counting utilities."""
    
    def __init__(self):
        self.encoders = {}
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text for a specific model."""
        if not TIKTOKEN_AVAILABLE:
            # Fallback: rough estimation
            return int(len(text.split()) * 1.3)
        
        try:
            if model not in self.encoders:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            encoder = self.encoders[model]
            return len(encoder.encode(text))
        except Exception:
            # Fallback for unknown models
            return int(len(text.split()) * 1.3)
    
    def count_messages_tokens(self, messages: List[Message], model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in a list of messages."""
        total = 0
        for message in messages:
            total += self.count_tokens(f"{message.role}: {message.content}", model)
            total += 3  # Overhead per message
        total += 3  # Overhead for the conversation
        return total


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, rpm: int = 60, tpm: int = 40000):
        self.rpm = rpm
        self.tpm = tpm
        self.request_times = []
        self.token_usage = []
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, estimated_tokens: int = 0) -> bool:
        """Check if request is within rate limits."""
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            # Check request rate limit
            if len(self.request_times) >= self.rpm:
                return False
            
            # Check token rate limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tpm:
                return False
            
            # Record this request
            self.request_times.append(now)
            if estimated_tokens > 0:
                self.token_usage.append((now, estimated_tokens))
            
            return True
    
    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limit would be exceeded."""
        while not await self.check_rate_limit(estimated_tokens):
            await asyncio.sleep(1)


class ResponseCache:
    """LLM response caching."""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.redis_client = None
        
        # Try to initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for LLM caching")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}. Using in-memory cache.")
                self.redis_client = None
    
    def _generate_cache_key(self, request: LLMRequest, config: LLMConfig) -> str:
        """Generate cache key for request."""
        # Create a deterministic hash of the request
        key_data = {
            'provider': config.provider.value,
            'model': request.model or config.model_name,
            'messages': [{'role': m.role, 'content': m.content} for m in request.messages],
            'temperature': request.temperature or config.temperature,
            'max_tokens': request.max_tokens or config.max_tokens,
            'top_p': request.top_p or config.top_p,
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"llm_cache:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def get(self, request: LLMRequest, config: LLMConfig) -> Optional[LLMResponse]:
        """Get cached response."""
        if not config.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(request, config)
        
        try:
            if self.redis_client:
                # Use Redis
                cached_data = await asyncio.to_thread(self.redis_client.get, cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    response = LLMResponse(**data)
                    response.cached = True
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return response
            else:
                # Use in-memory cache
                if cache_key in self.cache:
                    cached_time, response = self.cache[cache_key]
                    if time.time() - cached_time < self.ttl:
                        response.cached = True
                        logger.debug(f"In-memory cache hit for key: {cache_key}")
                        return response
                    else:
                        del self.cache[cache_key]
        
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def set(self, request: LLMRequest, config: LLMConfig, response: LLMResponse) -> None:
        """Cache response."""
        if not config.enable_caching or response.cached:
            return
        
        cache_key = self._generate_cache_key(request, config)
        
        try:
            response_data = {
                'content': response.content,
                'finish_reason': response.finish_reason,
                'model': response.model,
                'usage': response.usage,
                'response_time': response.response_time,
                'provider': response.provider.value,
                'metadata': response.metadata,
                'timestamp': response.timestamp.isoformat(),
            }
            
            if self.redis_client:
                # Use Redis
                await asyncio.to_thread(
                    self.redis_client.setex,
                    cache_key,
                    self.ttl,
                    json.dumps(response_data, default=str)
                )
            else:
                # Use in-memory cache
                if len(self.cache) >= self.max_size:
                    # Remove oldest entry
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                    del self.cache[oldest_key]
                
                self.cache[cache_key] = (time.time(), response)
            
            logger.debug(f"Cached response for key: {cache_key}")
        
        except Exception as e:
            logger.warning(f"Cache set error: {e}")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.content_filter = ContentFilter(config.security_level)
        self.token_counter = TokenCounter()
        self.rate_limiter = RateLimiter(config.rate_limit_rpm, config.rate_limit_tpm)
        self.cache = ResponseCache(config.cache_ttl)
        
        # Middleware hooks
        self.pre_process_hooks: List[Callable] = []
        self.post_process_hooks: List[Callable] = []
    
    def add_pre_process_hook(self, hook: Callable) -> None:
        """Add pre-processing hook."""
        self.pre_process_hooks.append(hook)
    
    def add_post_process_hook(self, hook: Callable) -> None:
        """Add post-processing hook."""
        self.post_process_hooks.append(hook)
    
    async def _run_pre_process_hooks(self, request: LLMRequest) -> LLMRequest:
        """Run pre-processing hooks."""
        for hook in self.pre_process_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    request = await hook(request)
                else:
                    request = hook(request)
            except Exception as e:
                logger.warning(f"Pre-process hook error: {e}")
        return request
    
    async def _run_post_process_hooks(self, response: LLMResponse) -> LLMResponse:
        """Run post-processing hooks."""
        for hook in self.post_process_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    response = await hook(response)
                else:
                    response = hook(response)
            except Exception as e:
                logger.warning(f"Post-process hook error: {e}")
        return response
    
    def _sanitize_request(self, request: LLMRequest) -> LLMRequest:
        """Sanitize request content."""
        sanitized_messages = []
        for message in request.messages:
            sanitized_content = self.content_filter.sanitize_input(message.content)
            sanitized_message = Message(
                role=message.role,
                content=sanitized_content,
                metadata=message.metadata,
                timestamp=message.timestamp
            )
            sanitized_messages.append(sanitized_message)
        
        request.messages = sanitized_messages
        return request
    
    def _validate_request(self, request: LLMRequest) -> None:
        """Validate request parameters."""
        if not request.messages:
            raise ModelError("Messages cannot be empty")
        
        # Check token limits
        estimated_tokens = self.token_counter.count_messages_tokens(
            request.messages, 
            request.model or self.config.model_name
        )
        
        max_tokens = request.max_tokens or self.config.max_tokens
        if estimated_tokens > max_tokens:
            raise ModelError(f"Request tokens ({estimated_tokens}) exceed limit ({max_tokens})")
        
        # Validate message roles
        valid_roles = {"system", "user", "assistant"}
        for message in request.messages:
            if message.role not in valid_roles:
                raise ModelError(f"Invalid message role: {message.role}")
    
    async def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Retry function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed")
                    break
        
        if last_exception:
            raise last_exception
        else:
            raise ModelError("Request failed with unknown error")
    
    @abstractmethod
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make the actual API request (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def _make_streaming_request(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Make streaming API request (to be implemented by subclasses)."""
        pass
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat request."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = await self.cache.get(request, self.config)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response
            
            # Run pre-processing hooks
            request = await self._run_pre_process_hooks(request)
            
            # Sanitize and validate
            request = self._sanitize_request(request)
            self._validate_request(request)
            
            # Check rate limits
            estimated_tokens = self.token_counter.count_messages_tokens(
                request.messages, 
                request.model or self.config.model_name
            )
            await self.rate_limiter.wait_if_needed(estimated_tokens)
            
            # Make request with retry
            response = await self._retry_with_backoff(self._make_request, request)
            
            # Cache response
            await self.cache.set(request, self.config, response)
            
            # Run post-processing hooks
            response = await self._run_post_process_hooks(response)
            
            # Log success (sanitized)
            sanitized_content = self.content_filter.sanitize_for_logging(response.content)
            logger.info(f"LLM request completed in {response.response_time:.2f}s, "
                       f"content: {sanitized_content[:100]}...")
            
            return response
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"LLM request failed after {total_time:.2f}s: {e}")
            if isinstance(e, LLMError):
                raise
            else:
                raise ModelError(f"Unexpected error: {e}", provider=self.config.provider)
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Stream a chat request."""
        try:
            # Run pre-processing hooks
            request = await self._run_pre_process_hooks(request)
            
            # Sanitize and validate
            request = self._sanitize_request(request)
            self._validate_request(request)
            
            # Check rate limits
            estimated_tokens = self.token_counter.count_messages_tokens(
                request.messages, 
                request.model or self.config.model_name
            )
            await self.rate_limiter.wait_if_needed(estimated_tokens)
            
            # Stream response
            async for chunk in self._make_streaming_request(request):
                yield chunk
                
        except Exception as e:
            logger.error(f"LLM streaming request failed: {e}")
            if isinstance(e, LLMError):
                raise
            else:
                raise ModelError(f"Unexpected streaming error: {e}", provider=self.config.provider)
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return self.config.provider
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "api_base": self.config.api_base
        }


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ModelError("OpenAI package not installed. Run: pip install openai")
        
        # Initialize OpenAI client
        api_key = config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ModelError("OpenAI API key not provided")
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            organization=config.organization,
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make OpenAI API request."""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "frequency_penalty": request.frequency_penalty or self.config.frequency_penalty,
                "presence_penalty": request.presence_penalty or self.config.presence_penalty,
                "stream": False,
                "user": request.user_id,
            }
            
            if request.stop:
                params["stop"] = request.stop
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            choice = response.choices[0]
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                model=response.model,
                usage=usage,
                response_time=time.time() - start_time,
                provider=LLMProvider.OPENAI,
                metadata={"request_id": getattr(response, 'id', None)}
            )
            
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}", retry_after=retry_after)
        
        except openai.AuthenticationError as e:
            raise SecurityError(f"OpenAI authentication failed: {e}")
        
        except openai.BadRequestError as e:
            raise ModelError(f"OpenAI bad request: {e}")
        
        except Exception as e:
            raise ModelError(f"OpenAI request failed: {e}")
    
    def _make_streaming_request(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Make streaming OpenAI API request."""
        return self._stream_openai(request)
    
    async def _stream_openai(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Internal streaming implementation."""
        try:
            # Prepare messages
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "frequency_penalty": request.frequency_penalty or self.config.frequency_penalty,
                "presence_penalty": request.presence_penalty or self.config.presence_penalty,
                "stream": True,
                "user": request.user_id,
            }
            
            if request.stop:
                params["stop"] = request.stop
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamingChunk(
                        content=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                        index=chunk.choices[0].index,
                        metadata={"chunk_id": chunk.id}
                    )
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise ModelError(f"OpenAI streaming failed: {e}")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not ANTHROPIC_AVAILABLE:
            raise ModelError("Anthropic package not installed. Run: pip install anthropic")
        
        # Initialize Anthropic client
        api_key = config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ModelError("Anthropic API key not provided")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
    
    def _convert_messages(self, messages: List[Message]) -> Tuple[str, List[Dict]]:
        """Convert messages to Anthropic format."""
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, anthropic_messages
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make Anthropic API request."""
        start_time = time.time()
        
        try:
            # Convert messages
            system_message, messages = self._convert_messages(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "stream": False,
            }
            
            if system_message:
                params["system"] = system_message
            
            if request.stop:
                params["stop_sequences"] = request.stop
            
            # Make API call
            response = await self.client.messages.create(**params)
            
            # Extract response data
            content = ""
            if response.content:
                content = " ".join([block.text for block in response.content if hasattr(block, 'text')])
            
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            
            return LLMResponse(
                content=content,
                finish_reason=response.stop_reason or "stop",
                model=response.model,
                usage=usage,
                response_time=time.time() - start_time,
                provider=LLMProvider.ANTHROPIC,
                metadata={"request_id": response.id}
            )
            
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        
        except anthropic.AuthenticationError as e:
            raise SecurityError(f"Anthropic authentication failed: {e}")
        
        except anthropic.BadRequestError as e:
            raise ModelError(f"Anthropic bad request: {e}")
        
        except Exception as e:
            raise ModelError(f"Anthropic request failed: {e}")
    
    def _make_streaming_request(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Make streaming Anthropic API request."""
        return self._stream_anthropic(request)
    
    async def _stream_anthropic(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Internal streaming implementation."""
        try:
            # Convert messages
            system_message, messages = self._convert_messages(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model or self.config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "stream": True,
            }
            
            if system_message:
                params["system"] = system_message
            
            if request.stop:
                params["stop_sequences"] = request.stop
            
            # Make streaming API call
            async with self.client.messages.stream(**params) as stream:
                async for chunk in stream:
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        yield StreamingChunk(
                            content=chunk.delta.text,
                            finish_reason=None,
                            index=0,
                            metadata={"chunk_type": chunk.type}
                        )
                    elif hasattr(chunk, 'type') and chunk.type == 'message_stop':
                        yield StreamingChunk(
                            content="",
                            finish_reason="stop",
                            index=0,
                            metadata={"chunk_type": chunk.type}
                        )
                        
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise ModelError(f"Anthropic streaming failed: {e}")


class SelfHostedClient(BaseLLMClient):
    """Self-hosted model client implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not config.api_base:
            raise ModelError("API base URL required for self-hosted models")
        
        # Parse URL to validate
        parsed = urlparse(config.api_base)
        if not parsed.scheme or not parsed.netloc:
            raise ModelError("Invalid API base URL")
        
        self.api_base = config.api_base.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            **config.custom_headers
        }
        
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"
    
    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make self-hosted API request."""
        import aiohttp
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Prepare request payload
            payload = {
                "model": request.model or self.config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "stream": False,
            }
            
            if request.stop:
                payload["stop"] = request.stop
            
            # Make API call
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_base}/v1/chat/completions",
                    json=payload,
                    headers=self.headers,
                    proxy=self.config.proxy_url
                ) as response:
                    
                    if response.status == 429:
                        retry_after = response.headers.get('Retry-After')
                        raise RateLimitError(
                            f"Rate limit exceeded",
                            retry_after=int(retry_after) if retry_after else None
                        )
                    
                    if response.status == 401:
                        raise SecurityError("Authentication failed")
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        raise ModelError(f"API error {response.status}: {error_text}")
                    
                    data = await response.json()
            
            # Extract response data
            choice = data['choices'][0]
            usage = data.get('usage', {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
            
            return LLMResponse(
                content=choice['message']['content'] or "",
                finish_reason=choice.get('finish_reason', 'stop'),
                model=data.get('model', self.config.model_name),
                usage=usage,
                response_time=time.time() - start_time,
                provider=LLMProvider.SELF_HOSTED,
                metadata={"api_base": self.api_base}
            )
            
        except aiohttp.ClientError as e:
            raise ModelError(f"Self-hosted API connection failed: {e}")
        
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise ModelError(f"Self-hosted API request failed: {e}")
    
    def _make_streaming_request(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Make streaming self-hosted API request."""
        return self._stream_self_hosted(request)
    
    async def _stream_self_hosted(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Internal streaming implementation."""
        import aiohttp
        
        try:
            # Prepare messages
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Prepare request payload
            payload = {
                "model": request.model or self.config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "stream": True,
            }
            
            if request.stop:
                payload["stop"] = request.stop
            
            # Make streaming API call
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.api_base}/v1/chat/completions",
                    json=payload,
                    headers=self.headers,
                    proxy=self.config.proxy_url
                ) as response:
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        raise ModelError(f"API error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choice = data['choices'][0]
                                delta = choice.get('delta', {})
                                
                                if 'content' in delta and delta['content']:
                                    yield StreamingChunk(
                                        content=delta['content'],
                                        finish_reason=choice.get('finish_reason'),
                                        index=choice.get('index', 0),
                                        metadata={"api_base": self.api_base}
                                    )
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Self-hosted streaming error: {e}")
            raise ModelError(f"Self-hosted streaming failed: {e}")
