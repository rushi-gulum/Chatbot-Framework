"""
LLM Module for Chatcore Framework
=================================

Enterprise-grade Large Language Model (LLM) integration module providing
secure, scalable, and modular LLM services for chatbot applications.

Key Features:
- Multi-provider support (OpenAI, Anthropic, LLaMA, Self-hosted)
- Security: Input sanitization, rate limiting, secure logging
- Performance: Async I/O, request batching, response caching
- Extensibility: Factory pattern, middleware hooks
- Reliability: Retry logic, circuit breakers, health monitoring

Quick Start:
    from chatcore.llm import InferenceRouter, LLMConfig, LLMProvider
    
    # Configure providers
    openai_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    
    # Create router
    router = InferenceRouter()
    router.add_provider("openai-main", openai_config)
    
    # Make requests
    request = LLMRequest(messages=[Message(role="user", content="Hello!")])
    response = await router.complete(request)

Author: Chatbot Framework Team
Version: 1.0.0
"""

from .llm_client import (
    # Core classes
    BaseLLMClient,
    OpenAIClient,
    AnthropicClient,
    SelfHostedClient,
    
    # Configuration
    LLMConfig,
    LLMRequest,
    LLMResponse,
    Message,
    StreamingChunk,
    
    # Enums
    LLMProvider,
    ResponseFormat,
    SecurityLevel,
    
    # Exceptions
    LLMError,
    RateLimitError,
    SecurityError,
    ModelError,
    
    # Utilities
    ContentFilter,
    TokenCounter,
    RateLimiter,
    ResponseCache,
)

from .inference_router import (
    # Main router
    InferenceRouter,
    
    # Factory
    LLMClientFactory,
    
    # Configuration
    RoutingStrategy,
    RoutingRule,
    ProviderStatus,
    ProviderHealth,
    
    # Middleware
    MiddlewarePipeline,
    request_logging_middleware,
    response_logging_middleware,
    security_middleware,
    cost_tracking_middleware,
    
    # Monitoring
    InferenceMetrics,
    CircuitBreaker,
)

__version__ = "1.0.0"
__author__ = "Chatbot Framework Team"

# Module-level exports
__all__ = [
    # Core client classes
    "BaseLLMClient",
    "OpenAIClient", 
    "AnthropicClient",
    "SelfHostedClient",
    
    # Router and factory
    "InferenceRouter",
    "LLMClientFactory",
    
    # Configuration classes
    "LLMConfig",
    "LLMRequest", 
    "LLMResponse",
    "Message",
    "StreamingChunk",
    "RoutingRule",
    
    # Enums
    "LLMProvider",
    "ResponseFormat", 
    "SecurityLevel",
    "RoutingStrategy",
    "ProviderStatus",
    
    # Exceptions
    "LLMError",
    "RateLimitError",
    "SecurityError", 
    "ModelError",
    
    # Utilities
    "ContentFilter",
    "TokenCounter",
    "RateLimiter", 
    "ResponseCache",
    "MiddlewarePipeline",
    "CircuitBreaker",
    
    # Built-in middleware
    "request_logging_middleware",
    "response_logging_middleware",
    "security_middleware", 
    "cost_tracking_middleware",
    
    # Data classes
    "ProviderHealth",
    "InferenceMetrics",
]

# Package metadata
OPENAI_AVAILABLE = True
ANTHROPIC_AVAILABLE = True
REDIS_AVAILABLE = True

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
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Convenience functions
def create_openai_client(api_key: str, model_name: str = "gpt-3.5-turbo", **kwargs) -> OpenAIClient:
    """Convenience function to create OpenAI client."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    return OpenAIClient(config)


def create_anthropic_client(api_key: str, model_name: str = "claude-3-sonnet-20240229", **kwargs) -> AnthropicClient:
    """Convenience function to create Anthropic client."""
    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    return AnthropicClient(config)


def create_self_hosted_client(api_base: str, model_name: str, api_key: str = None, **kwargs) -> SelfHostedClient:
    """Convenience function to create self-hosted client."""
    config = LLMConfig(
        provider=LLMProvider.SELF_HOSTED,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        **kwargs
    )
    return SelfHostedClient(config)


def create_basic_router(providers: list = None) -> InferenceRouter:
    """Create a basic inference router with default configuration."""
    router = InferenceRouter()
    
    # Add built-in middleware
    router.middleware.add_pre_request_middleware(request_logging_middleware)
    router.middleware.add_post_response_middleware(response_logging_middleware)
    router.middleware.add_pre_request_middleware(security_middleware)
    router.middleware.add_post_response_middleware(cost_tracking_middleware)
    
    # Add providers if specified
    if providers:
        for i, provider_config in enumerate(providers):
            if isinstance(provider_config, dict):
                config = LLMConfig(**provider_config)
                router.add_provider(f"provider_{i}", config)
            elif isinstance(provider_config, LLMConfig):
                router.add_provider(f"provider_{i}", provider_config)
    
    return router
