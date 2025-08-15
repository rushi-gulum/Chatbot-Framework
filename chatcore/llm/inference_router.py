"""
LLM Inference Router for Chatcore Framework
==========================================

Production-ready LLM inference routing system with factory pattern,
load balancing, failover, and intelligent request routing.

Key Features:
- Multi-provider factory pattern
- Load balancing and failover
- Request routing based on model capabilities
- Health monitoring and circuit breaker
- Middleware pipeline for pre/post processing
- Comprehensive monitoring and analytics

Author: Chatbot Framework Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Set
import json
from abc import ABC, abstractmethod

from .llm_client import (
    BaseLLMClient, OpenAIClient, AnthropicClient, SelfHostedClient,
    LLMConfig, LLMRequest, LLMResponse, StreamingChunk, Message,
    LLMProvider, LLMError, RateLimitError, ModelError, SecurityError,
    ResponseFormat, SecurityLevel
)

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Request routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    COST_OPTIMIZED = "cost_optimized"
    SPECIFIC_MODEL = "specific_model"


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ProviderHealth:
    """Provider health metrics."""
    status: ProviderStatus = ProviderStatus.HEALTHY
    response_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    consecutive_failures: int = 0
    circuit_breaker_until: Optional[datetime] = None


@dataclass
class RoutingRule:
    """Request routing rule."""
    condition: Callable[[LLMRequest], bool]
    provider: LLMProvider
    model: Optional[str] = None
    priority: int = 0
    enabled: bool = True


@dataclass
class InferenceMetrics:
    """Inference metrics tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    provider_usage: Dict[str, int] = field(default_factory=dict)
    model_usage: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0


class MiddlewarePipeline:
    """Middleware pipeline for request/response processing."""
    
    def __init__(self):
        self.pre_request_middleware: List[Callable] = []
        self.post_response_middleware: List[Callable] = []
        self.error_middleware: List[Callable] = []
    
    def add_pre_request_middleware(self, middleware: Callable) -> None:
        """Add pre-request middleware."""
        self.pre_request_middleware.append(middleware)
    
    def add_post_response_middleware(self, middleware: Callable) -> None:
        """Add post-response middleware."""
        self.post_response_middleware.append(middleware)
    
    def add_error_middleware(self, middleware: Callable) -> None:
        """Add error handling middleware."""
        self.error_middleware.append(middleware)
    
    async def process_request(self, request: LLMRequest, context: Dict[str, Any]) -> LLMRequest:
        """Process request through middleware pipeline."""
        for middleware in self.pre_request_middleware:
            try:
                if asyncio.iscoroutinefunction(middleware):
                    request = await middleware(request, context)
                else:
                    request = middleware(request, context)
            except Exception as e:
                logger.warning(f"Pre-request middleware error: {e}")
        return request
    
    async def process_response(self, response: LLMResponse, context: Dict[str, Any]) -> LLMResponse:
        """Process response through middleware pipeline."""
        for middleware in self.post_response_middleware:
            try:
                if asyncio.iscoroutinefunction(middleware):
                    response = await middleware(response, context)
                else:
                    response = middleware(response, context)
            except Exception as e:
                logger.warning(f"Post-response middleware error: {e}")
        return response
    
    async def process_error(self, error: Exception, context: Dict[str, Any]) -> Exception:
        """Process error through middleware pipeline."""
        for middleware in self.error_middleware:
            try:
                if asyncio.iscoroutinefunction(middleware):
                    error = await middleware(error, context)
                else:
                    error = middleware(error, context)
            except Exception as e:
                logger.warning(f"Error middleware error: {e}")
        return error


class CircuitBreaker:
    """Circuit breaker for provider failure handling."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: int = 60,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self) -> None:
        """Record successful request."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker closed - service recovered")
        else:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure = datetime.utcnow()
        
        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened - failure threshold reached")
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"
            self.success_count = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure and datetime.utcnow() - self.last_failure > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
                self.success_count = 0
                logger.info("Circuit breaker half-open - testing service")
                return True
            return False
        else:  # HALF_OPEN
            return True


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    _client_classes = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.SELF_HOSTED: SelfHostedClient,
        LLMProvider.AZURE_OPENAI: OpenAIClient,  # Uses OpenAI client with different base
        LLMProvider.LLAMA: SelfHostedClient,     # Treated as self-hosted
    }
    
    @classmethod
    def create_client(cls, config: LLMConfig) -> BaseLLMClient:
        """Create LLM client instance."""
        client_class = cls._client_classes.get(config.provider)
        if not client_class:
            raise ModelError(f"Unsupported provider: {config.provider}")
        
        try:
            return client_class(config)
        except Exception as e:
            raise ModelError(f"Failed to create {config.provider} client: {e}")
    
    @classmethod
    def register_provider(cls, provider: LLMProvider, client_class: type) -> None:
        """Register custom provider."""
        cls._client_classes[provider] = client_class
        logger.info(f"Registered custom provider: {provider}")


class InferenceRouter:
    """Main inference router with load balancing and failover."""
    
    def __init__(self, 
                 routing_strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
                 enable_circuit_breaker: bool = True,
                 health_check_interval: int = 300):
        
        self.routing_strategy = routing_strategy
        self.enable_circuit_breaker = enable_circuit_breaker
        self.health_check_interval = health_check_interval
        
        # Core components
        self.clients: Dict[str, BaseLLMClient] = {}
        self.provider_health: Dict[str, ProviderHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.routing_rules: List[RoutingRule] = []
        self.middleware = MiddlewarePipeline()
        self.metrics = InferenceMetrics()
        
        # State management
        self.round_robin_index = 0
        self.provider_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.model_capabilities: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self._health_check_task = None
        self._metrics_task = None
        self._running = False
    
    async def start(self) -> None:
        """Start the inference router."""
        self._running = True
        
        # Start background tasks
        if self.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        logger.info("Inference router started")
    
    async def stop(self) -> None:
        """Stop the inference router."""
        self._running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self._metrics_task:
            self._metrics_task.cancel()
        
        logger.info("Inference router stopped")
    
    def add_provider(self, 
                    provider_id: str, 
                    config: LLMConfig,
                    weight: float = 1.0,
                    capabilities: Optional[Set[str]] = None) -> None:
        """Add LLM provider."""
        try:
            client = LLMClientFactory.create_client(config)
            self.clients[provider_id] = client
            self.provider_health[provider_id] = ProviderHealth()
            
            if self.enable_circuit_breaker:
                self.circuit_breakers[provider_id] = CircuitBreaker()
            
            self.provider_weights[provider_id] = weight
            
            if capabilities:
                self.model_capabilities[provider_id] = capabilities
            
            logger.info(f"Added provider: {provider_id} ({config.provider.value})")
            
        except Exception as e:
            logger.error(f"Failed to add provider {provider_id}: {e}")
            raise
    
    def remove_provider(self, provider_id: str) -> None:
        """Remove LLM provider."""
        if provider_id in self.clients:
            del self.clients[provider_id]
            del self.provider_health[provider_id]
            
            if provider_id in self.circuit_breakers:
                del self.circuit_breakers[provider_id]
            
            if provider_id in self.provider_weights:
                del self.provider_weights[provider_id]
            
            if provider_id in self.model_capabilities:
                del self.model_capabilities[provider_id]
            
            logger.info(f"Removed provider: {provider_id}")
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add routing rule."""
        self.routing_rules.append(rule)
        # Sort by priority (higher first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule for {rule.provider}")
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers."""
        healthy = []
        for provider_id, health in self.provider_health.items():
            if health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                if self.enable_circuit_breaker:
                    breaker = self.circuit_breakers.get(provider_id)
                    if breaker and breaker.can_execute():
                        healthy.append(provider_id)
                else:
                    healthy.append(provider_id)
        return healthy
    
    def select_provider(self, request: LLMRequest) -> Optional[str]:
        """Select provider based on routing strategy."""
        # First check routing rules
        for rule in self.routing_rules:
            if rule.enabled and rule.condition(request):
                provider_id = self._find_provider_for_rule(rule)
                if provider_id:
                    return provider_id
        
        # Fallback to strategy-based selection
        healthy_providers = self.get_healthy_providers()
        if not healthy_providers:
            return None
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_providers)
        
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(healthy_providers)
        
        elif self.routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._select_fastest_response(healthy_providers)
        
        elif self.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._select_cost_optimized(healthy_providers, request)
        
        else:
            return random.choice(healthy_providers)
    
    def _find_provider_for_rule(self, rule: RoutingRule) -> Optional[str]:
        """Find provider matching routing rule."""
        target_provider = rule.provider
        
        for provider_id, client in self.clients.items():
            if client.get_provider() == target_provider:
                # Check if provider is healthy
                if provider_id in self.get_healthy_providers():
                    # Check if model matches (if specified)
                    if rule.model:
                        if client.config.model_name == rule.model:
                            return provider_id
                    else:
                        return provider_id
        
        return None
    
    def _select_round_robin(self, providers: List[str]) -> str:
        """Round-robin provider selection."""
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    def _select_least_loaded(self, providers: List[str]) -> str:
        """Select provider with least load."""
        # Use error count as load metric
        def load_score(provider_id: str) -> float:
            health = self.provider_health[provider_id]
            return health.error_count + (1.0 / health.success_rate)
        
        return min(providers, key=load_score)
    
    def _select_fastest_response(self, providers: List[str]) -> str:
        """Select provider with fastest response time."""
        def response_time(provider_id: str) -> float:
            health = self.provider_health[provider_id]
            return health.response_time or float('inf')
        
        return min(providers, key=response_time)
    
    def _select_cost_optimized(self, providers: List[str], request: LLMRequest) -> str:
        """Select most cost-effective provider."""
        # Simplified cost model - can be enhanced with actual pricing
        cost_weights = {
            LLMProvider.OPENAI: 1.0,
            LLMProvider.ANTHROPIC: 1.2,
            LLMProvider.SELF_HOSTED: 0.3,
            LLMProvider.LLAMA: 0.1,
        }
        
        def cost_score(provider_id: str) -> float:
            client = self.clients[provider_id]
            provider_type = client.get_provider()
            base_cost = cost_weights.get(provider_type, 1.0)
            
            # Adjust for response time (time is money)
            health = self.provider_health[provider_id]
            time_penalty = (health.response_time or 1.0) / 10.0
            
            return base_cost + time_penalty
        
        return min(providers, key=cost_score)
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete inference request."""
        start_time = time.time()
        context = {
            "request_id": f"req_{int(time.time() * 1000)}",
            "start_time": start_time,
            "provider_id": None,
        }
        
        try:
            # Process request through middleware
            request = await self.middleware.process_request(request, context)
            
            # Select provider
            provider_id = self.select_provider(request)
            if not provider_id:
                raise ModelError("No healthy providers available")
            
            context["provider_id"] = provider_id
            client = self.clients[provider_id]
            
            # Make request
            response = await client.complete(request)
            
            # Update metrics
            self._record_success(provider_id, response.response_time)
            
            # Process response through middleware
            response = await self.middleware.process_response(response, context)
            
            # Update usage metrics
            self._update_metrics(provider_id, client.get_provider(), request, response, success=True)
            
            logger.info(f"Request completed via {provider_id} in {response.response_time:.2f}s")
            return response
            
        except Exception as e:
            # Process error through middleware
            error = await self.middleware.process_error(e, context)
            
            # Record failure if provider was selected
            if context.get("provider_id"):
                self._record_failure(context["provider_id"], str(error))
                self._update_metrics(context["provider_id"], None, request, None, success=False, error=error)
            
            logger.error(f"Request failed: {error}")
            raise error
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamingChunk]:
        """Stream inference request."""
        context = {
            "request_id": f"stream_{int(time.time() * 1000)}",
            "start_time": time.time(),
            "provider_id": None,
        }
        
        try:
            # Process request through middleware
            request = await self.middleware.process_request(request, context)
            
            # Select provider
            provider_id = self.select_provider(request)
            if not provider_id:
                raise ModelError("No healthy providers available")
            
            context["provider_id"] = provider_id
            client = self.clients[provider_id]
            
            # Stream response
            chunk_count = 0
            async for chunk in client.stream(request):
                chunk_count += 1
                yield chunk
            
            # Record success
            response_time = time.time() - context["start_time"]
            self._record_success(provider_id, response_time)
            
            logger.info(f"Streaming completed via {provider_id}, {chunk_count} chunks")
            
        except Exception as e:
            # Process error through middleware
            error = await self.middleware.process_error(e, context)
            
            # Record failure
            if context.get("provider_id"):
                self._record_failure(context["provider_id"], str(error))
            
            logger.error(f"Streaming failed: {error}")
            raise error
    
    def _record_success(self, provider_id: str, response_time: float) -> None:
        """Record successful request."""
        health = self.provider_health[provider_id]
        health.last_success = datetime.utcnow()
        health.response_time = (health.response_time + response_time) / 2 if health.response_time else response_time
        health.consecutive_failures = 0
        
        # Update success rate (rolling average)
        total_requests = health.error_count + self.metrics.successful_requests
        if total_requests > 0:
            health.success_rate = self.metrics.successful_requests / total_requests
        
        if health.status == ProviderStatus.UNHEALTHY:
            health.status = ProviderStatus.HEALTHY
        
        # Update circuit breaker
        if self.enable_circuit_breaker and provider_id in self.circuit_breakers:
            self.circuit_breakers[provider_id].record_success()
    
    def _record_failure(self, provider_id: str, error_msg: str) -> None:
        """Record failed request."""
        health = self.provider_health[provider_id]
        health.last_error = datetime.utcnow()
        health.error_count += 1
        health.consecutive_failures += 1
        
        # Update success rate
        total_requests = health.error_count + self.metrics.successful_requests
        if total_requests > 0:
            health.success_rate = self.metrics.successful_requests / total_requests
        
        # Update status based on consecutive failures
        if health.consecutive_failures >= 3:
            health.status = ProviderStatus.UNHEALTHY
        elif health.consecutive_failures >= 1:
            health.status = ProviderStatus.DEGRADED
        
        # Update circuit breaker
        if self.enable_circuit_breaker and provider_id in self.circuit_breakers:
            self.circuit_breakers[provider_id].record_failure()
    
    def _update_metrics(self, 
                       provider_id: str, 
                       provider_type: Optional[LLMProvider],
                       request: LLMRequest, 
                       response: Optional[LLMResponse], 
                       success: bool,
                       error: Optional[Exception] = None) -> None:
        """Update usage metrics."""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
            if response:
                self.metrics.total_response_time += response.response_time
                if response.cached:
                    self.metrics.cache_hits += 1
                else:
                    self.metrics.cache_misses += 1
        else:
            self.metrics.failed_requests += 1
            if error:
                error_type = type(error).__name__
                self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
        
        # Update provider usage
        self.metrics.provider_usage[provider_id] = self.metrics.provider_usage.get(provider_id, 0) + 1
        
        # Update model usage
        model = request.model or (response.model if response else "unknown")
        self.metrics.model_usage[model] = self.metrics.model_usage.get(model, 0) + 1
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all providers."""
        for provider_id, client in self.clients.items():
            try:
                # Simple health check - create a minimal request
                health_request = LLMRequest(
                    messages=[Message(role="user", content="ping")],
                    max_tokens=1,
                    temperature=0.0
                )
                
                start_time = time.time()
                await client.complete(health_request)
                response_time = time.time() - start_time
                
                # Update health status
                health = self.provider_health[provider_id]
                health.response_time = response_time
                health.status = ProviderStatus.HEALTHY
                
                logger.debug(f"Health check passed for {provider_id}")
                
            except Exception as e:
                logger.warning(f"Health check failed for {provider_id}: {e}")
                health = self.provider_health[provider_id]
                health.status = ProviderStatus.UNHEALTHY
    
    async def _metrics_loop(self) -> None:
        """Background metrics collection loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                # Here you could send metrics to monitoring systems
                logger.debug(f"Metrics: {self.get_metrics_summary()}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        total = self.metrics.total_requests
        avg_response_time = (
            self.metrics.total_response_time / self.metrics.successful_requests
            if self.metrics.successful_requests > 0 else 0
        )
        
        return {
            "total_requests": total,
            "success_rate": self.metrics.successful_requests / total if total > 0 else 0,
            "average_response_time": avg_response_time,
            "cache_hit_rate": (
                self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
            ),
            "provider_usage": dict(self.metrics.provider_usage),
            "model_usage": dict(self.metrics.model_usage),
            "error_distribution": dict(self.metrics.error_types),
            "provider_health": {
                pid: {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "success_rate": health.success_rate,
                    "error_count": health.error_count
                }
                for pid, health in self.provider_health.items()
            }
        }
    
    def get_provider_status(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed provider status."""
        if provider_id not in self.clients:
            return None
        
        health = self.provider_health[provider_id]
        client = self.clients[provider_id]
        
        return {
            "provider_id": provider_id,
            "provider_type": client.get_provider().value,
            "model_info": client.get_model_info(),
            "health": {
                "status": health.status.value,
                "response_time": health.response_time,
                "success_rate": health.success_rate,
                "error_count": health.error_count,
                "consecutive_failures": health.consecutive_failures,
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_error": health.last_error.isoformat() if health.last_error else None,
            },
            "circuit_breaker": (
                self.circuit_breakers[provider_id].state
                if provider_id in self.circuit_breakers else None
            ),
            "weight": self.provider_weights[provider_id],
            "capabilities": list(self.model_capabilities.get(provider_id, set()))
        }


# Built-in middleware functions
async def request_logging_middleware(request: LLMRequest, context: Dict[str, Any]) -> LLMRequest:
    """Log request details."""
    logger.info(f"Processing request {context['request_id']}: "
               f"{len(request.messages)} messages, model: {request.model}")
    return request


async def response_logging_middleware(response: LLMResponse, context: Dict[str, Any]) -> LLMResponse:
    """Log response details."""
    logger.info(f"Response for {context['request_id']}: "
               f"{len(response.content)} chars, {response.usage.get('total_tokens', 0)} tokens")
    return response


async def security_middleware(request: LLMRequest, context: Dict[str, Any]) -> LLMRequest:
    """Additional security checks."""
    # Add custom security validation here
    for message in request.messages:
        if len(message.content) > 50000:  # 50KB limit
            raise SecurityError("Message content too large")
    
    return request


def cost_tracking_middleware(response: LLMResponse, context: Dict[str, Any]) -> LLMResponse:
    """Track costs per request."""
    # Add cost calculation based on tokens and provider
    tokens = response.usage.get('total_tokens', 0)
    estimated_cost = tokens * 0.00002  # Rough estimate
    response.metadata['estimated_cost'] = estimated_cost
    return response
