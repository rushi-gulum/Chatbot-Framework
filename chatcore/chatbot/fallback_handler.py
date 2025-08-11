"""
Enterprise Fallback Handler for Chatbot Framework

This module provides intelligent fallback mechanisms with:
- Multi-level fallback strategies
- Context-aware error recovery
- Escalation path management
- Performance monitoring and learning
"""

from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base_core import IFallbackHandler, ChatMessage, ConversationContext, secure_logger


class FallbackTrigger(Enum):
    """Triggers that activate fallback mechanisms."""
    LOW_CONFIDENCE = "low_confidence"
    PARSING_ERROR = "parsing_error"
    INTENT_NOT_FOUND = "intent_not_found"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    UNEXPECTED_ERROR = "unexpected_error"
    USER_CONFUSION = "user_confusion"
    REPEATED_FAILURE = "repeated_failure"


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    CLARIFICATION = "clarification"
    SUGGESTION = "suggestion"
    GENERIC_HELP = "generic_help"
    HUMAN_HANDOFF = "human_handoff"
    APOLOGY = "apology"
    REDIRECT = "redirect"
    SEARCH = "search"
    ALTERNATIVE_PHRASING = "alternative_phrasing"
    CONTEXT_RESET = "context_reset"
    EDUCATIONAL = "educational"


@dataclass
class FallbackContext:
    """Context information for fallback handling."""
    trigger: FallbackTrigger
    original_query: str
    confidence_scores: Dict[str, float]
    attempt_count: int
    session_failure_count: int
    last_successful_intent: Optional[str]
    conversation_length: int
    user_frustration_level: float
    available_services: List[str]
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class FallbackResponse:
    """Response from fallback handling."""
    strategy_used: FallbackStrategy
    response_text: str
    next_actions: List[str]
    escalation_recommended: bool
    confidence: float
    fallback_data: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_used': self.strategy_used.value,
            'response_text': self.response_text,
            'next_actions': self.next_actions,
            'escalation_recommended': self.escalation_recommended,
            'confidence': self.confidence,
            'fallback_data': self.fallback_data,
            'timestamp': self.timestamp.isoformat()
        }


class FrustrationDetector:
    """Detects user frustration patterns."""
    
    def __init__(self):
        self.frustration_indicators = [
            r'\b(help|stuck|confused|frustrated|annoying|stupid|useless)\b',
            r'\b(what the|wtf|seriously|come on|this is)\b',
            r'\b(doesn\'t work|not working|broken|wrong)\b',
            r'\b(give up|forget it|never mind)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r'[?]{2,}',  # Multiple question marks
            r'\b(please|just|simply)\b.*\b(help|work|understand)\b'
        ]
    
    def detect_frustration_level(self, text: str, conversation_history: List[str]) -> float:
        """Detect user frustration level (0.0 to 1.0)."""
        import re
        
        frustration_score = 0.0
        text_lower = text.lower()
        
        # Check for frustration indicators in current message
        for pattern in self.frustration_indicators:
            matches = len(re.findall(pattern, text_lower))
            frustration_score += matches * 0.2
        
        # Check for repeated similar messages (indicates frustration)
        if conversation_history:
            recent_messages = conversation_history[-3:]
            similar_count = sum(1 for msg in recent_messages if self._similarity(text, msg) > 0.7)
            frustration_score += similar_count * 0.3
        
        # Check for increasing message length (users tend to explain more when frustrated)
        if len(text) > 100:
            frustration_score += 0.1
        
        return min(1.0, frustration_score)
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class ResponseGenerator:
    """Generates contextual fallback responses."""
    
    def __init__(self):
        self.response_templates = {
            FallbackStrategy.CLARIFICATION: [
                "I want to help, but I need more information. Could you please provide more details about {topic}?",
                "I'm not quite sure I understand. Could you rephrase that or give me more context?",
                "Help me understand better - are you looking for {suggestion1} or {suggestion2}?",
                "Could you be more specific about what you'd like me to help you with?"
            ],
            FallbackStrategy.SUGGESTION: [
                "I might not have understood exactly, but here are some things I can help you with: {suggestions}",
                "Based on what you've said, you might be interested in: {suggestions}",
                "I can help you with several things. Would any of these work: {suggestions}?",
                "Here are some popular options that might interest you: {suggestions}"
            ],
            FallbackStrategy.GENERIC_HELP: [
                "I'm here to help! I can assist you with {capabilities}. What would you like to do?",
                "I'd be happy to help you. Some things I can do include: {capabilities}",
                "Let me know how I can assist you. I can help with: {capabilities}",
                "I'm designed to help with {capabilities}. What can I do for you today?"
            ],
            FallbackStrategy.APOLOGY: [
                "I apologize, but I'm having trouble understanding that. Let me try to help in a different way.",
                "Sorry about the confusion. Let me see if I can assist you better.",
                "I apologize for the difficulty. Can we try approaching this differently?",
                "Sorry, I seem to be missing something. Could you help me understand what you need?"
            ],
            FallbackStrategy.HUMAN_HANDOFF: [
                "I think you might benefit from speaking with a human agent who can better assist you. Would you like me to connect you?",
                "For complex requests like this, our human support team would be better equipped to help. Shall I transfer you?",
                "I'd like to connect you with a specialist who can provide more detailed assistance. Is that okay?",
                "This seems like something our human support team should handle. Would you like me to escalate this?"
            ],
            FallbackStrategy.REDIRECT: [
                "While I can't help with that specific request, I can assist you with {alternatives}.",
                "That's outside my current capabilities, but I can help you with {alternatives} instead.",
                "I'm not able to handle that, but here's what I can do: {alternatives}",
                "Let me redirect you to something I can definitely help with: {alternatives}"
            ],
            FallbackStrategy.EDUCATIONAL: [
                "Let me explain how this works: {explanation}. Does that help clarify things?",
                "Here's some information that might be useful: {explanation}",
                "To help you better understand, {explanation}. Would you like to know more?",
                "I can provide some background: {explanation}. Is this what you were looking for?"
            ]
        }
    
    def generate_response(self, strategy: FallbackStrategy, context: FallbackContext,
                         available_data: Dict[str, Any]) -> str:
        """Generate a contextual response based on strategy and context."""
        templates = self.response_templates.get(strategy, ["I'm not sure how to help with that."])
        template = random.choice(templates)
        
        # Fill in template variables based on available data
        try:
            if strategy == FallbackStrategy.SUGGESTION:
                suggestions = available_data.get('suggestions', ['general assistance', 'information lookup'])
                template = template.format(suggestions=', '.join(suggestions[:3]))
            
            elif strategy == FallbackStrategy.GENERIC_HELP:
                capabilities = available_data.get('capabilities', ['answering questions', 'providing information'])
                template = template.format(capabilities=', '.join(capabilities[:3]))
            
            elif strategy == FallbackStrategy.CLARIFICATION:
                topic = available_data.get('topic', 'your request')
                suggestion1 = available_data.get('suggestion1', 'option A')
                suggestion2 = available_data.get('suggestion2', 'option B')
                template = template.format(topic=topic, suggestion1=suggestion1, suggestion2=suggestion2)
            
            elif strategy == FallbackStrategy.REDIRECT:
                alternatives = available_data.get('alternatives', ['general assistance'])
                template = template.format(alternatives=', '.join(alternatives[:3]))
            
            elif strategy == FallbackStrategy.EDUCATIONAL:
                explanation = available_data.get('explanation', 'the system works by processing your requests')
                template = template.format(explanation=explanation)
        
        except KeyError:
            # If template formatting fails, use a safe fallback
            template = "I'm here to help, but I need more information to assist you properly."
        
        return template


class FallbackHandler(IFallbackHandler):
    """Enterprise fallback handler with intelligent recovery strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fallback handler."""
        self.config = config or {}
        self.frustration_detector = FrustrationDetector()
        self.response_generator = ResponseGenerator()
        
        # Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.max_attempts_before_escalation = self.config.get('max_attempts_before_escalation', 3)
        self.frustration_threshold = self.config.get('frustration_threshold', 0.7)
        self.enable_learning = self.config.get('enable_learning', True)
        
        # Session tracking
        self.session_failures: Dict[str, List[Dict[str, Any]]] = {}
        self.session_patterns: Dict[str, Dict[str, int]] = {}
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness: Dict[FallbackStrategy, Dict[str, float]] = {
            strategy: {'success_count': 0, 'total_count': 0, 'effectiveness': 0.5}
            for strategy in FallbackStrategy
        }
        
        # Available system capabilities for suggestions
        self.system_capabilities = self.config.get('capabilities', [
            'answering questions',
            'providing information',
            'helping with bookings',
            'general assistance'
        ])
        
        # Performance metrics
        self.metrics = {
            'total_fallbacks': 0,
            'successful_recoveries': 0,
            'escalations': 0,
            'strategy_usage': {s.value: 0 for s in FallbackStrategy},
            'trigger_frequency': {t.value: 0 for t in FallbackTrigger}
        }
        
        secure_logger.info("FallbackHandler initialized", extra={
            'confidence_threshold': self.confidence_threshold,
            'max_attempts': self.max_attempts_before_escalation,
            'learning_enabled': self.enable_learning
        })
    
    async def handle_fallback(self, trigger: str, context: Dict[str, Any]) -> FallbackResponse:
        """
        Handle fallback situation with appropriate strategy.
        
        Args:
            trigger: What triggered the fallback
            context: Context information including conversation state
            
        Returns:
            Fallback response with strategy and next actions
        """
        try:
            # Parse trigger
            fallback_trigger = FallbackTrigger(trigger)
            
            # Build fallback context
            fallback_context = self._build_fallback_context(fallback_trigger, context)
            
            # Update session tracking
            self._update_session_tracking(fallback_context)
            
            # Select appropriate strategy
            strategy = self._select_strategy(fallback_context)
            
            # Generate response
            response_text = self._generate_response(strategy, fallback_context)
            
            # Determine next actions
            next_actions = self._determine_next_actions(strategy, fallback_context)
            
            # Check if escalation is recommended
            escalation_recommended = self._should_escalate(fallback_context)
            
            # Calculate confidence in this fallback
            confidence = self._calculate_fallback_confidence(strategy, fallback_context)
            
            # Create response
            response = FallbackResponse(
                strategy_used=strategy,
                response_text=response_text,
                next_actions=next_actions,
                escalation_recommended=escalation_recommended,
                confidence=confidence,
                fallback_data={
                    'trigger': trigger,
                    'attempt_count': fallback_context.attempt_count,
                    'frustration_level': fallback_context.user_frustration_level
                },
                timestamp=datetime.utcnow()
            )
            
            # Update metrics
            self._update_metrics(fallback_trigger, strategy, response)
            
            secure_logger.info("Fallback handled", extra={
                'trigger': trigger,
                'strategy': strategy.value,
                'escalation_recommended': escalation_recommended,
                'confidence': confidence,
                'session_id': context.get('session_id', 'unknown')
            })
            
            return response
            
        except Exception as e:
            secure_logger.error(f"Error in fallback handler: {str(e)}", extra={
                'trigger': trigger,
                'error_type': type(e).__name__
            })
            
            # Return safe emergency fallback
            return FallbackResponse(
                strategy_used=FallbackStrategy.APOLOGY,
                response_text="I'm sorry, I'm experiencing some technical difficulties. Please try again or contact support.",
                next_actions=['retry', 'contact_support'],
                escalation_recommended=True,
                confidence=0.1,
                fallback_data={'error': str(e)},
                timestamp=datetime.utcnow()
            )
    
    def _build_fallback_context(self, trigger: FallbackTrigger, context: Dict[str, Any]) -> FallbackContext:
        """Build comprehensive fallback context."""
        session_id = context.get('session_id', 'unknown')
        original_query = context.get('original_query', '')
        conversation_history = context.get('conversation_history', [])
        
        # Calculate frustration level
        frustration_level = self.frustration_detector.detect_frustration_level(
            original_query, conversation_history
        )
        
        # Get session failure count
        session_failures = self.session_failures.get(session_id, [])
        
        # Count attempts for this specific query/issue
        attempt_count = sum(1 for failure in session_failures 
                           if failure.get('query', '') == original_query) + 1
        
        return FallbackContext(
            trigger=trigger,
            original_query=original_query,
            confidence_scores=context.get('confidence_scores', {}),
            attempt_count=attempt_count,
            session_failure_count=len(session_failures),
            last_successful_intent=context.get('last_successful_intent'),
            conversation_length=len(conversation_history),
            user_frustration_level=frustration_level,
            available_services=context.get('available_services', self.system_capabilities),
            error_details=context.get('error_details')
        )
    
    def _select_strategy(self, context: FallbackContext) -> FallbackStrategy:
        """Select the most appropriate fallback strategy."""
        # High frustration or many failures -> escalate
        if (context.user_frustration_level > self.frustration_threshold or 
            context.session_failure_count >= self.max_attempts_before_escalation):
            return FallbackStrategy.HUMAN_HANDOFF
        
        # Strategy selection based on trigger
        if context.trigger == FallbackTrigger.LOW_CONFIDENCE:
            if context.attempt_count == 1:
                return FallbackStrategy.CLARIFICATION
            else:
                return FallbackStrategy.SUGGESTION
        
        elif context.trigger == FallbackTrigger.INTENT_NOT_FOUND:
            return FallbackStrategy.SUGGESTION if context.attempt_count <= 2 else FallbackStrategy.GENERIC_HELP
        
        elif context.trigger == FallbackTrigger.PARSING_ERROR:
            return FallbackStrategy.ALTERNATIVE_PHRASING if context.attempt_count == 1 else FallbackStrategy.CLARIFICATION
        
        elif context.trigger == FallbackTrigger.SERVICE_UNAVAILABLE:
            return FallbackStrategy.REDIRECT
        
        elif context.trigger == FallbackTrigger.USER_CONFUSION:
            return FallbackStrategy.EDUCATIONAL if context.attempt_count <= 2 else FallbackStrategy.GENERIC_HELP
        
        elif context.trigger == FallbackTrigger.REPEATED_FAILURE:
            return FallbackStrategy.HUMAN_HANDOFF
        
        # Default strategy based on effectiveness
        return self._get_most_effective_strategy()
    
    def _get_most_effective_strategy(self) -> FallbackStrategy:
        """Get the most effective strategy based on historical performance."""
        best_strategy = FallbackStrategy.CLARIFICATION
        best_effectiveness = 0.0
        
        for strategy, stats in self.strategy_effectiveness.items():
            if stats['total_count'] > 0 and stats['effectiveness'] > best_effectiveness:
                best_effectiveness = stats['effectiveness']
                best_strategy = strategy
        
        return best_strategy
    
    def _generate_response(self, strategy: FallbackStrategy, context: FallbackContext) -> str:
        """Generate appropriate response text."""
        available_data = {
            'suggestions': context.available_services,
            'capabilities': self.system_capabilities,
            'topic': context.original_query,
            'suggestion1': 'getting information',
            'suggestion2': 'booking assistance',
            'alternatives': context.available_services,
            'explanation': 'I process your requests and try to provide helpful responses'
        }
        
        return self.response_generator.generate_response(strategy, context, available_data)
    
    def _determine_next_actions(self, strategy: FallbackStrategy, context: FallbackContext) -> List[str]:
        """Determine recommended next actions."""
        actions = []
        
        if strategy == FallbackStrategy.CLARIFICATION:
            actions = ['wait_for_clarification', 'provide_examples']
        elif strategy == FallbackStrategy.SUGGESTION:
            actions = ['present_options', 'wait_for_selection']
        elif strategy == FallbackStrategy.GENERIC_HELP:
            actions = ['show_capabilities', 'wait_for_new_request']
        elif strategy == FallbackStrategy.HUMAN_HANDOFF:
            actions = ['initiate_transfer', 'collect_contact_info']
        elif strategy == FallbackStrategy.REDIRECT:
            actions = ['show_alternatives', 'redirect_to_service']
        elif strategy == FallbackStrategy.EDUCATIONAL:
            actions = ['provide_explanation', 'offer_examples']
        else:
            actions = ['wait_for_input', 'monitor_response']
        
        return actions
    
    def _should_escalate(self, context: FallbackContext) -> bool:
        """Determine if escalation is recommended."""
        return (
            context.user_frustration_level > self.frustration_threshold or
            context.session_failure_count >= self.max_attempts_before_escalation or
            context.trigger == FallbackTrigger.REPEATED_FAILURE or
            context.attempt_count >= 4
        )
    
    def _calculate_fallback_confidence(self, strategy: FallbackStrategy, context: FallbackContext) -> float:
        """Calculate confidence in the fallback strategy."""
        base_confidence = self.strategy_effectiveness[strategy]['effectiveness']
        
        # Adjust based on context
        if context.user_frustration_level > 0.5:
            base_confidence *= 0.8  # Lower confidence when user is frustrated
        
        if context.attempt_count > 2:
            base_confidence *= 0.7  # Lower confidence on repeated attempts
        
        if context.session_failure_count > 3:
            base_confidence *= 0.6  # Lower confidence in problematic sessions
        
        return max(0.1, min(1.0, base_confidence))
    
    def _update_session_tracking(self, context: FallbackContext) -> None:
        """Update session tracking with current failure."""
        session_id = context.original_query  # Use query as session identifier for this example
        
        if session_id not in self.session_failures:
            self.session_failures[session_id] = []
        
        failure_record = {
            'trigger': context.trigger.value,
            'query': context.original_query,
            'timestamp': datetime.utcnow(),
            'frustration_level': context.user_frustration_level
        }
        
        self.session_failures[session_id].append(failure_record)
        
        # Cleanup old failures (keep only last 10 per session)
        self.session_failures[session_id] = self.session_failures[session_id][-10:]
    
    def _update_metrics(self, trigger: FallbackTrigger, strategy: FallbackStrategy, 
                       response: FallbackResponse) -> None:
        """Update performance metrics."""
        self.metrics['total_fallbacks'] += 1
        self.metrics['strategy_usage'][strategy.value] += 1
        self.metrics['trigger_frequency'][trigger.value] += 1
        
        if response.escalation_recommended:
            self.metrics['escalations'] += 1
    
    async def record_fallback_outcome(self, session_id: str, strategy: str, 
                                    success: bool) -> None:
        """Record the outcome of a fallback attempt for learning."""
        try:
            if not self.enable_learning:
                return
            
            strategy_enum = FallbackStrategy(strategy)
            stats = self.strategy_effectiveness[strategy_enum]
            
            stats['total_count'] += 1
            if success:
                stats['success_count'] += 1
                self.metrics['successful_recoveries'] += 1
            
            # Update effectiveness (rolling average)
            stats['effectiveness'] = stats['success_count'] / stats['total_count']
            
            secure_logger.info("Fallback outcome recorded", extra={
                'session_id': session_id,
                'strategy': strategy,
                'success': success,
                'effectiveness': stats['effectiveness']
            })
            
        except Exception as e:
            secure_logger.error(f"Error recording fallback outcome: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fallback handler performance metrics."""
        recovery_rate = (
            self.metrics['successful_recoveries'] / self.metrics['total_fallbacks']
            if self.metrics['total_fallbacks'] > 0 else 0
        )
        
        escalation_rate = (
            self.metrics['escalations'] / self.metrics['total_fallbacks']
            if self.metrics['total_fallbacks'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'recovery_rate': recovery_rate,
            'escalation_rate': escalation_rate,
            'strategy_effectiveness': {
                k.value: v['effectiveness'] for k, v in self.strategy_effectiveness.items()
            },
            'active_sessions': len(self.session_failures)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the fallback handler."""
        try:
            # Test fallback handling
            test_context = {
                'session_id': 'test_session',
                'original_query': 'test query',
                'conversation_history': [],
                'confidence_scores': {'intent': 0.3}
            }
            
            response = await self.handle_fallback('low_confidence', test_context)
            
            return {
                'status': 'healthy',
                'test_successful': response is not None,
                'test_strategy': response.strategy_used.value,
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.get_metrics()
            }


# Export the main classes
__all__ = ['FallbackHandler', 'FallbackResponse', 'FallbackTrigger', 'FallbackStrategy']
