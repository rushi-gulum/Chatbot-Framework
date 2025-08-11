"""
Enterprise Dialogue Manager for Chatbot Framework

This module provides intelligent conversation flow management with:
- Context-aware dialogue state tracking
- Intent recognition and slot filling
- Multi-turn conversation handling
- Dynamic response generation strategies
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .base_core import IDialogueManager, ChatMessage, ConversationContext, SentimentType, secure_logger


class DialogueState(Enum):
    """Dialogue state enumeration."""
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    TASK_EXECUTION = "task_execution"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    FAREWELL = "farewell"
    ERROR_HANDLING = "error_handling"
    ESCALATION = "escalation"


class IntentType(Enum):
    """User intent categories."""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    INFORMATION = "information"
    HELP = "help"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


@dataclass
class Slot:
    """Dialogue slot for information collection."""
    name: str
    value: Optional[Any] = None
    confidence: float = 0.0
    confirmed: bool = False
    required: bool = True
    prompt: str = ""
    validation_pattern: Optional[str] = None


@dataclass
class Intent:
    """Recognized user intent."""
    type: IntentType
    confidence: float
    slots: Dict[str, Slot] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)


@dataclass
class DialogueContext:
    """Extended dialogue context with state tracking."""
    state: DialogueState
    intent: Optional[Intent]
    slots: Dict[str, Slot]
    conversation_history: List[ChatMessage]
    turn_count: int
    last_bot_action: Optional[str]
    pending_confirmations: List[str]
    collected_information: Dict[str, Any]
    user_preferences: Dict[str, Any]
    session_start_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'state': self.state.value,
            'intent': self.intent.__dict__ if self.intent else None,
            'slots': {k: v.__dict__ for k, v in self.slots.items()},
            'turn_count': self.turn_count,
            'last_bot_action': self.last_bot_action,
            'pending_confirmations': self.pending_confirmations,
            'collected_information': self.collected_information,
            'user_preferences': self.user_preferences,
            'session_start_time': self.session_start_time.isoformat()
        }


class IntentRecognizer:
    """Simple rule-based intent recognition."""
    
    def __init__(self):
        self.intent_patterns = {
            IntentType.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(how are you|what\'s up|greetings)\b'
            ],
            IntentType.GOODBYE: [
                r'\b(bye|goodbye|see you|farewell|have a good day)\b',
                r'\b(thank you|thanks|that\'s all)\b.*\b(bye|goodbye)\b'
            ],
            IntentType.QUESTION: [
                r'\b(what|where|when|why|how|who|which)\b',
                r'.*\?$',
                r'\b(can you tell me|do you know|could you explain)\b'
            ],
            IntentType.REQUEST: [
                r'\b(please|can you|could you|would you|i need|i want)\b',
                r'\b(help me|assist me|show me)\b'
            ],
            IntentType.COMPLAINT: [
                r'\b(problem|issue|wrong|error|not working|broken)\b',
                r'\b(disappointed|frustrated|angry|upset)\b',
                r'\b(complaint|complain|dissatisfied)\b'
            ],
            IntentType.COMPLIMENT: [
                r'\b(great|excellent|amazing|wonderful|fantastic|good job)\b',
                r'\b(thank you|thanks|appreciate|helpful)\b',
                r'\b(love|like|impressed)\b'
            ],
            IntentType.BOOKING: [
                r'\b(book|reserve|schedule|appointment|meeting)\b',
                r'\b(available|availability|free time)\b'
            ],
            IntentType.CANCELLATION: [
                r'\b(cancel|cancellation|remove|delete)\b',
                r'\b(don\'t want|no longer need|change my mind)\b'
            ],
            IntentType.INFORMATION: [
                r'\b(information|details|specs|features|about)\b',
                r'\b(tell me about|more about|learn about)\b'
            ],
            IntentType.HELP: [
                r'\b(help|support|assistance|guide)\b',
                r'\b(how to|how do i|can you help)\b'
            ]
        }
        
        self.slot_patterns = {
            'name': r'\b(my name is|i\'m|call me|i am)\s+([A-Za-z]+)\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'date': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'time': r'\b(\d{1,2}:\d{2}(?:\s?[ap]m)?)\b',
            'number': r'\b\d+\b'
        }
    
    def recognize_intent(self, text: str) -> Intent:
        """Recognize intent from user text."""
        text_lower = text.lower()
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        matched_keywords = []
        
        for intent_type, patterns in self.intent_patterns.items():
            confidence = 0.0
            keywords = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    confidence += 0.3  # Each pattern match adds confidence
                    if isinstance(matches[0], str):
                        keywords.extend([matches[0]])
                    else:
                        keywords.extend([m for m in matches[0] if isinstance(m, str)])
            
            # Normalize confidence
            confidence = min(1.0, confidence)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_type
                matched_keywords = keywords
        
        # Extract slots
        slots = self._extract_slots(text)
        
        return Intent(
            type=best_intent,
            confidence=best_confidence,
            slots=slots,
            keywords=matched_keywords
        )
    
    def _extract_slots(self, text: str) -> Dict[str, Slot]:
        """Extract slots from text using patterns."""
        slots = {}
        
        for slot_name, pattern in self.slot_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if slot_name == 'name' and isinstance(matches[0], tuple):
                    value = matches[0][1]  # Get the captured group
                else:
                    value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                
                slots[slot_name] = Slot(
                    name=slot_name,
                    value=value,
                    confidence=0.8,  # Pattern-based extraction confidence
                    confirmed=False,
                    required=True
                )
        
        return slots


class DialogueManager(IDialogueManager):
    """Enterprise dialogue manager with state tracking and flow control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dialogue manager."""
        self.config = config or {}
        self.intent_recognizer = IntentRecognizer()
        
        # Active dialogue contexts by session
        self.dialogue_contexts: Dict[str, DialogueContext] = {}
        
        # Configuration
        self.max_turns_per_session = self.config.get('max_turns_per_session', 50)
        self.session_timeout_minutes = self.config.get('session_timeout_minutes', 30)
        self.confirmation_threshold = self.config.get('confirmation_threshold', 0.7)
        
        # Response templates
        self.response_templates = {
            DialogueState.GREETING: [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Welcome! I'm here to assist you."
            ],
            DialogueState.CLARIFICATION: [
                "I'm not sure I understand. Could you please clarify?",
                "Can you provide more details about that?",
                "Could you rephrase that for me?"
            ],
            DialogueState.CONFIRMATION: [
                "Just to confirm, you want to {action}. Is that correct?",
                "Let me make sure I understand: {details}. Is this right?",
                "Can you confirm that {information}?"
            ],
            DialogueState.FAREWELL: [
                "Thank you for using our service. Have a great day!",
                "Goodbye! Feel free to reach out if you need anything else.",
                "Take care! I'm here whenever you need help."
            ]
        }
        
        # Performance metrics
        self.metrics = {
            'sessions_managed': 0,
            'total_turns': 0,
            'successful_completions': 0,
            'escalations': 0,
            'average_session_length': 0.0
        }
        
        secure_logger.info("DialogueManager initialized", extra={
            'max_turns': self.max_turns_per_session,
            'session_timeout': self.session_timeout_minutes
        })
    
    async def process_message(self, message: ChatMessage, context: ConversationContext) -> Dict[str, Any]:
        """
        Process incoming message and manage dialogue flow.
        
        Args:
            message: User message
            context: Conversation context
            
        Returns:
            Processing result with response strategy
        """
        try:
            session_id = context.session_id
            
            # Get or create dialogue context
            dialogue_ctx = self._get_or_create_dialogue_context(session_id, context)
            
            # Clean up expired sessions
            await self._cleanup_expired_sessions()
            
            # Recognize intent
            intent = self.intent_recognizer.recognize_intent(message.content)
            dialogue_ctx.intent = intent
            dialogue_ctx.turn_count += 1
            
            # Update conversation history
            dialogue_ctx.conversation_history.append(message)
            
            # Determine next dialogue state
            next_state = self._determine_next_state(dialogue_ctx, intent)
            previous_state = dialogue_ctx.state
            dialogue_ctx.state = next_state
            
            # Update slots with extracted information
            self._update_slots(dialogue_ctx, intent.slots)
            
            # Generate response strategy
            response_strategy = await self._generate_response_strategy(dialogue_ctx, intent)
            
            # Update metrics
            self._update_metrics(dialogue_ctx, previous_state, next_state)
            
            secure_logger.info("Message processed by dialogue manager", extra={
                'session_id': session_id,
                'intent': intent.type.value,
                'intent_confidence': intent.confidence,
                'previous_state': previous_state.value,
                'next_state': next_state.value,
                'turn_count': dialogue_ctx.turn_count
            })
            
            return {
                'response_strategy': response_strategy,
                'dialogue_state': next_state.value,
                'intent': intent.type.value,
                'intent_confidence': intent.confidence,
                'slots_filled': len([s for s in dialogue_ctx.slots.values() if s.value]),
                'turn_count': dialogue_ctx.turn_count,
                'session_context': dialogue_ctx.to_dict()
            }
            
        except Exception as e:
            secure_logger.error(f"Error processing message in dialogue manager: {str(e)}", extra={
                'session_id': context.session_id,
                'message_length': len(message.content),
                'error_type': type(e).__name__
            })
            
            # Return safe fallback
            return {
                'response_strategy': 'fallback',
                'dialogue_state': DialogueState.ERROR_HANDLING.value,
                'intent': IntentType.UNKNOWN.value,
                'intent_confidence': 0.0,
                'error': str(e)
            }
    
    def _get_or_create_dialogue_context(self, session_id: str, context: ConversationContext) -> DialogueContext:
        """Get existing or create new dialogue context."""
        if session_id not in self.dialogue_contexts:
            dialogue_ctx = DialogueContext(
                state=DialogueState.GREETING,
                intent=None,
                slots={},
                conversation_history=[],
                turn_count=0,
                last_bot_action=None,
                pending_confirmations=[],
                collected_information={},
                user_preferences={},
                session_start_time=datetime.utcnow()
            )
            self.dialogue_contexts[session_id] = dialogue_ctx
            self.metrics['sessions_managed'] += 1
        
        return self.dialogue_contexts[session_id]
    
    def _determine_next_state(self, dialogue_ctx: DialogueContext, intent: Intent) -> DialogueState:
        """Determine next dialogue state based on current context and intent."""
        current_state = dialogue_ctx.state
        intent_type = intent.type
        
        # State transition logic
        if current_state == DialogueState.GREETING:
            if intent_type == IntentType.GREETING:
                return DialogueState.INFORMATION_GATHERING
            elif intent_type in [IntentType.REQUEST, IntentType.QUESTION, IntentType.BOOKING]:
                return DialogueState.INFORMATION_GATHERING
            elif intent_type == IntentType.COMPLAINT:
                return DialogueState.ERROR_HANDLING
            elif intent_type == IntentType.GOODBYE:
                return DialogueState.FAREWELL
        
        elif current_state == DialogueState.INFORMATION_GATHERING:
            if intent_type == IntentType.COMPLAINT:
                return DialogueState.ERROR_HANDLING
            elif self._all_required_slots_filled(dialogue_ctx):
                return DialogueState.CONFIRMATION
            elif intent_confidence := intent.confidence < 0.5:
                return DialogueState.CLARIFICATION
            else:
                return DialogueState.INFORMATION_GATHERING
        
        elif current_state == DialogueState.CLARIFICATION:
            if intent.confidence > 0.7:
                return DialogueState.INFORMATION_GATHERING
            elif dialogue_ctx.turn_count > 3:
                return DialogueState.ESCALATION
            else:
                return DialogueState.CLARIFICATION
        
        elif current_state == DialogueState.CONFIRMATION:
            if intent_type in [IntentType.COMPLIMENT, IntentType.GOODBYE]:
                return DialogueState.FAREWELL
            elif intent_type == IntentType.COMPLAINT:
                return DialogueState.ERROR_HANDLING
            else:
                return DialogueState.TASK_EXECUTION
        
        elif current_state == DialogueState.TASK_EXECUTION:
            if intent_type == IntentType.GOODBYE:
                return DialogueState.FAREWELL
            elif intent_type == IntentType.COMPLAINT:
                return DialogueState.ERROR_HANDLING
            else:
                return DialogueState.CONFIRMATION
        
        elif current_state == DialogueState.ERROR_HANDLING:
            if dialogue_ctx.turn_count > 5:
                return DialogueState.ESCALATION
            elif intent_type == IntentType.GOODBYE:
                return DialogueState.FAREWELL
            else:
                return DialogueState.INFORMATION_GATHERING
        
        # Default: stay in current state
        return current_state
    
    def _all_required_slots_filled(self, dialogue_ctx: DialogueContext) -> bool:
        """Check if all required slots are filled."""
        if not dialogue_ctx.slots:
            return False
        
        return all(
            slot.value is not None and slot.confirmed
            for slot in dialogue_ctx.slots.values()
            if slot.required
        )
    
    def _update_slots(self, dialogue_ctx: DialogueContext, new_slots: Dict[str, Slot]) -> None:
        """Update dialogue context slots with new information."""
        for slot_name, slot in new_slots.items():
            if slot_name in dialogue_ctx.slots:
                # Update existing slot
                existing_slot = dialogue_ctx.slots[slot_name]
                if slot.confidence > existing_slot.confidence:
                    existing_slot.value = slot.value
                    existing_slot.confidence = slot.confidence
            else:
                # Add new slot
                dialogue_ctx.slots[slot_name] = slot
    
    async def _generate_response_strategy(self, dialogue_ctx: DialogueContext, intent: Intent) -> str:
        """Generate response strategy based on dialogue state and intent."""
        state = dialogue_ctx.state
        
        # Define response strategies based on state
        if state == DialogueState.GREETING:
            return "greeting"
        elif state == DialogueState.INFORMATION_GATHERING:
            if dialogue_ctx.slots:
                missing_slots = [s for s in dialogue_ctx.slots.values() if not s.value]
                if missing_slots:
                    return f"request_information_{missing_slots[0].name}"
            return "information_gathering"
        elif state == DialogueState.CLARIFICATION:
            return "clarification"
        elif state == DialogueState.CONFIRMATION:
            return "confirmation"
        elif state == DialogueState.TASK_EXECUTION:
            return "task_execution"
        elif state == DialogueState.FAREWELL:
            return "farewell"
        elif state == DialogueState.ERROR_HANDLING:
            return "error_handling"
        elif state == DialogueState.ESCALATION:
            return "escalation"
        
        return "default"
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired dialogue sessions."""
        try:
            current_time = datetime.utcnow()
            expired_sessions = []
            
            for session_id, dialogue_ctx in self.dialogue_contexts.items():
                session_age = current_time - dialogue_ctx.session_start_time
                if session_age > timedelta(minutes=self.session_timeout_minutes):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.dialogue_contexts[session_id]
                secure_logger.info(f"Expired dialogue session removed: {session_id}")
            
        except Exception as e:
            secure_logger.error(f"Error during session cleanup: {str(e)}")
    
    def _update_metrics(self, dialogue_ctx: DialogueContext, 
                       previous_state: DialogueState, next_state: DialogueState) -> None:
        """Update performance metrics."""
        self.metrics['total_turns'] += 1
        
        if next_state == DialogueState.FAREWELL and previous_state != DialogueState.ERROR_HANDLING:
            self.metrics['successful_completions'] += 1
        elif next_state == DialogueState.ESCALATION:
            self.metrics['escalations'] += 1
        
        # Update average session length
        total_sessions = self.metrics['sessions_managed']
        if total_sessions > 0:
            self.metrics['average_session_length'] = self.metrics['total_turns'] / total_sessions
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of dialogue session."""
        if session_id not in self.dialogue_contexts:
            return None
        
        dialogue_ctx = self.dialogue_contexts[session_id]
        session_duration = datetime.utcnow() - dialogue_ctx.session_start_time
        
        return {
            'session_id': session_id,
            'current_state': dialogue_ctx.state.value,
            'turn_count': dialogue_ctx.turn_count,
            'session_duration_minutes': session_duration.total_seconds() / 60,
            'slots_collected': len([s for s in dialogue_ctx.slots.values() if s.value]),
            'last_intent': dialogue_ctx.intent.type.value if dialogue_ctx.intent else None,
            'collected_information': dialogue_ctx.collected_information,
            'pending_confirmations': dialogue_ctx.pending_confirmations
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get dialogue manager performance metrics."""
        active_sessions = len(self.dialogue_contexts)
        completion_rate = (
            self.metrics['successful_completions'] / self.metrics['sessions_managed']
            if self.metrics['sessions_managed'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'active_sessions': active_sessions,
            'completion_rate': completion_rate,
            'escalation_rate': (
                self.metrics['escalations'] / self.metrics['sessions_managed']
                if self.metrics['sessions_managed'] > 0 else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the dialogue manager."""
        try:
            # Test intent recognition
            test_message = ChatMessage(
                content="Hello, I need help with booking",
                user_id="test_user",
                session_id="test_session"
            )
            
            test_context = ConversationContext(
                session_id="test_session",
                user_id="test_user"
            )
            
            result = await self.process_message(test_message, test_context)
            
            # Cleanup test session
            if "test_session" in self.dialogue_contexts:
                del self.dialogue_contexts["test_session"]
            
            return {
                'status': 'healthy',
                'test_successful': 'response_strategy' in result,
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.get_metrics()
            }


# Export the main classes
__all__ = ['DialogueManager', 'DialogueState', 'IntentType', 'DialogueContext', 'Intent', 'Slot']
