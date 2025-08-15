"""
Simple Intent Recognition
========================

Basic intent recognition for chatbot messages.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Intent confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class IntentResult:
    """Intent recognition result."""
    intent: str
    confidence: float
    confidence_level: ConfidenceLevel
    entities: Dict[str, str]
    
    def __post_init__(self):
        if self.confidence >= 0.8:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW


class SimpleIntentRecognizer:
    """Simple rule-based intent recognizer."""
    
    def __init__(self):
        """Initialize with basic intent patterns."""
        self.intent_patterns = {
            'greeting': [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\bhow are you\b',
                r'\bwhat\'s up\b'
            ],
            'goodbye': [
                r'\b(bye|goodbye|see you|farewell|talk to you later)\b',
                r'\bhave a good day\b',
                r'\bthanks and goodbye\b'
            ],
            'question': [
                r'\b(what|how|when|where|why|which|who)\b.*\?',
                r'\bcan you tell me\b',
                r'\bi want to know\b'
            ],
            'help': [
                r'\b(help|assist|support)\b',
                r'\bi need help\b',
                r'\bcan you help\b'
            ],
            'booking': [
                r'\b(book|reserve|schedule|appointment)\b',
                r'\bi want to book\b',
                r'\bmake a reservation\b'
            ],
            'complaint': [
                r'\b(problem|issue|complaint|wrong|error)\b',
                r'\bi\'m not happy\b',
                r'\bthis is not working\b'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'time': r'\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'number': r'\b\d+\b'
        }
    
    async def recognize_intent(self, text: str) -> IntentResult:
        """
        Recognize intent from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            IntentResult with intent and confidence
        """
        text_lower = text.lower()
        best_intent = 'unknown'
        best_confidence = 0.0
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            intent_score = 0.0
            pattern_count = len(patterns)
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    intent_score += 1.0
            
            # Calculate confidence as percentage of matching patterns
            confidence = intent_score / pattern_count if pattern_count > 0 else 0.0
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent
        
        # Extract entities
        entities = self._extract_entities(text)
        
        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            confidence_level=ConfidenceLevel.LOW,  # Will be set in __post_init__
            entities=entities
        )
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities from text."""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0]  # Take first match
        
        return entities
    
    def add_intent_pattern(self, intent: str, pattern: str):
        """Add a new intent pattern."""
        if intent not in self.intent_patterns:
            self.intent_patterns[intent] = []
        self.intent_patterns[intent].append(pattern)
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported intents."""
        return list(self.intent_patterns.keys())
