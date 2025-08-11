"""
Enterprise Query Disambiguator for Chatbot Framework

This module provides intelligent query disambiguation with:
- Multi-interpretation detection and ranking
- Context-aware clarification
- Dynamic choice presentation
- Learning from user selections
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
import hashlib

from .base_core import IDisambiguator, secure_logger


class AmbiguityType(Enum):
    """Types of ambiguity in user queries."""
    LEXICAL = "lexical"  # Word has multiple meanings
    SYNTACTIC = "syntactic"  # Sentence structure unclear
    SEMANTIC = "semantic"  # Intent unclear
    REFERENCE = "reference"  # Pronoun/reference unclear
    SCOPE = "scope"  # Unclear what the query applies to


@dataclass
class Interpretation:
    """Possible interpretation of an ambiguous query."""
    id: str
    text: str
    confidence: float
    intent: str
    parameters: Dict[str, Any]
    clarification_needed: bool
    supporting_evidence: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'confidence': self.confidence,
            'intent': self.intent,
            'parameters': self.parameters,
            'clarification_needed': self.clarification_needed,
            'supporting_evidence': self.supporting_evidence
        }


@dataclass
class DisambiguationResult:
    """Result of disambiguation analysis."""
    is_ambiguous: bool
    ambiguity_type: Optional[AmbiguityType]
    interpretations: List[Interpretation]
    confidence_spread: float  # Difference between top two interpretations
    recommended_action: str
    clarification_questions: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_ambiguous': self.is_ambiguous,
            'ambiguity_type': self.ambiguity_type.value if self.ambiguity_type else None,
            'interpretations': [i.to_dict() for i in self.interpretations],
            'confidence_spread': self.confidence_spread,
            'recommended_action': self.recommended_action,
            'clarification_questions': self.clarification_questions,
            'timestamp': self.timestamp.isoformat()
        }


class AmbiguityDetector:
    """Detects various types of ambiguity in user queries."""
    
    def __init__(self):
        # Ambiguous words with multiple meanings
        self.ambiguous_terms = {
            'book': ['reservation', 'literature', 'record'],
            'bank': ['financial_institution', 'river_side', 'data_storage'],
            'order': ['command', 'purchase', 'sequence'],
            'date': ['calendar_date', 'romantic_meeting', 'fruit'],
            'account': ['user_account', 'financial_account', 'report'],
            'address': ['location', 'speak_to', 'handle'],
            'charge': ['battery_charge', 'financial_charge', 'accusation'],
            'check': ['verify', 'payment_method', 'examine'],
            'file': ['document', 'tool', 'submit'],
            'park': ['recreational_area', 'vehicle_parking'],
            'match': ['sports_game', 'comparison', 'fire_starter'],
            'track': ['monitor', 'path', 'music_song'],
            'course': ['educational_class', 'path_direction', 'meal_course'],
            'spring': ['season', 'water_source', 'mechanical_component'],
            'fall': ['autumn_season', 'drop_down', 'failure']
        }
        
        # Pronouns that can cause reference ambiguity
        self.ambiguous_pronouns = ['it', 'this', 'that', 'them', 'they', 'those', 'these']
        
        # Words indicating uncertainty or multiple options
        self.uncertainty_indicators = [
            'maybe', 'perhaps', 'possibly', 'either', 'or', 'might',
            'could be', 'not sure', 'think', 'probably', 'any of'
        ]
    
    def detect_lexical_ambiguity(self, text: str) -> List[Tuple[str, List[str]]]:
        """Detect words with multiple meanings."""
        words = re.findall(r'\b\w+\b', text.lower())
        ambiguous_words = []
        
        for word in words:
            if word in self.ambiguous_terms:
                ambiguous_words.append((word, self.ambiguous_terms[word]))
        
        return ambiguous_words
    
    def detect_reference_ambiguity(self, text: str, context_history: List[str]) -> List[str]:
        """Detect unclear pronoun references."""
        text_lower = text.lower()
        ambiguous_refs = []
        
        for pronoun in self.ambiguous_pronouns:
            if pronoun in text_lower:
                # Check if there are multiple possible referents in context
                if len(context_history) > 0:
                    # Simple heuristic: if there are multiple nouns in recent context
                    recent_context = ' '.join(context_history[-3:]).lower()
                    nouns = re.findall(r'\b[a-z]+(?:ing|tion|ness|ment|er|or)\b', recent_context)
                    if len(set(nouns)) > 1:
                        ambiguous_refs.append(pronoun)
        
        return ambiguous_refs
    
    def detect_semantic_ambiguity(self, text: str) -> bool:
        """Detect unclear intent or semantic meaning."""
        text_lower = text.lower()
        
        # Check for uncertainty indicators
        for indicator in self.uncertainty_indicators:
            if indicator in text_lower:
                return True
        
        # Check for multiple possible actions
        action_words = re.findall(r'\b(book|cancel|check|update|delete|modify|change)\b', text_lower)
        if len(set(action_words)) > 1:
            return True
        
        # Check for question words without clear structure
        question_words = ['what', 'where', 'when', 'why', 'how', 'which', 'who']
        question_count = sum(1 for qw in question_words if qw in text_lower)
        if question_count > 1:
            return True
        
        return False


class InterpretationGenerator:
    """Generates possible interpretations for ambiguous queries."""
    
    def __init__(self):
        self.interpretation_templates = {
            'book': [
                {'intent': 'make_reservation', 'template': 'Make a reservation for {entity}'},
                {'intent': 'find_literature', 'template': 'Find books about {entity}'},
                {'intent': 'record_entry', 'template': 'Record an entry for {entity}'}
            ],
            'order': [
                {'intent': 'make_purchase', 'template': 'Place an order for {entity}'},
                {'intent': 'sort_items', 'template': 'Sort/order {entity} by criteria'},
                {'intent': 'give_command', 'template': 'Command or instruct about {entity}'}
            ],
            'check': [
                {'intent': 'verify_status', 'template': 'Check the status of {entity}'},
                {'intent': 'payment_method', 'template': 'Use check as payment for {entity}'},
                {'intent': 'examine_item', 'template': 'Examine or inspect {entity}'}
            ]
        }
    
    def generate_interpretations(self, text: str, ambiguous_words: List[Tuple[str, List[str]]],
                               context: Dict[str, Any]) -> List[Interpretation]:
        """Generate possible interpretations for ambiguous text."""
        interpretations = []
        
        for word, meanings in ambiguous_words:
            if word in self.interpretation_templates:
                templates = self.interpretation_templates[word]
                
                for i, template_info in enumerate(templates):
                    # Extract entity from context or text
                    entity = self._extract_entity(text, word)
                    
                    interpretation = Interpretation(
                        id=f"{word}_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}",
                        text=template_info['template'].format(entity=entity or '[entity]'),
                        confidence=0.7 / (i + 1),  # Decreasing confidence
                        intent=template_info['intent'],
                        parameters={'ambiguous_word': word, 'entity': entity},
                        clarification_needed=True,
                        supporting_evidence=[f"Word '{word}' detected"]
                    )
                    interpretations.append(interpretation)
        
        # If no specific templates, generate generic interpretations
        if not interpretations:
            interpretations = self._generate_generic_interpretations(text)
        
        return interpretations
    
    def _extract_entity(self, text: str, ambiguous_word: str) -> Optional[str]:
        """Extract the entity that the ambiguous word refers to."""
        # Simple extraction: look for nouns near the ambiguous word
        words = text.split()
        
        try:
            word_index = words.index(ambiguous_word)
            
            # Look for nouns before and after
            for offset in [1, -1, 2, -2]:
                check_index = word_index + offset
                if 0 <= check_index < len(words):
                    candidate = words[check_index]
                    # Simple heuristic: if it's not a common word, might be an entity
                    if len(candidate) > 2 and candidate.lower() not in ['the', 'a', 'an', 'and', 'or', 'for', 'to']:
                        return candidate
        except ValueError:
            pass
        
        return None
    
    def _generate_generic_interpretations(self, text: str) -> List[Interpretation]:
        """Generate generic interpretations when specific patterns don't match."""
        interpretations = []
        
        # Basic intent categories
        intent_patterns = [
            {'pattern': r'\b(find|search|look|show)\b', 'intent': 'information_request'},
            {'pattern': r'\b(book|reserve|schedule)\b', 'intent': 'booking_request'},
            {'pattern': r'\b(cancel|delete|remove)\b', 'intent': 'cancellation_request'},
            {'pattern': r'\b(update|change|modify)\b', 'intent': 'modification_request'},
            {'pattern': r'\b(help|assist|support)\b', 'intent': 'help_request'}
        ]
        
        for i, pattern_info in enumerate(intent_patterns):
            if re.search(pattern_info['pattern'], text.lower()):
                interpretation = Interpretation(
                    id=f"generic_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}",
                    text=f"Process as {pattern_info['intent'].replace('_', ' ')}",
                    confidence=0.5 - (i * 0.1),
                    intent=pattern_info['intent'],
                    parameters={'pattern_matched': pattern_info['pattern']},
                    clarification_needed=True,
                    supporting_evidence=[f"Pattern matched: {pattern_info['pattern']}"]
                )
                interpretations.append(interpretation)
        
        return interpretations


class Disambiguator(IDisambiguator):
    """Enterprise disambiguator with multi-strategy analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the disambiguator."""
        self.config = config or {}
        self.ambiguity_detector = AmbiguityDetector()
        self.interpretation_generator = InterpretationGenerator()
        
        # Configuration
        self.ambiguity_threshold = self.config.get('ambiguity_threshold', 0.3)
        self.max_interpretations = self.config.get('max_interpretations', 5)
        self.confidence_spread_threshold = self.config.get('confidence_spread_threshold', 0.2)
        
        # Learning data for improving disambiguation
        self.disambiguation_history = []
        self.user_choice_patterns = {}
        
        # Performance metrics
        self.metrics = {
            'queries_analyzed': 0,
            'ambiguous_queries': 0,
            'clarifications_requested': 0,
            'successful_disambiguations': 0,
            'total_processing_time': 0.0
        }
        
        secure_logger.info("Disambiguator initialized", extra={
            'ambiguity_threshold': self.ambiguity_threshold,
            'max_interpretations': self.max_interpretations
        })
    
    async def disambiguate(self, query: str, context: Optional[Dict[str, Any]] = None) -> DisambiguationResult:
        """
        Analyze query for ambiguity and provide disambiguation options.
        
        Args:
            query: User query to analyze
            context: Optional context including conversation history
            
        Returns:
            Disambiguation analysis result
        """
        start_time = asyncio.get_event_loop().time()
        
        if not query or not query.strip():
            return DisambiguationResult(
                is_ambiguous=False,
                ambiguity_type=None,
                interpretations=[],
                confidence_spread=0.0,
                recommended_action="request_input",
                clarification_questions=[],
                timestamp=datetime.utcnow()
            )
        
        try:
            context = context or {}
            conversation_history = context.get('conversation_history', [])
            
            # Detect different types of ambiguity
            lexical_ambiguity = self.ambiguity_detector.detect_lexical_ambiguity(query)
            reference_ambiguity = self.ambiguity_detector.detect_reference_ambiguity(
                query, conversation_history
            )
            semantic_ambiguity = self.ambiguity_detector.detect_semantic_ambiguity(query)
            
            # Determine primary ambiguity type
            ambiguity_type = self._determine_primary_ambiguity_type(
                lexical_ambiguity, reference_ambiguity, semantic_ambiguity
            )
            
            # Generate interpretations if ambiguous
            interpretations = []
            is_ambiguous = False
            
            if ambiguity_type:
                is_ambiguous = True
                interpretations = self.interpretation_generator.generate_interpretations(
                    query, lexical_ambiguity, context
                )
                
                # Apply learning from previous user choices
                interpretations = self._apply_learned_preferences(interpretations, context)
                
                # Limit number of interpretations
                interpretations = interpretations[:self.max_interpretations]
            
            # Calculate confidence spread
            confidence_spread = self._calculate_confidence_spread(interpretations)
            
            # Determine recommended action
            recommended_action = self._determine_recommended_action(
                is_ambiguous, confidence_spread, interpretations
            )
            
            # Generate clarification questions
            clarification_questions = self._generate_clarification_questions(
                ambiguity_type, lexical_ambiguity, interpretations
            )
            
            # Create result
            result = DisambiguationResult(
                is_ambiguous=is_ambiguous,
                ambiguity_type=ambiguity_type,
                interpretations=interpretations,
                confidence_spread=confidence_spread,
                recommended_action=recommended_action,
                clarification_questions=clarification_questions,
                timestamp=datetime.utcnow()
            )
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(result, processing_time)
            
            # Store for learning
            self.disambiguation_history.append({
                'query': query,
                'result': result.to_dict(),
                'timestamp': datetime.utcnow()
            })
            
            secure_logger.info("Query disambiguation completed", extra={
                'query_length': len(query),
                'is_ambiguous': is_ambiguous,
                'ambiguity_type': ambiguity_type.value if ambiguity_type else None,
                'interpretations_count': len(interpretations),
                'confidence_spread': confidence_spread,
                'processing_time': processing_time
            })
            
            return result
            
        except Exception as e:
            secure_logger.error(f"Error during disambiguation: {str(e)}", extra={
                'query': query[:100],  # Log first 100 chars
                'error_type': type(e).__name__
            })
            
            # Return safe fallback
            return DisambiguationResult(
                is_ambiguous=False,
                ambiguity_type=None,
                interpretations=[],
                confidence_spread=0.0,
                recommended_action="fallback",
                clarification_questions=["I'm having trouble understanding. Could you rephrase that?"],
                timestamp=datetime.utcnow()
            )
    
    def _determine_primary_ambiguity_type(self, lexical_ambiguity: List[Tuple[str, List[str]]],
                                        reference_ambiguity: List[str],
                                        semantic_ambiguity: bool) -> Optional[AmbiguityType]:
        """Determine the primary type of ambiguity."""
        if lexical_ambiguity:
            return AmbiguityType.LEXICAL
        elif reference_ambiguity:
            return AmbiguityType.REFERENCE
        elif semantic_ambiguity:
            return AmbiguityType.SEMANTIC
        
        return None
    
    def _calculate_confidence_spread(self, interpretations: List[Interpretation]) -> float:
        """Calculate the confidence spread between interpretations."""
        if len(interpretations) < 2:
            return 1.0  # No ambiguity if only one interpretation
        
        confidences = [i.confidence for i in interpretations]
        confidences.sort(reverse=True)
        
        return confidences[0] - confidences[1]
    
    def _determine_recommended_action(self, is_ambiguous: bool, confidence_spread: float,
                                    interpretations: List[Interpretation]) -> str:
        """Determine the recommended action based on analysis."""
        if not is_ambiguous:
            return "proceed"
        
        if confidence_spread > self.confidence_spread_threshold:
            return "proceed_with_best"  # High confidence in top interpretation
        elif len(interpretations) <= 2:
            return "request_clarification"  # Ask user to choose
        else:
            return "present_options"  # Show multiple options
    
    def _generate_clarification_questions(self, ambiguity_type: Optional[AmbiguityType],
                                        lexical_ambiguity: List[Tuple[str, List[str]]],
                                        interpretations: List[Interpretation]) -> List[str]:
        """Generate clarification questions based on ambiguity type."""
        questions = []
        
        if ambiguity_type == AmbiguityType.LEXICAL and lexical_ambiguity:
            word, meanings = lexical_ambiguity[0]
            questions.append(f"When you say '{word}', do you mean:")
            
        elif ambiguity_type == AmbiguityType.REFERENCE:
            questions.append("Could you clarify what you're referring to?")
            
        elif ambiguity_type == AmbiguityType.SEMANTIC:
            questions.append("I see a few ways to interpret that. Did you mean:")
            
        # Add interpretation-specific questions
        if interpretations and len(interpretations) > 1:
            for i, interpretation in enumerate(interpretations[:3]):  # Top 3
                questions.append(f"{i+1}. {interpretation.text}")
        
        if not questions:
            questions.append("Could you provide more details about what you're looking for?")
        
        return questions
    
    def _apply_learned_preferences(self, interpretations: List[Interpretation],
                                 context: Dict[str, Any]) -> List[Interpretation]:
        """Apply learned user preferences to ranking."""
        user_id = context.get('user_id')
        if not user_id or user_id not in self.user_choice_patterns:
            return interpretations
        
        # Boost interpretations that the user has chosen before
        user_patterns = self.user_choice_patterns[user_id]
        
        for interpretation in interpretations:
            intent = interpretation.intent
            if intent in user_patterns:
                # Boost confidence based on historical choices
                boost = min(0.2, user_patterns[intent] * 0.05)
                interpretation.confidence = min(1.0, interpretation.confidence + boost)
        
        # Re-sort by confidence
        interpretations.sort(key=lambda x: x.confidence, reverse=True)
        
        return interpretations
    
    def _update_metrics(self, result: DisambiguationResult, processing_time: float) -> None:
        """Update performance metrics."""
        self.metrics['queries_analyzed'] += 1
        self.metrics['total_processing_time'] += processing_time
        
        if result.is_ambiguous:
            self.metrics['ambiguous_queries'] += 1
            
        if result.recommended_action in ['request_clarification', 'present_options']:
            self.metrics['clarifications_requested'] += 1
    
    async def record_user_choice(self, query: str, chosen_interpretation_id: str,
                               user_id: str) -> None:
        """Record user's disambiguation choice for learning."""
        try:
            # Find the interpretation that was chosen
            for history_entry in reversed(self.disambiguation_history):
                if history_entry['query'] == query:
                    interpretations = history_entry['result']['interpretations']
                    for interpretation in interpretations:
                        if interpretation['id'] == chosen_interpretation_id:
                            intent = interpretation['intent']
                            
                            # Update user choice patterns
                            if user_id not in self.user_choice_patterns:
                                self.user_choice_patterns[user_id] = {}
                            
                            self.user_choice_patterns[user_id][intent] = (
                                self.user_choice_patterns[user_id].get(intent, 0) + 1
                            )
                            
                            self.metrics['successful_disambiguations'] += 1
                            
                            secure_logger.info("User disambiguation choice recorded", extra={
                                'user_id': user_id,
                                'chosen_intent': intent,
                                'query': query[:50]
                            })
                            return
            
        except Exception as e:
            secure_logger.error(f"Error recording user choice: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get disambiguator performance metrics."""
        return {
            **self.metrics,
            'average_processing_time': (
                self.metrics['total_processing_time'] / self.metrics['queries_analyzed']
                if self.metrics['queries_analyzed'] > 0 else 0
            ),
            'ambiguity_rate': (
                self.metrics['ambiguous_queries'] / self.metrics['queries_analyzed']
                if self.metrics['queries_analyzed'] > 0 else 0
            ),
            'clarification_rate': (
                self.metrics['clarifications_requested'] / self.metrics['queries_analyzed']
                if self.metrics['queries_analyzed'] > 0 else 0
            ),
            'disambiguation_success_rate': (
                self.metrics['successful_disambiguations'] / self.metrics['clarifications_requested']
                if self.metrics['clarifications_requested'] > 0 else 0
            ),
            'active_user_patterns': len(self.user_choice_patterns)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the disambiguator."""
        try:
            # Test with an ambiguous query
            test_query = "I want to book something"
            test_context = {'user_id': 'test_user'}
            
            result = await self.disambiguate(test_query, test_context)
            
            return {
                'status': 'healthy',
                'test_successful': result is not None,
                'test_ambiguous': result.is_ambiguous,
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.get_metrics()
            }


# Export the main classes
__all__ = ['Disambiguator', 'DisambiguationResult', 'Interpretation', 'AmbiguityType']
