"""
Enterprise Sentiment Analysis for Chatbot Framework

This module provides comprehensive sentiment analysis capabilities with:
- Multi-dimensional sentiment analysis (emotion, polarity, confidence)
- Real-time sentiment tracking and trends
- Configurable sensitivity and thresholds
- Performance optimization and caching
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import re
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

from .base_core import ISentimentAnalyzer, SentimentType, secure_logger


class EmotionType(Enum):
    """Extended emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result."""
    sentiment: SentimentType
    confidence: float
    polarity_score: float  # -1.0 to 1.0
    emotion: EmotionType
    emotion_confidence: float
    keywords: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sentiment': self.sentiment.value,
            'confidence': self.confidence,
            'polarity_score': self.polarity_score,
            'emotion': self.emotion.value,
            'emotion_confidence': self.emotion_confidence,
            'keywords': self.keywords,
            'timestamp': self.timestamp.isoformat()
        }


class SentimentLexicon:
    """Sentiment lexicon with weighted words and phrases."""
    
    def __init__(self):
        # Positive words with weights
        self.positive_words = {
            'excellent': 0.9, 'amazing': 0.8, 'wonderful': 0.8, 'fantastic': 0.8,
            'great': 0.7, 'good': 0.6, 'nice': 0.5, 'okay': 0.3, 'fine': 0.3,
            'love': 0.8, 'like': 0.6, 'enjoy': 0.7, 'appreciate': 0.6,
            'happy': 0.7, 'pleased': 0.6, 'satisfied': 0.6, 'delighted': 0.8,
            'awesome': 0.8, 'brilliant': 0.8, 'perfect': 0.9, 'outstanding': 0.9,
            'superb': 0.8, 'magnificent': 0.8, 'fabulous': 0.8, 'terrific': 0.8,
            'marvelous': 0.8, 'splendid': 0.8, 'remarkable': 0.7, 'impressive': 0.7,
            'positive': 0.6, 'optimistic': 0.6, 'cheerful': 0.7, 'joyful': 0.8
        }
        
        # Negative words with weights
        self.negative_words = {
            'terrible': -0.9, 'awful': -0.8, 'horrible': -0.8, 'disgusting': -0.8,
            'bad': -0.7, 'poor': -0.6, 'disappointing': -0.7, 'unfortunate': -0.6,
            'hate': -0.8, 'dislike': -0.6, 'despise': -0.8, 'loathe': -0.9,
            'angry': -0.7, 'upset': -0.6, 'frustrated': -0.7, 'annoyed': -0.6,
            'sad': -0.6, 'depressed': -0.8, 'unhappy': -0.7, 'miserable': -0.8,
            'worried': -0.6, 'concerned': -0.5, 'anxious': -0.7, 'nervous': -0.6,
            'scared': -0.7, 'afraid': -0.7, 'fearful': -0.7, 'terrified': -0.9,
            'negative': -0.6, 'pessimistic': -0.6, 'gloomy': -0.7, 'tragic': -0.8,
            'disaster': -0.8, 'crisis': -0.7, 'problem': -0.5, 'issue': -0.4,
            'wrong': -0.5, 'error': -0.4, 'mistake': -0.4, 'failure': -0.7
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'incredibly': 1.4, 'absolutely': 1.4,
            'totally': 1.3, 'completely': 1.3, 'really': 1.2, 'quite': 1.1,
            'rather': 1.1, 'pretty': 1.1, 'somewhat': 0.8, 'slightly': 0.7,
            'barely': 0.5, 'hardly': 0.5, 'scarcely': 0.5, 'not': -1.0
        }
        
        # Emotion keywords
        self.emotion_keywords = {
            EmotionType.JOY: ['happy', 'joy', 'excited', 'thrilled', 'elated', 'cheerful', 'delighted'],
            EmotionType.SADNESS: ['sad', 'depressed', 'melancholy', 'grief', 'sorrow', 'blue', 'down'],
            EmotionType.ANGER: ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'frustrated'],
            EmotionType.FEAR: ['scared', 'afraid', 'fearful', 'anxious', 'worried', 'nervous', 'panic'],
            EmotionType.SURPRISE: ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered'],
            EmotionType.DISGUST: ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated', 'appalled'],
            EmotionType.NEUTRAL: ['okay', 'fine', 'alright', 'normal', 'standard', 'average', 'typical']
        }
    
    def get_word_sentiment(self, word: str) -> float:
        """Get sentiment score for a word."""
        word_lower = word.lower()
        if word_lower in self.positive_words:
            return self.positive_words[word_lower]
        elif word_lower in self.negative_words:
            return self.negative_words[word_lower]
        return 0.0
    
    def get_intensifier_weight(self, word: str) -> float:
        """Get intensifier weight for a word."""
        return self.intensifiers.get(word.lower(), 1.0)
    
    def detect_emotion(self, text: str) -> Tuple[EmotionType, float]:
        """Detect primary emotion in text."""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            emotion_scores[emotion] = score
        
        # Find emotion with highest score
        if not any(emotion_scores.values()):
            return EmotionType.NEUTRAL, 0.5
        
        primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        max_score = emotion_scores[primary_emotion]
        
        # Calculate confidence based on keyword matches
        total_keywords = sum(len(keywords) for keywords in self.emotion_keywords.values())
        confidence = min(0.9, max_score / 3)  # Cap at 0.9, normalize by expected matches
        
        return primary_emotion, confidence


class SentimentAnalyzer(ISentimentAnalyzer):
    """Enterprise sentiment analyzer with advanced features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sentiment analyzer."""
        self.config = config or {}
        self.lexicon = SentimentLexicon()
        
        # Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.neutral_threshold = self.config.get('neutral_threshold', 0.1)
        self.enable_emotion_detection = self.config.get('enable_emotion_detection', True)
        
        # Sentiment history for trend analysis
        self.sentiment_history: List[SentimentResult] = []
        self.max_history_size = self.config.get('max_history_size', 100)
        
        # Performance metrics
        self.metrics = {
            'analyses_performed': 0,
            'total_processing_time': 0.0,
            'sentiment_distribution': {s.value: 0 for s in SentimentType},
            'emotion_distribution': {e.value: 0 for e in EmotionType}
        }
        
        # Cache for recent analyses
        self.analysis_cache = {}
        self.max_cache_size = 50
        
        secure_logger.info("SentimentAnalyzer initialized", extra={
            'confidence_threshold': self.confidence_threshold,
            'emotion_detection': self.enable_emotion_detection
        })
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Perform comprehensive sentiment analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detailed sentiment analysis result
        """
        start_time = asyncio.get_event_loop().time()
        
        if not text or not text.strip():
            return SentimentResult(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                polarity_score=0.0,
                emotion=EmotionType.NEUTRAL,
                emotion_confidence=0.0,
                keywords=[],
                timestamp=datetime.utcnow()
            )
        
        try:
            # Check cache first
            text_hash = hash(text.strip().lower())
            if text_hash in self.analysis_cache:
                cached_result = self.analysis_cache[text_hash]
                secure_logger.debug("Sentiment analysis cache hit")
                return cached_result
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Extract sentiment features
            polarity_score, confidence, keywords = self._calculate_sentiment_score(processed_text)
            
            # Determine sentiment category
            sentiment = self._categorize_sentiment(polarity_score, confidence)
            
            # Detect emotion if enabled
            emotion = EmotionType.NEUTRAL
            emotion_confidence = 0.0
            if self.enable_emotion_detection:
                emotion, emotion_confidence = self.lexicon.detect_emotion(text)
            
            # Create result
            result = SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                polarity_score=polarity_score,
                emotion=emotion,
                emotion_confidence=emotion_confidence,
                keywords=keywords,
                timestamp=datetime.utcnow()
            )
            
            # Update metrics and history
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(result, processing_time)
            self._add_to_history(result)
            
            # Cache result
            if len(self.analysis_cache) < self.max_cache_size:
                self.analysis_cache[text_hash] = result
            
            secure_logger.info("Sentiment analysis completed", extra={
                'sentiment': sentiment.value,
                'confidence': confidence,
                'polarity_score': polarity_score,
                'emotion': emotion.value,
                'processing_time': processing_time,
                'text_length': len(text)
            })
            
            return result
            
        except Exception as e:
            secure_logger.error(f"Error during sentiment analysis: {str(e)}", extra={
                'text_length': len(text),
                'error_type': type(e).__name__
            })
            # Return neutral sentiment on error
            return SentimentResult(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                polarity_score=0.0,
                emotion=EmotionType.NEUTRAL,
                emotion_confidence=0.0,
                keywords=[],
                timestamp=datetime.utcnow()
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_sentiment_score(self, text: str) -> Tuple[float, float, List[str]]:
        """Calculate sentiment polarity score and confidence."""
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0, 0.0, []
        
        sentiment_scores = []
        sentiment_keywords = []
        i = 0
        
        while i < len(words):
            word = words[i]
            sentiment_score = self.lexicon.get_word_sentiment(word)
            
            if sentiment_score != 0:
                # Check for intensifiers before this word
                intensifier = 1.0
                if i > 0:
                    prev_word = words[i - 1]
                    intensifier = self.lexicon.get_intensifier_weight(prev_word)
                
                # Apply intensifier
                final_score = sentiment_score * intensifier
                sentiment_scores.append(final_score)
                sentiment_keywords.append(word)
            
            i += 1
        
        if not sentiment_scores:
            return 0.0, 0.0, []
        
        # Calculate overall polarity
        polarity_score = sum(sentiment_scores) / len(sentiment_scores)
        
        # Normalize to [-1, 1] range
        polarity_score = max(-1.0, min(1.0, polarity_score))
        
        # Calculate confidence based on number of sentiment words
        sentiment_word_ratio = len(sentiment_scores) / len(words)
        confidence = min(0.95, sentiment_word_ratio * 2)  # Cap at 0.95
        
        # Adjust confidence based on polarity strength
        polarity_strength = abs(polarity_score)
        confidence = confidence * (0.5 + polarity_strength * 0.5)
        
        return polarity_score, confidence, sentiment_keywords
    
    def _categorize_sentiment(self, polarity_score: float, confidence: float) -> SentimentType:
        """Categorize sentiment based on polarity score and confidence."""
        # If confidence is too low, return neutral
        if confidence < self.confidence_threshold:
            return SentimentType.NEUTRAL
        
        # If polarity is within neutral threshold, return neutral
        if abs(polarity_score) <= self.neutral_threshold:
            return SentimentType.NEUTRAL
        
        # Categorize based on polarity
        if polarity_score > 0:
            return SentimentType.POSITIVE
        else:
            return SentimentType.NEGATIVE
    
    def _update_metrics(self, result: SentimentResult, processing_time: float) -> None:
        """Update performance metrics."""
        self.metrics['analyses_performed'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['sentiment_distribution'][result.sentiment.value] += 1
        self.metrics['emotion_distribution'][result.emotion.value] += 1
    
    def _add_to_history(self, result: SentimentResult) -> None:
        """Add result to sentiment history."""
        self.sentiment_history.append(result)
        
        # Maintain history size limit
        if len(self.sentiment_history) > self.max_history_size:
            self.sentiment_history = self.sentiment_history[-self.max_history_size:]
    
    async def get_sentiment_trend(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get sentiment trend analysis over a time window.
        
        Args:
            time_window_hours: Time window in hours to analyze
            
        Returns:
            Sentiment trend analysis
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Filter recent sentiment results
            recent_results = [
                result for result in self.sentiment_history
                if result.timestamp >= cutoff_time
            ]
            
            if not recent_results:
                return {
                    'trend': 'insufficient_data',
                    'sample_size': 0,
                    'time_window_hours': time_window_hours
                }
            
            # Calculate statistics
            polarity_scores = [r.polarity_score for r in recent_results]
            confidences = [r.confidence for r in recent_results]
            
            # Sentiment distribution
            sentiment_counts = {}
            for sentiment in SentimentType:
                sentiment_counts[sentiment.value] = sum(
                    1 for r in recent_results if r.sentiment == sentiment
                )
            
            # Emotion distribution
            emotion_counts = {}
            for emotion in EmotionType:
                emotion_counts[emotion.value] = sum(
                    1 for r in recent_results if r.emotion == emotion
                )
            
            # Calculate trend direction
            if len(polarity_scores) >= 2:
                # Compare first half vs second half
                mid_point = len(polarity_scores) // 2
                first_half_avg = statistics.mean(polarity_scores[:mid_point])
                second_half_avg = statistics.mean(polarity_scores[mid_point:])
                
                if second_half_avg > first_half_avg + 0.1:
                    trend_direction = 'improving'
                elif second_half_avg < first_half_avg - 0.1:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'stable'
            
            return {
                'trend': trend_direction,
                'sample_size': len(recent_results),
                'time_window_hours': time_window_hours,
                'average_polarity': statistics.mean(polarity_scores),
                'average_confidence': statistics.mean(confidences),
                'polarity_std_dev': statistics.stdev(polarity_scores) if len(polarity_scores) > 1 else 0,
                'sentiment_distribution': sentiment_counts,
                'emotion_distribution': emotion_counts,
                'most_common_sentiment': max(sentiment_counts.keys(), key=lambda k: sentiment_counts[k]),
                'most_common_emotion': max(emotion_counts.keys(), key=lambda k: emotion_counts[k])
            }
            
        except Exception as e:
            secure_logger.error(f"Error calculating sentiment trend: {str(e)}")
            return {
                'trend': 'error',
                'error': str(e),
                'sample_size': len(self.sentiment_history)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sentiment analyzer performance metrics."""
        return {
            **self.metrics,
            'average_processing_time': (
                self.metrics['total_processing_time'] / self.metrics['analyses_performed']
                if self.metrics['analyses_performed'] > 0 else 0
            ),
            'cache_size': len(self.analysis_cache),
            'history_size': len(self.sentiment_history)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the sentiment analyzer."""
        try:
            # Test sentiment analysis with sample texts
            test_cases = [
                "I love this product!",
                "This is terrible and awful.",
                "It's okay, nothing special."
            ]
            
            results = []
            for test_text in test_cases:
                result = await self.analyze_sentiment(test_text)
                results.append(result.sentiment.value)
            
            expected = ['positive', 'negative', 'neutral']
            accuracy = sum(1 for r, e in zip(results, expected) if r == e) / len(expected)
            
            return {
                'status': 'healthy',
                'test_accuracy': accuracy,
                'test_results': dict(zip(test_cases, results)),
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.get_metrics()
            }


# Export the main classes
__all__ = ['SentimentAnalyzer', 'SentimentResult', 'EmotionType']
