"""
Enterprise Text Summarizer for Chatbot Framework

This module provides intelligent text summarization capabilities with:
- Multiple summarization strategies (extractive, abstractive)
- Configurable compression ratios
- Context-aware summarization
- Performance optimization
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import re
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .base_core import ISummarizer, secure_logger


class SummarizationStrategy(Enum):
    """Summarization strategy options."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"


@dataclass
class SummaryConfig:
    """Configuration for summarization."""
    strategy: SummarizationStrategy = SummarizationStrategy.EXTRACTIVE
    max_length: int = 200
    min_length: int = 50
    compression_ratio: float = 0.3
    preserve_key_points: bool = True
    include_sentiment: bool = False


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""
    
    @abstractmethod
    async def summarize_text(self, text: str, config: SummaryConfig) -> str:
        """Summarize the given text."""
        pass


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarization using sentence ranking."""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'would',
            'there', 'we', 'him', 'been', 'has', 'when', 'who', 'oil', 'sit'
        }
    
    async def summarize_text(self, text: str, config: SummaryConfig) -> str:
        """Extract key sentences based on word frequency."""
        if not text.strip():
            return ""
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text
        
        # Calculate sentence scores
        word_freq = self._calculate_word_frequency(text)
        sentence_scores = self._score_sentences(sentences, word_freq)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * config.compression_ratio))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Maintain original order
        selected_indices = sorted([sentences.index(sent) for sent, _ in top_sentences])
        summary = ". ".join([sentences[i] for i in selected_indices])
        
        return self._ensure_length_constraints(summary, config)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, int]:
        """Calculate word frequency, excluding stop words."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq
    
    def _score_sentences(self, sentences: List[str], word_freq: Dict[str, int]) -> Dict[str, float]:
        """Score sentences based on word frequency."""
        sentence_scores = {}
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
            else:
                sentence_scores[sentence] = 0
        
        return sentence_scores
    
    def _ensure_length_constraints(self, summary: str, config: SummaryConfig) -> str:
        """Ensure summary meets length constraints."""
        if len(summary) > config.max_length:
            # Truncate to max length
            summary = summary[:config.max_length].rsplit('.', 1)[0] + '.'
        elif len(summary) < config.min_length:
            # If too short, try to include more content
            summary = summary + "..."
        
        return summary


class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarization (simplified implementation)."""
    
    async def summarize_text(self, text: str, config: SummaryConfig) -> str:
        """Generate abstractive summary."""
        # This is a simplified implementation
        # In production, this would use a transformer model
        
        if not text.strip():
            return ""
        
        # Extract key phrases and concepts
        key_phrases = self._extract_key_phrases(text)
        
        # Generate summary based on key phrases
        summary = f"Summary: {', '.join(key_phrases[:5])}"
        
        return self._ensure_length_constraints(summary, config)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple implementation - extract noun phrases
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        phrases = re.findall(r'\b\w+\s+\w+\b', text)
        
        return list(set(words + phrases))[:10]
    
    def _ensure_length_constraints(self, summary: str, config: SummaryConfig) -> str:
        """Ensure summary meets length constraints."""
        if len(summary) > config.max_length:
            summary = summary[:config.max_length] + "..."
        
        return summary


class Summarizer(ISummarizer):
    """Enterprise summarizer with multiple strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the summarizer."""
        self.config = config or {}
        self.default_config = SummaryConfig()
        
        # Initialize strategy implementations
        self.extractive_summarizer = ExtractiveSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer()
        
        # Performance metrics
        self.metrics = {
            'summaries_generated': 0,
            'total_processing_time': 0.0,
            'average_compression_ratio': 0.0
        }
        
        secure_logger.info("Summarizer initialized", extra={
            'default_strategy': self.default_config.strategy.value,
            'max_length': self.default_config.max_length
        })
    
    async def summarize(self, text: str, config: Optional[SummaryConfig] = None) -> str:
        """
        Summarize the given text.
        
        Args:
            text: Text to summarize
            config: Optional summarization configuration
            
        Returns:
            Summarized text
        """
        start_time = asyncio.get_event_loop().time()
        
        if not text or not text.strip():
            return ""
        
        try:
            # Use provided config or default
            summary_config = config or self.default_config
            
            # Choose summarizer based on strategy
            if summary_config.strategy == SummarizationStrategy.EXTRACTIVE:
                summarizer = self.extractive_summarizer
            elif summary_config.strategy == SummarizationStrategy.ABSTRACTIVE:
                summarizer = self.abstractive_summarizer
            else:  # HYBRID
                # For hybrid, use extractive as primary
                summarizer = self.extractive_summarizer
            
            # Generate summary
            summary = await summarizer.summarize_text(text, summary_config)
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(text, summary, processing_time)
            
            secure_logger.info("Text summarized successfully", extra={
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(text) if text else 0,
                'processing_time': processing_time,
                'strategy': summary_config.strategy.value
            })
            
            return summary
            
        except Exception as e:
            secure_logger.error(f"Error during summarization: {str(e)}", extra={
                'text_length': len(text),
                'error_type': type(e).__name__
            })
            # Return truncated text as fallback
            return text[:self.default_config.max_length] + "..." if len(text) > self.default_config.max_length else text
    
    def _update_metrics(self, original: str, summary: str, processing_time: float) -> None:
        """Update performance metrics."""
        self.metrics['summaries_generated'] += 1
        self.metrics['total_processing_time'] += processing_time
        
        compression_ratio = len(summary) / len(original) if original else 0
        total_summaries = self.metrics['summaries_generated']
        current_avg = self.metrics['average_compression_ratio']
        
        # Update rolling average
        self.metrics['average_compression_ratio'] = (
            (current_avg * (total_summaries - 1) + compression_ratio) / total_summaries
        )
    
    async def summarize_conversation(self, messages: List[Dict[str, Any]], 
                                   config: Optional[SummaryConfig] = None) -> str:
        """
        Summarize a conversation from multiple messages.
        
        Args:
            messages: List of message dictionaries
            config: Optional summarization configuration
            
        Returns:
            Conversation summary
        """
        if not messages:
            return ""
        
        try:
            # Combine messages into single text
            conversation_text = "\n".join([
                f"{msg.get('sender', 'User')}: {msg.get('text', '')}"
                for msg in messages
                if msg.get('text')
            ])
            
            # Create conversation-specific config
            conv_config = config or SummaryConfig(
                max_length=300,  # Longer for conversations
                compression_ratio=0.2,  # More aggressive compression
                preserve_key_points=True
            )
            
            summary = await self.summarize(conversation_text, conv_config)
            
            secure_logger.info("Conversation summarized", extra={
                'message_count': len(messages),
                'summary_length': len(summary)
            })
            
            return summary
            
        except Exception as e:
            secure_logger.error(f"Error summarizing conversation: {str(e)}")
            return "Unable to generate conversation summary."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get summarizer performance metrics."""
        return {
            **self.metrics,
            'average_processing_time': (
                self.metrics['total_processing_time'] / self.metrics['summaries_generated']
                if self.metrics['summaries_generated'] > 0 else 0
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the summarizer."""
        try:
            # Test summarization with sample text
            test_text = "This is a test sentence. This is another test sentence. Testing summarization capabilities."
            test_summary = await self.summarize(test_text)
            
            return {
                'status': 'healthy',
                'test_successful': bool(test_summary),
                'metrics': self.get_metrics()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.get_metrics()
            }


# Export the main class
__all__ = ['Summarizer', 'SummaryConfig', 'SummarizationStrategy']
