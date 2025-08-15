"""
Simple Conversation Summarizer
=============================

Basic text summarization for conversations.
"""

from typing import List, Any
import re


class ConversationSummarizer:
    """Simple conversation summarizer."""
    
    def __init__(self, max_summary_length: int = 500):
        """Initialize summarizer."""
        self.max_summary_length = max_summary_length
    
    async def summarize_text(self, text: str) -> str:
        """Summarize text."""
        if not text or len(text) <= self.max_summary_length:
            return text
        
        # Simple truncation for now
        return text[:self.max_summary_length] + "..."
    
    async def summarize_conversation(self, messages: List[Any]) -> str:
        """Summarize a conversation."""
        if not messages:
            return "No conversation to summarize."
        
        # Convert messages to text
        text_lines = []
        for msg in messages:
            if hasattr(msg, 'text'):
                text_lines.append(msg.text)
            else:
                text_lines.append(str(msg))
        
        full_text = " ".join(text_lines)
        return await self.summarize_text(full_text)
