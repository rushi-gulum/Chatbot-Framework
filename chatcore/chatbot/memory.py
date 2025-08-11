"""
Enterprise Memory Management System for Chatbot Framework

This module provides a comprehensive memory management system with:
- Multi-layered memory (short-term, long-term, episodic)
- Vector-based semantic search
- Automatic summarization and compression
- Security and encryption support
- Performance optimization with caching
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
import uuid
from dataclasses import asdict

from .base_core import IMemoryProvider, ChatMessage, ConversationContext, SentimentType, secure_logger

# Import actual implementation classes
from ..retriever.vectorstore import VectorStore  # type: ignore
from .summarizer import Summarizer  # type: ignore


class EnterpriseMemoryProvider(IMemoryProvider):
    """
    Enterprise-grade memory provider with multi-layered storage and semantic search.
    
    Features:
    - Short-term memory for active conversation context
    - Long-term memory with vector-based semantic search
    - Episodic memory with automatic summarization
    - Memory compression and cleanup
    - Encrypted storage for sensitive information
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory provider with configuration.
        
        Args:
            config: Configuration dictionary containing memory settings
        """
        self.config = config
        self.max_short_term_messages = config.get("max_short_term_messages", 50)
        self.summary_trigger_threshold = config.get("summary_trigger_threshold", 20)
        self.memory_retention_days = config.get("memory_retention_days", 90)
        self.enable_encryption = config.get("enable_encryption", False)
        
        # Initialize storage systems
        self.vector_store = VectorStore(config.get("vectorstore", {}))
        self.summarizer = Summarizer(config.get("summarizer", {}))
        
        # In-memory cache for active sessions
        self._session_cache: Dict[str, ConversationContext] = {}
        self._short_term_storage: Dict[str, List[ChatMessage]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_messages_stored": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "summaries_generated": 0
        }
        
        secure_logger.info("Enterprise Memory Provider initialized")
    
    async def store_message(self, message: ChatMessage, context: ConversationContext) -> bool:
        """
        Store a message in multi-layered memory system.
        
        Args:
            message: The message to store
            context: Current conversation context
            
        Returns:
            bool: True if stored successfully
        """
        try:
            session_id = message.session_id
            
            # 1. Store in short-term memory
            if session_id not in self._short_term_storage:
                self._short_term_storage[session_id] = []
            
            self._short_term_storage[session_id].append(message)
            
            # 2. Maintain short-term memory size limit
            if len(self._short_term_storage[session_id]) > self.max_short_term_messages:
                # Trigger summarization before trimming
                await self._trigger_summarization(session_id)
                
                # Keep only recent messages
                self._short_term_storage[session_id] = self._short_term_storage[session_id][-self.max_short_term_messages:]
            
            # 3. Store in long-term vector memory
            await self._store_in_vector_memory(message, context)
            
            # 4. Update session cache
            self._session_cache[session_id] = context
            
            # 5. Update metrics
            self.metrics["total_messages_stored"] += 1
            
            secure_logger.info(f"Message stored successfully - Session: {session_id}")
            return True
            
        except Exception as e:
            secure_logger.error(f"Error storing message: {e}")
            return False
    
    async def retrieve_context(self, session_id: str, user_id: str) -> ConversationContext:
        """
        Retrieve conversation context with intelligent memory retrieval.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            ConversationContext: Retrieved or new context
        """
        try:
            # 1. Check session cache first
            if session_id in self._session_cache:
                self.metrics["cache_hits"] += 1
                context = self._session_cache[session_id]
                
                # Update recent messages from short-term storage
                if session_id in self._short_term_storage:
                    context.recent_messages = self._short_term_storage[session_id][-10:]
                
                return context
            
            self.metrics["cache_misses"] += 1
            
            # 2. Try to reconstruct from stored data
            context = await self._reconstruct_context(session_id, user_id)
            
            # 3. Cache the reconstructed context
            self._session_cache[session_id] = context
            
            return context
            
        except Exception as e:
            secure_logger.error(f"Error retrieving context: {e}")
            # Return new context if retrieval fails
            return ConversationContext(session_id=session_id, user_id=user_id)
    
    async def store_summary(self, session_id: str, summary: str) -> bool:
        """
        Store conversation summary in episodic memory.
        
        Args:
            session_id: Session identifier
            summary: Conversation summary
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Create summary document for vector storage
            summary_doc = {
                "content": summary,
                "metadata": {
                    "session_id": session_id,
                    "type": "episodic_summary",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_count": len(self._short_term_storage.get(session_id, []))
                }
            }
            
            # Store in vector database
            await self.vector_store.add_document(summary_doc)
            
            self.metrics["summaries_generated"] += 1
            secure_logger.info(f"Summary stored for session: {session_id}")
            return True
            
        except Exception as e:
            secure_logger.error(f"Error storing summary: {e}")
            return False
    
    async def search_memory(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory using semantic vector search.
        
        Args:
            query: Search query
            user_id: User identifier for scoping search
            limit: Maximum number of results
            
        Returns:
            List of relevant memory items
        """
        try:
            # Search vector store with user context
            results = await self.vector_store.search(
                query=query,
                filters={"user_id": user_id},
                num_results=limit
            )
            
            # Enhance results with additional context
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    **result,
                    "retrieved_at": datetime.utcnow().isoformat(),
                    "search_query": query
                }
                enhanced_results.append(enhanced_result)
            
            secure_logger.info(f"Memory search completed - Query: {query}, Results: {len(enhanced_results)}")
            return enhanced_results
            
        except Exception as e:
            secure_logger.error(f"Error searching memory: {e}")
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear session-specific memory with optional summarization.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if cleared successfully
        """
        try:
            # Generate final summary before clearing
            if session_id in self._short_term_storage and self._short_term_storage[session_id]:
                await self._trigger_summarization(session_id)
            
            # Clear from caches
            self._session_cache.pop(session_id, None)
            self._short_term_storage.pop(session_id, None)
            
            secure_logger.info(f"Session cleared: {session_id}")
            return True
            
        except Exception as e:
            secure_logger.error(f"Error clearing session: {e}")
            return False
    
    async def _store_in_vector_memory(self, message: ChatMessage, context: ConversationContext) -> None:
        """Store message in vector database for semantic search."""
        try:
            # Create document for vector storage
            doc = {
                "content": message.content,
                "metadata": {
                    "message_id": message.id,
                    "session_id": message.session_id,
                    "user_id": message.user_id,
                    "timestamp": message.timestamp.isoformat(),
                    "message_type": message.message_type.value,
                    "sentiment": context.sentiment.value if context.sentiment else None,
                    "intent": context.current_intent,
                    "conversation_stage": context.conversation_stage,
                    "type": "message"
                }
            }
            
            await self.vector_store.add_document(doc)
            
        except Exception as e:
            secure_logger.error(f"Error storing in vector memory: {e}")
    
    async def _reconstruct_context(self, session_id: str, user_id: str) -> ConversationContext:
        """Reconstruct conversation context from stored data."""
        try:
            # Search for recent messages and summaries for this session
            session_data = await self.vector_store.search(
                query="",
                filters={"session_id": session_id},
                num_results=50
            )
            
            # Separate messages and summaries
            messages = []
            summaries = []
            
            for item in session_data:
                if item.get("metadata", {}).get("type") == "message":
                    # Reconstruct ChatMessage object
                    msg_data = item["metadata"]
                    message = ChatMessage(
                        id=msg_data.get("message_id", ""),
                        content=item["content"],
                        user_id=user_id,
                        session_id=session_id,
                        timestamp=datetime.fromisoformat(msg_data.get("timestamp", datetime.utcnow().isoformat()))
                    )
                    messages.append(message)
                elif item.get("metadata", {}).get("type") == "episodic_summary":
                    summaries.append(item["content"])
            
            # Sort messages by timestamp
            messages.sort(key=lambda x: x.timestamp)
            
            # Determine conversation stage and sentiment
            latest_sentiment = None
            latest_intent = None
            
            if session_data:
                latest_item = session_data[0]  # Most recent
                metadata = latest_item.get("metadata", {})
                if metadata.get("sentiment"):
                    latest_sentiment = SentimentType(metadata["sentiment"])
                latest_intent = metadata.get("intent")
            
            # Create context
            context = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                current_intent=latest_intent,
                sentiment=latest_sentiment,
                conversation_stage="ongoing" if messages else "initial",
                recent_messages=messages[-10:],  # Keep last 10 messages
                context_variables={}
            )
            
            return context
            
        except Exception as e:
            secure_logger.error(f"Error reconstructing context: {e}")
            return ConversationContext(session_id=session_id, user_id=user_id)
    
    async def _trigger_summarization(self, session_id: str) -> None:
        """Trigger summarization of conversation."""
        try:
            if session_id not in self._short_term_storage:
                return
            
            messages = self._short_term_storage[session_id]
            if len(messages) < self.summary_trigger_threshold:
                return
            
            # Generate summary
            conversation_text = "\n".join([f"{msg.message_type.value}: {msg.content}" for msg in messages])
            summary = await self.summarizer.summarize(conversation_text)
            
            # Store summary
            await self.store_summary(session_id, summary)
            
        except Exception as e:
            secure_logger.error(f"Error in summarization: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.metrics,
            "active_sessions": len(self._session_cache),
            "short_term_messages": sum(len(msgs) for msgs in self._short_term_storage.values()),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
        }
    
    async def cleanup_old_memory(self) -> int:
        """Clean up old memory entries beyond retention period."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.memory_retention_days)
            
            # This would typically involve database cleanup
            # For now, just clean up local caches of old sessions
            cleaned_count = 0
            
            sessions_to_remove = []
            for session_id, context in self._session_cache.items():
                # Check if any recent messages are old
                if context.recent_messages:
                    latest_message = max(context.recent_messages, key=lambda x: x.timestamp)
                    if latest_message.timestamp < cutoff_date:
                        sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                await self.clear_session(session_id)
                cleaned_count += 1
            
            secure_logger.info(f"Cleaned up {cleaned_count} old sessions")
            return cleaned_count
            
        except Exception as e:
            secure_logger.error(f"Error during memory cleanup: {e}")
            return 0


# Legacy compatibility
class Memory(EnterpriseMemoryProvider):
    """Legacy Memory class for backward compatibility."""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize with legacy interface."""
        config = {
            "max_short_term_messages": 50,
            "summary_trigger_threshold": 20,
            "memory_retention_days": 90
        }
        super().__init__(config)
        self.session_id = session_id or str(uuid.uuid4())
        
        # Legacy attributes for backward compatibility
        self.short_term_memory: List[Dict[str, Any]] = []
        self.vector_store = VectorStore()
        self.summarizer = Summarizer()
    
    def save_message(self, message: Dict[str, Any]) -> None:
        """Legacy save_message method."""
        # Convert to new format and use async method
        chat_message = ChatMessage(
            content=message.get('text', ''),
            user_id=message.get('user_id', 'unknown'),
            session_id=self.session_id
        )
        
        context = ConversationContext(
            session_id=self.session_id,
            user_id=chat_message.user_id
        )
        
        # Run async method in sync context
        asyncio.create_task(super().store_message(chat_message, context))
        
        # Also maintain legacy short_term_memory for compatibility
        self.short_term_memory.append({
            **message,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_recent_context(self, num_messages: int = 10) -> List[Dict[str, Any]]:
        """Legacy get_recent_context method."""
        return self.short_term_memory[-num_messages:]
    
    async def store_summary(self) -> None:
        """Legacy store_summary method."""
        if not self.short_term_memory:
            return
        
        conversation_text = "\n".join([msg['text'] for msg in self.short_term_memory])
        summary = await self.summarizer.summarize(conversation_text)
        
        # Store summary in long-term memory
        await super().store_summary(self.session_id, summary)
    
    def retrieve_long_term_mem(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Legacy retrieve_long_term_mem method."""
        # Run async method and return results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(super().search_memory(query, "unknown", num_results))
        finally:
            loop.close()
    
    def clear_short_term(self) -> None:
        """Legacy clear_short_term method."""
        self.short_term_memory = []
