"""
Simple Conversation Memory
=========================

Basic memory management for conversations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ChatMessage:
    """Represents a chat message."""
    user_id: str
    text: str
    timestamp: datetime
    is_user: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ConversationSession:
    """Represents a conversation session."""
    session_id: str
    user_id: str
    messages: List[ChatMessage]
    started_at: datetime
    last_activity: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConversationMemory:
    """Simple conversation memory manager."""
    
    def __init__(self, max_messages_per_session: int = 100, session_timeout_hours: int = 24):
        """Initialize conversation memory."""
        self.max_messages_per_session = max_messages_per_session
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # In-memory storage
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        
    async def create_session(self, session_id: str, user_id: str) -> ConversationSession:
        """Create a new conversation session."""
        now = datetime.utcnow()
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            started_at=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        self.user_sessions[user_id].append(session_id)
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session."""
        return self.sessions.get(session_id)
    
    async def add_message(self, session_id: str, message: ChatMessage) -> bool:
        """Add message to session."""
        session = await self.get_session(session_id)
        if not session:
            session = await self.create_session(session_id, message.user_id)
        
        session.messages.append(message)
        session.last_activity = datetime.utcnow()
        
        # Trim messages if exceeding limit
        if len(session.messages) > self.max_messages_per_session:
            session.messages = session.messages[-self.max_messages_per_session:]
        
        return True
    
    async def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get conversation history for session."""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    async def shutdown(self):
        """Shutdown memory manager."""
        pass
