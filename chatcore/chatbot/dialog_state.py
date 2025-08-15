"""
Simple Dialog State Management
=============================

Basic dialog state tracking for multi-turn conversations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class DialogStatus(Enum):
    """Dialog session status."""
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DialogTurn:
    """Single turn in a dialog."""
    user_input: str
    bot_response: str
    intent: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DialogState:
    """Dialog state for a conversation session."""
    session_id: str
    user_id: str
    status: DialogStatus = DialogStatus.ACTIVE
    current_intent: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    turns: List[DialogTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_turn(self, turn: DialogTurn):
        """Add a turn to the dialog."""
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()
        if turn.intent:
            self.current_intent = turn.intent
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get value from dialog context."""
        return self.context.get(key, default)
    
    def set_context_value(self, key: str, value: Any):
        """Set value in dialog context."""
        self.context[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_last_user_input(self) -> str:
        """Get the last user input."""
        if self.turns:
            return self.turns[-1].user_input
        return ""
    
    def get_conversation_history(self, limit: int = 10) -> List[str]:
        """Get conversation history as list of messages."""
        history = []
        for turn in self.turns[-limit:]:
            history.append(f"User: {turn.user_input}")
            history.append(f"Bot: {turn.bot_response}")
        return history


class SimpleDialogManager:
    """Simple dialog state manager."""
    
    def __init__(self):
        """Initialize dialog manager."""
        self.sessions: Dict[str, DialogState] = {}
    
    async def get_or_create_session(self, session_id: str, user_id: str) -> DialogState:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = DialogState(
                session_id=session_id,
                user_id=user_id
            )
        return self.sessions[session_id]
    
    async def update_session(self, session_id: str, user_input: str, 
                           bot_response: str, intent: str = "", 
                           entities: Optional[Dict[str, Any]] = None) -> bool:
        """Update session with new turn."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        turn = DialogTurn(
            user_input=user_input,
            bot_response=bot_response,
            intent=intent,
            entities=entities or {}
        )
        
        session.add_turn(turn)
        return True
    
    async def set_context(self, session_id: str, key: str, value: Any) -> bool:
        """Set context value for session."""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].set_context_value(key, value)
        return True
    
    async def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get context value from session."""
        if session_id not in self.sessions:
            return default
        
        return self.sessions[session_id].get_context_value(key, default)
    
    async def end_session(self, session_id: str, status: DialogStatus = DialogStatus.COMPLETED) -> bool:
        """End a dialog session."""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].status = status
        self.sessions[session_id].updated_at = datetime.utcnow()
        return True
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions."""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.status == DialogStatus.ACTIVE)
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'completed_sessions': total_sessions - active_sessions
        }
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        old_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.updated_at < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        return len(old_sessions)
