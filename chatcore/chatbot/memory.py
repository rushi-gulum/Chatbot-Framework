from typing import List, Dict, Any
from datetime import datetime
import json
from ..retriever.vectorstore import VectorStore
from ..chatbot.summarizer import Summarizer

class Memory:
    def __init__(self, session_id: str):
        """
        Initialize multi-layered memory system.
        
        Args:
            session_id: Unique identifier for the current session
        """
        self.session_id = session_id
        self.short_term_memory: List[Dict[str, Any]] = []  # In-session message buffer
        self.vector_store = VectorStore()  # For long-term memory
        self.summarizer = Summarizer()  # For generating episodic memories
        
    def save_message(self, message: Dict[str, Any]) -> None:
        """
        Save a message to both short-term and long-term memory.
        
        Args:
            message: Dictionary containing message data (text, timestamp, role, etc.)
        """
        # Add to short-term memory buffer
        self.short_term_memory.append({
            **message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Store in long-term memory vectorstore
        self.vector_store.add_document({
            'content': message['text'],
            'metadata': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'role': message.get('role', 'user'),
                'type': 'message'
            }
        })
        
    def get_recent_context(self, num_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages from short-term memory.
        
        Args:
            num_messages: Number of most recent messages to retrieve
            
        Returns:
            List of recent messages
        """
        return self.short_term_memory[-num_messages:]
    
    def store_summary(self) -> None:
        """
        Create and store an episodic memory summary of the current session.
        """
        if not self.short_term_memory:
            return
            
        # Generate summary of the conversation
        conversation_text = "\n".join(
            [msg['text'] for msg in self.short_term_memory]
        )
        summary = self.summarizer.summarize(conversation_text)
        
        # Store the summary in long-term memory
        self.vector_store.add_document({
            'content': summary,
            'metadata': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'type': 'summary'
            }
        })
        
    def retrieve_long_term_mem(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search long-term memory for relevant information.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of relevant memories (messages and summaries)
        """
        return self.vector_store.search(
            query=query,
            num_results=num_results
        )

    def clear_short_term(self) -> None:
        """
        Clear the short-term memory buffer.
        """
        self.short_term_memory = []
