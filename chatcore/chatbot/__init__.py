"""
Chatbot Core Module
==================

Core chatbot engine responsible for conversation management, 
intent processing, and response generation.
"""

from .base_core import ChatbotCore
from .memory import ConversationMemory, ChatMessage, ConversationSession
from .summarizer import ConversationSummarizer
from .intent_recognizer import SimpleIntentRecognizer, IntentResult
from .dialog_state import SimpleDialogManager, DialogState, DialogTurn

__version__ = "1.0.0"

__all__ = [
    'ChatbotCore',
    'ConversationMemory',
    'ChatMessage',
    'ConversationSession',
    'ConversationSummarizer',
    'SimpleIntentRecognizer',
    'IntentResult',
    'SimpleDialogManager',
    'DialogState',
    'DialogTurn'
]
