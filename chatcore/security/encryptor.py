"""
Data Encryption Module for Chatbot Framework

Provides encryption capabilities for sensitive data storage and transmission.
"""

import base64
import hashlib
from typing import Optional


class DataEncryptor:
    """Simple data encryptor for demonstration purposes."""
    
    def __init__(self, key: Optional[str] = None):
        """Initialize encryptor with optional key."""
        self.key = key or "default_encryption_key"
        
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        # Simple base64 encoding for demo (use proper encryption in production)
        encoded = base64.b64encode(data.encode()).decode()
        return encoded
        
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        # Simple base64 decoding for demo (use proper decryption in production)
        try:
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return decoded
        except Exception:
            return encrypted_data  # Return original if decryption fails