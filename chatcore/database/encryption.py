"""
Data Encryption Manager
======================

Provides secure encryption/decryption for sensitive data at rest.
Supports multiple encryption algorithms and key management strategies.

Security Features:
- AES-256 encryption with Fernet (symmetric)
- Key derivation from master keys
- Field-level and record-level encryption
- Secure key rotation support
- Environment-based key management

Usage:
    encryption_manager = EncryptionManager(encryption_key)
    
    # Encrypt sensitive data
    encrypted_data = await encryption_manager.encrypt_data(
        {"email": "user@example.com", "ssn": "123-45-6789"}
    )
    
    # Decrypt when needed
    decrypted_data = await encryption_manager.decrypt_data(encrypted_data)
"""

import os
import asyncio
import base64
import json
from typing import Any, Dict, List, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

from .base import EncryptionLevel, SecurityError

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Manages data encryption and decryption for database storage.
    
    Provides secure encryption of sensitive data using industry-standard
    algorithms and best practices for key management.
    """
    
    # Fields that should always be encrypted if present
    SENSITIVE_FIELDS = {
        'password', 'ssn', 'social_security_number', 'credit_card',
        'credit_card_number', 'bank_account', 'routing_number',
        'api_key', 'secret', 'token', 'private_key', 'email',
        'phone', 'phone_number', 'address', 'personal_data'
    }
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        encryption_level: EncryptionLevel = EncryptionLevel.FIELD_LEVEL,
        salt: Optional[bytes] = None
    ):
        """
        Initialize encryption manager.
        
        Args:
            encryption_key: Master encryption key (preferably from environment)
            encryption_level: Level of encryption to apply
            salt: Optional salt for key derivation
        """
        self.encryption_level = encryption_level
        self._master_key = encryption_key or self._get_master_key()
        self._salt = salt or self._generate_salt()
        self._fernet = self._create_fernet_instance()
        
        if not self._master_key:
            logger.warning("No encryption key provided - encryption disabled")
            self.encryption_level = EncryptionLevel.NONE
    
    def _get_master_key(self) -> Optional[str]:
        """
        Get master encryption key from environment variables.
        
        Returns:
            Encryption key or None if not found
        """
        # Try multiple environment variable names
        key_vars = [
            'CHATCORE_ENCRYPTION_KEY',
            'DATABASE_ENCRYPTION_KEY', 
            'ENCRYPTION_KEY',
            'MASTER_KEY'
        ]
        
        for var in key_vars:
            key = os.getenv(var)
            if key:
                logger.info(f"Loaded encryption key from {var}")
                return key
        
        logger.warning("No encryption key found in environment variables")
        return None
    
    def _generate_salt(self) -> bytes:
        """
        Generate or retrieve salt for key derivation.
        
        Returns:
            32-byte salt
        """
        # In production, you might want to store/retrieve salt from a secure location
        salt_env = os.getenv('CHATCORE_ENCRYPTION_SALT')
        if salt_env:
            try:
                return base64.b64decode(salt_env)
            except Exception:
                logger.warning("Invalid salt in environment, generating new one")
        
        # Generate new salt
        salt = os.urandom(32)
        logger.info("Generated new encryption salt")
        return salt
    
    def _create_fernet_instance(self) -> Optional[Fernet]:
        """
        Create Fernet encryption instance from master key.
        
        Returns:
            Fernet instance or None if no key available
        """
        if not self._master_key:
            return None
        
        try:
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self._salt,
                iterations=100000,  # OWASP recommended minimum
            )
            key = base64.urlsafe_b64encode(
                kdf.derive(self._master_key.encode('utf-8'))
            )
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to create encryption instance: {e}")
            raise SecurityError(f"Encryption initialization failed: {e}")
    
    async def encrypt_data(
        self,
        data: Union[Dict[str, Any], str, bytes],
        force_encrypt: bool = False
    ) -> Union[Dict[str, Any], str]:
        """
        Encrypt data based on encryption level and sensitivity.
        
        Args:
            data: Data to encrypt
            force_encrypt: Force encryption regardless of level
            
        Returns:
            Encrypted data in same format as input
            
        Raises:
            SecurityError: If encryption fails
        """
        if self.encryption_level == EncryptionLevel.NONE and not force_encrypt:
            return data
        
        if not self._fernet:
            if force_encrypt:
                raise SecurityError("Encryption requested but no key available")
            return data
        
        try:
            if isinstance(data, dict):
                return await self._encrypt_dict(data, force_encrypt)
            elif isinstance(data, str):
                return await self._encrypt_string(data)
            elif isinstance(data, bytes):
                return await self._encrypt_bytes(data)
            else:
                # Convert to JSON string and encrypt
                json_str = json.dumps(data, default=str)
                return await self._encrypt_string(json_str)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityError(f"Data encryption failed: {e}")
    
    async def decrypt_data(
        self,
        data: Union[Dict[str, Any], str],
        is_encrypted: bool = True
    ) -> Union[Dict[str, Any], str]:
        """
        Decrypt previously encrypted data.
        
        Args:
            data: Encrypted data
            is_encrypted: Whether data is actually encrypted
            
        Returns:
            Decrypted data
            
        Raises:
            SecurityError: If decryption fails
        """
        if not is_encrypted or not self._fernet:
            return data
        
        try:
            if isinstance(data, dict):
                return await self._decrypt_dict(data)
            elif isinstance(data, str):
                return await self._decrypt_string(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Data decryption failed: {e}")
    
    async def _encrypt_dict(
        self,
        data: Dict[str, Any],
        force_encrypt: bool = False
    ) -> Dict[str, Any]:
        """
        Encrypt dictionary data based on encryption level.
        
        Args:
            data: Dictionary to encrypt
            force_encrypt: Force encryption of all fields
            
        Returns:
            Dictionary with encrypted fields
        """
        if self.encryption_level == EncryptionLevel.RECORD_LEVEL or force_encrypt:
            # Encrypt entire record
            json_str = json.dumps(data, default=str)
            encrypted_str = await self._encrypt_string(json_str)
            return {
                '_encrypted_data': encrypted_str,
                '_encryption_version': '1.0',
                '_encryption_timestamp': asyncio.get_event_loop().time()
            }
        
        elif self.encryption_level == EncryptionLevel.FIELD_LEVEL:
            # Encrypt only sensitive fields
            encrypted_data = {}
            for key, value in data.items():
                if self._is_sensitive_field(key) or force_encrypt:
                    if isinstance(value, (str, int, float, bool)):
                        encrypted_data[key] = await self._encrypt_string(str(value))
                        encrypted_data[f'_{key}_encrypted'] = True
                    else:
                        # For complex types, convert to JSON first
                        json_str = json.dumps(value, default=str)
                        encrypted_data[key] = await self._encrypt_string(json_str)
                        encrypted_data[f'_{key}_encrypted'] = True
                else:
                    encrypted_data[key] = value
            return encrypted_data
        
        return data
    
    async def _decrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt dictionary data.
        
        Args:
            data: Encrypted dictionary
            
        Returns:
            Decrypted dictionary
        """
        # Check if this is a record-level encrypted dict
        if '_encrypted_data' in data:
            encrypted_str = data['_encrypted_data']
            decrypted_str = await self._decrypt_string(encrypted_str)
            return json.loads(decrypted_str)
        
        # Field-level decryption
        decrypted_data = {}
        for key, value in data.items():
            if key.startswith('_') and key.endswith('_encrypted'):
                # Skip encryption metadata
                continue
            
            if f'_{key}_encrypted' in data and data[f'_{key}_encrypted']:
                # This field is encrypted
                try:
                    decrypted_str = await self._decrypt_string(value)
                    # Try to parse as JSON first, fall back to string
                    try:
                        decrypted_data[key] = json.loads(decrypted_str)
                    except json.JSONDecodeError:
                        decrypted_data[key] = decrypted_str
                except Exception as e:
                    logger.warning(f"Failed to decrypt field {key}: {e}")
                    decrypted_data[key] = value
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    async def _encrypt_string(self, data: str) -> str:
        """
        Encrypt string data.
        
        Args:
            data: String to encrypt
            
        Returns:
            Base64 encoded encrypted string
        """
        if not self._fernet:
            return data
        
        encrypted_bytes = self._fernet.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    async def _decrypt_string(self, encrypted_data: str) -> str:
        """
        Decrypt string data.
        
        Args:
            encrypted_data: Base64 encoded encrypted string
            
        Returns:
            Decrypted string
        """
        if not self._fernet:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to decrypt string data: {e}")
            raise SecurityError(f"String decryption failed: {e}")
    
    async def _encrypt_bytes(self, data: bytes) -> str:
        """
        Encrypt binary data.
        
        Args:
            data: Bytes to encrypt
            
        Returns:
            Base64 encoded encrypted string
        """
        if not self._fernet:
            return base64.b64encode(data).decode('utf-8')
        
        encrypted_bytes = self._fernet.encrypt(data)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """
        Check if a field contains sensitive data that should be encrypted.
        
        Args:
            field_name: Field name to check
            
        Returns:
            True if field should be encrypted
        """
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.SENSITIVE_FIELDS)
    
    async def rotate_key(self, new_key: str) -> None:
        """
        Rotate encryption key (for future implementation).
        
        This would involve:
        1. Creating new Fernet instance with new key
        2. Re-encrypting all encrypted data with new key
        3. Updating key storage
        
        Args:
            new_key: New encryption key
        """
        # TODO: Implement key rotation
        logger.warning("Key rotation not yet implemented")
        raise NotImplementedError("Key rotation not yet implemented")
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """
        Get information about current encryption configuration.
        
        Returns:
            Encryption configuration info
        """
        return {
            'encryption_level': self.encryption_level.value,
            'encryption_enabled': self._fernet is not None,
            'sensitive_fields': list(self.SENSITIVE_FIELDS),
            'key_available': self._master_key is not None
        }
