"""
Enterprise Embedder Implementation for Chatbot Framework

This module provides embedder implementations with support for:
- OpenAI embeddings
- HuggingFace transformers
- Cohere embeddings
- Sentence Transformers
- Custom embedding models
- Async operations and caching
- Batch processing optimization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import hashlib
import time
from dataclasses import dataclass
import json
import numpy as np

from .base import BaseEmbedder, Embedding, EmbedderType, EmbeddingError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations."""
    total_requests: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings implementation with async support."""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            super().__init__(config)
            self.model = config.get('model', 'text-embedding-3-small')
            self.api_key = config.get('api_key')
            self.max_tokens = config.get('max_tokens', 8192)
            self.batch_size = config.get('batch_size', 100)
            
            # Initialize metrics as a dictionary with proper structure
            self._metrics.update({
                'total_requests': 0,
                'total_tokens': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'avg_latency_ms': 0.0,
                'error_count': 0
            })
            
            if not self.api_key:
                raise EmbeddingError("OpenAI API key is required")
            
            # Initialize OpenAI client
            self.client = None
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise EmbeddingError("openai package is required for OpenAI embeddings")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI embedder: {str(e)}")
    
    async def embed_text(self, text: str) -> Embedding:
        """Generate embedding for a single text using OpenAI API."""
        if not self.client:
            raise EmbeddingError("OpenAI client not initialized")
            
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached:
            self.metrics['cache_hits'] += 1
            return cached
        
        start_time = time.time()
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Update metrics
            self.metrics['total_requests'] += 1
            self.metrics['total_tokens'] += response.usage.total_tokens
            self.metrics['cache_misses'] += 1
            
            latency = (time.time() - start_time) * 1000
            self.metrics['avg_latency_ms'] = (
                (self.metrics['avg_latency_ms'] * (self.metrics['total_requests'] - 1) + latency) 
                / self.metrics['total_requests']
            )
            
            # Cache the result
            self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingError(f"Failed to generate OpenAI embedding: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for multiple texts with batching."""
        if not texts:
            return []
        
        if not self.client:
            raise EmbeddingError("OpenAI client not initialized")
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = self._get_cached_embedding(text)
            if cached:
                embeddings.append(cached)
                self.metrics['cache_hits'] += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return [e for e in embeddings if e is not None]
        
        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[i:i + self.batch_size]
            batch_indices = uncached_indices[i:i + self.batch_size]
            
            try:
                start_time = time.time()
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['total_tokens'] += response.usage.total_tokens
                self.metrics['cache_misses'] += len(batch)
                
                latency = (time.time() - start_time) * 1000
                self.metrics['avg_latency_ms'] = (
                    (self.metrics['avg_latency_ms'] * (self.metrics['total_requests'] - 1) + latency) 
                    / self.metrics['total_requests']
                )
                
                # Store results and cache
                for j, (text, embedding_data) in enumerate(zip(batch, response.data)):
                    embedding = embedding_data.embedding
                    embeddings[batch_indices[j]] = embedding
                    self._cache_embedding(text, embedding)
                    
            except Exception as e:
                self.metrics['error_count'] += 1
                logger.error(f"OpenAI batch embedding error: {e}")
                raise EmbeddingError(f"Failed to generate OpenAI batch embeddings: {e}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension based on model."""
        dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        return dimensions.get(self.model, 1536)


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace transformers implementation for embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            super().__init__(config)
            self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            self.device = config.get('device', 'cpu')
            self.max_length = config.get('max_length', 512)
            self.batch_size = config.get('batch_size', 32)
            
            # Initialize metrics as a dictionary with proper structure
            self._metrics.update({
                'total_requests': 0,
                'total_tokens': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'avg_latency_ms': 0.0,
                'error_count': 0
            })
            
            # Initialize model
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise EmbeddingError("sentence-transformers package is required")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize HuggingFace embedder: {str(e)}")
    
    async def embed_text(self, text: str) -> Embedding:
        """Generate embedding for a single text."""
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached:
            self.metrics['cache_hits'] += 1
            return cached
        
        start_time = time.time()
        
        try:
            # Run in thread pool for async compatibility
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode([text], convert_to_numpy=True)[0].tolist()
            )
            
            # Update metrics
            self.metrics['total_requests'] += 1
            self.metrics['cache_misses'] += 1
            
            latency = (time.time() - start_time) * 1000
            self.metrics['avg_latency_ms'] = (
                (self.metrics['avg_latency_ms'] * (self.metrics['total_requests'] - 1) + latency) 
                / self.metrics['total_requests']
            )
            
            # Cache the result
            self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"HuggingFace embedding error: {e}")
            raise EmbeddingError(f"Failed to generate HuggingFace embedding: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = self._get_cached_embedding(text)
            if cached:
                embeddings.append(cached)
                self.metrics['cache_hits'] += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return embeddings
        
        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[i:i + self.batch_size]
            batch_indices = uncached_indices[i:i + self.batch_size]
            
            try:
                start_time = time.time()
                
                # Run in thread pool for async compatibility
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(batch, convert_to_numpy=True, batch_size=self.batch_size)
                )
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['cache_misses'] += len(batch)
                
                latency = (time.time() - start_time) * 1000
                self.metrics['avg_latency_ms'] = (
                    (self.metrics['avg_latency_ms'] * (self.metrics['total_requests'] - 1) + latency) 
                    / self.metrics['total_requests']
                )
                
                # Store results and cache
                for j, (text, embedding) in enumerate(zip(batch, batch_embeddings)):
                    embedding_list = embedding.tolist()
                    embeddings[batch_indices[j]] = embedding_list
                    self._cache_embedding(text, embedding_list)
                    
            except Exception as e:
                self.metrics['error_count'] += 1
                logger.error(f"HuggingFace batch embedding error: {e}")
                raise EmbeddingError(f"Failed to generate HuggingFace batch embeddings: {e}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension from model."""
        if self.model is None:
            return 384  # Default dimension
        dim = self.model.get_sentence_embedding_dimension()
        return dim if dim is not None else 384


class CohereEmbedder(BaseEmbedder):
    """Cohere embeddings implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            super().__init__(config)
            self.api_key = config.get('api_key')
            self.model = config.get('model', 'embed-english-v3.0')
            self.input_type = config.get('input_type', 'search_document')
            self.batch_size = config.get('batch_size', 96)
            
            # Initialize metrics as a dictionary with proper structure
            self._metrics.update({
                'total_requests': 0,
                'total_tokens': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'avg_latency_ms': 0.0,
                'error_count': 0
            })
            
            if not self.api_key:
                raise EmbeddingError("Cohere API key is required")
            
            # Initialize Cohere client
            self.client = None
            try:
                import cohere
                self.client = cohere.AsyncClient(api_key=self.api_key)
            except ImportError:
                raise EmbeddingError("cohere package is required for Cohere embeddings")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Cohere embedder: {str(e)}")
    
    async def embed_text(self, text: str) -> Embedding:
        """Generate embedding for a single text using Cohere API."""
        if not self.client:
            raise EmbeddingError("Cohere client not initialized")
            
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached:
            self.metrics['cache_hits'] += 1
            return cached
        
        start_time = time.time()
        
        try:
            response = await self.client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type
            )
            
            # Extract embedding - Cohere API returns complex types, use type ignore
            embedding_data = response.embeddings  # type: ignore
            
            # Convert to proper List[float] format
            if hasattr(embedding_data, '__getitem__') and hasattr(embedding_data, '__len__'):
                embedding = list(embedding_data[0])  # type: ignore
            else:
                embedding = list(embedding_data)  # type: ignore
                
            # Ensure all elements are floats
            embedding = [float(x) for x in embedding if isinstance(x, (int, float))]
            
            # Update metrics
            self.metrics['total_requests'] += 1
            self.metrics['cache_misses'] += 1
            
            latency = (time.time() - start_time) * 1000
            self.metrics['avg_latency_ms'] = (
                (self.metrics['avg_latency_ms'] * (self.metrics['total_requests'] - 1) + latency) 
                / self.metrics['total_requests']
            )
            
            # Cache the result
            self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Cohere embedding error: {e}")
            raise EmbeddingError(f"Failed to generate Cohere embedding: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for multiple texts with batching."""
        if not texts:
            return []
        
        if not self.client:
            raise EmbeddingError("Cohere client not initialized")
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = self._get_cached_embedding(text)
            if cached:
                embeddings.append(cached)
                self.metrics['cache_hits'] += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return [e for e in embeddings if e is not None]
        
        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[i:i + self.batch_size]
            batch_indices = uncached_indices[i:i + self.batch_size]
            
            try:
                start_time = time.time()
                
                response = await self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=self.input_type
                )
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['cache_misses'] += len(batch)
                
                latency = (time.time() - start_time) * 1000
                self.metrics['avg_latency_ms'] = (
                    (self.metrics['avg_latency_ms'] * (self.metrics['total_requests'] - 1) + latency) 
                    / self.metrics['total_requests']
                )
                
                # Store results and cache - use type ignore for Cohere complex types
                response_embeddings = response.embeddings  # type: ignore
                for j, (text, embedding_data) in enumerate(zip(batch, response_embeddings)):
                    embedding: List[float] = list(embedding_data)  # type: ignore
                    embeddings[batch_indices[j]] = embedding
                    self._cache_embedding(text, embedding)
                    
            except Exception as e:
                self.metrics['error_count'] += 1
                logger.error(f"Cohere batch embedding error: {e}")
                raise EmbeddingError(f"Failed to generate Cohere batch embeddings: {e}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension based on model."""
        dimensions = {
            'embed-english-v3.0': 1024,
            'embed-multilingual-v3.0': 1024,
            'embed-english-light-v3.0': 384,
            'embed-multilingual-light-v3.0': 384
        }
        return dimensions.get(self.model, 1024)


class EmbedderFactory:
    """Factory for creating embedder instances."""
    
    @staticmethod
    def create_embedder(embedder_type: EmbedderType, config: Dict[str, Any]) -> BaseEmbedder:
        """Create embedder instance based on type."""
        embedder_map = {
            EmbedderType.OPENAI: OpenAIEmbedder,
            EmbedderType.HUGGINGFACE: HuggingFaceEmbedder,
            EmbedderType.SENTENCE_TRANSFORMERS: HuggingFaceEmbedder,  # Alias
            EmbedderType.COHERE: CohereEmbedder,
        }
        
        if embedder_type not in embedder_map:
            raise EmbeddingError(f"Unsupported embedder type: {embedder_type}")
        
        embedder_class = embedder_map[embedder_type]
        return embedder_class(config)


# Utility functions for embeddings
def cosine_similarity(embedding1: Embedding, embedding2: Embedding) -> float:
    """Calculate cosine similarity between two embeddings."""
    import numpy as np
    
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def normalize_embedding(embedding: Embedding) -> Embedding:
    """Normalize embedding to unit length."""
    import numpy as np
    
    vec = np.array(embedding)
    norm = np.linalg.norm(vec)
    
    if norm == 0:
        return embedding
    
    return (vec / norm).tolist()