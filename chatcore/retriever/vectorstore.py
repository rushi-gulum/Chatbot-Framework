"""
Enterprise Vector Store for Chatbot Framework

This module provides vector storage and retrieval capabilities with:
- In-memory and persistent storage options
- Semantic similarity search
- Embedding management
- Performance optimization
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import asyncio
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from pathlib import Path

from ..chatbot.base_core import secure_logger


@dataclass
class Document:
    """Document representation for vector storage."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SearchResult:
    """Search result with similarity score."""
    document: Document
    similarity_score: float
    rank: int


class SimpleEmbedder:
    """Simple embedding implementation using TF-IDF-like approach."""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.documents_processed = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2]
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency."""
        tf = {}
        total_tokens = len(tokens)
        
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        # Normalize by total tokens
        for token in tf:
            tf[token] = tf[token] / total_tokens
        
        return tf
    
    def update_vocabulary(self, documents: List[str]) -> None:
        """Update vocabulary and IDF scores."""
        # Count document frequency for each term
        doc_freq = {}
        total_docs = len(documents)
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # Calculate IDF scores
        for token, freq in doc_freq.items():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
            
            # IDF = log(total_docs / doc_freq)
            self.idf_scores[token] = np.log(total_docs / freq) if freq > 0 else 0
        
        self.documents_processed += total_docs
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        tokens = self._tokenize(text)
        tf_scores = self._calculate_tf(tokens)
        
        # Create vector of size vocabulary
        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            # If no vocabulary, return zero vector
            return [0.0] * 100  # Fixed size for consistency
        
        vector = [0.0] * max(100, vocab_size)  # Ensure minimum size
        
        for token, tf in tf_scores.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_scores.get(token, 0)
                if idx < len(vector):
                    vector[idx] = tf * idf
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [float(v / norm) for v in vector]
        
        return vector


class VectorStore:
    """Enterprise vector store with similarity search capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vector store."""
        self.config = config or {}
        self.documents: Dict[str, Document] = {}
        self.embedder = SimpleEmbedder()
        
        # Configuration
        self.max_documents = self.config.get('max_documents', 10000)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.1)
        self.storage_path = self.config.get('storage_path', 'vectorstore_data.json')
        
        # Performance metrics
        self.metrics = {
            'documents_stored': 0,
            'searches_performed': 0,
            'total_search_time': 0.0,
            'cache_hits': 0
        }
        
        # Simple cache for recent searches
        self.search_cache = {}
        self.max_cache_size = 100
        
        secure_logger.info("VectorStore initialized", extra={
            'max_documents': self.max_documents,
            'similarity_threshold': self.similarity_threshold
        })
    
    async def add_document(self, document: Dict[str, Any]) -> str:
        """
        Add a document to the vector store.
        
        Args:
            document: Document dictionary with 'content' and optional 'metadata'
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID if not provided
            doc_id = document.get('id')
            if not doc_id:
                content_hash = hashlib.md5(document['content'].encode()).hexdigest()
                doc_id = f"doc_{content_hash}_{len(self.documents)}"
            
            # Create document object
            doc = Document(
                id=doc_id,
                content=document['content'],
                metadata=document.get('metadata', {}),
                timestamp=datetime.utcnow()
            )
            
            # Generate embedding
            doc.embedding = self.embedder.embed_text(doc.content)
            
            # Store document
            self.documents[doc_id] = doc
            self.metrics['documents_stored'] += 1
            
            # Update embedder vocabulary periodically
            if len(self.documents) % 50 == 0:
                await self._update_embedder_vocabulary()
            
            # Check storage limits
            if len(self.documents) > self.max_documents:
                await self._cleanup_old_documents()
            
            secure_logger.info("Document added to vector store", extra={
                'doc_id': doc_id,
                'content_length': len(doc.content),
                'total_documents': len(self.documents)
            })
            
            return doc_id
            
        except Exception as e:
            secure_logger.error(f"Error adding document: {str(e)}", extra={
                'error_type': type(e).__name__
            })
            raise
    
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                    num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            num_results: Maximum number of results
            
        Returns:
            List of search results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check cache first
            cache_key = f"{query}_{str(filters)}_{num_results}"
            if cache_key in self.search_cache:
                self.metrics['cache_hits'] += 1
                return self.search_cache[cache_key]
            
            if not self.documents:
                return []
            
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc in self.documents.items():
                if doc.embedding:
                    # Apply filters if provided
                    if filters and not self._matches_filters(doc.metadata, filters):
                        continue
                    
                    similarity = self._cosine_similarity(query_embedding, doc.embedding)
                    if similarity >= self.similarity_threshold:
                        similarities.append((doc_id, similarity))
            
            # Sort by similarity and take top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:num_results]
            
            # Format results
            results = []
            for rank, (doc_id, similarity) in enumerate(top_results):
                doc = self.documents[doc_id]
                result = {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'similarity_score': similarity,
                    'rank': rank + 1,
                    'timestamp': doc.timestamp.isoformat() if doc.timestamp else None
                }
                results.append(result)
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics['searches_performed'] += 1
            self.metrics['total_search_time'] += processing_time
            
            # Cache result
            if len(self.search_cache) < self.max_cache_size:
                self.search_cache[cache_key] = results
            
            secure_logger.info("Vector search completed", extra={
                'query_length': len(query),
                'results_found': len(results),
                'processing_time': processing_time,
                'cache_hit': False
            })
            
            return results
            
        except Exception as e:
            secure_logger.error(f"Error during vector search: {str(e)}", extra={
                'query': query[:100],  # Log first 100 chars
                'error_type': type(e).__name__
            })
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Ensure vectors have same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = np.sqrt(sum(a * a for a in vec1))
        magnitude2 = np.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    async def _update_embedder_vocabulary(self) -> None:
        """Update embedder vocabulary with current documents."""
        try:
            document_texts = [doc.content for doc in self.documents.values()]
            self.embedder.update_vocabulary(document_texts)
            
            # Re-embed documents with updated vocabulary
            for doc in self.documents.values():
                doc.embedding = self.embedder.embed_text(doc.content)
            
            secure_logger.info("Embedder vocabulary updated", extra={
                'vocabulary_size': len(self.embedder.vocabulary),
                'documents_processed': len(document_texts)
            })
            
        except Exception as e:
            secure_logger.error(f"Error updating embedder vocabulary: {str(e)}")
    
    async def _cleanup_old_documents(self) -> None:
        """Remove oldest documents to maintain storage limits."""
        try:
            # Sort documents by timestamp and remove oldest
            sorted_docs = sorted(
                self.documents.items(),
                key=lambda x: x[1].timestamp or datetime.min
            )
            
            # Remove oldest 10% of documents
            remove_count = len(self.documents) // 10
            for i in range(remove_count):
                doc_id, _ = sorted_docs[i]
                del self.documents[doc_id]
            
            secure_logger.info("Cleaned up old documents", extra={
                'removed_count': remove_count,
                'remaining_documents': len(self.documents)
            })
            
        except Exception as e:
            secure_logger.error(f"Error during cleanup: {str(e)}")
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                secure_logger.info(f"Document deleted: {doc_id}")
                return True
            return False
        except Exception as e:
            secure_logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID."""
        try:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                return {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'timestamp': doc.timestamp.isoformat() if doc.timestamp else None
                }
            return None
        except Exception as e:
            secure_logger.error(f"Error retrieving document: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.embedder.vocabulary),
            'metrics': {
                **self.metrics,
                'average_search_time': (
                    self.metrics['total_search_time'] / self.metrics['searches_performed']
                    if self.metrics['searches_performed'] > 0 else 0
                ),
                'cache_hit_rate': (
                    self.metrics['cache_hits'] / self.metrics['searches_performed']
                    if self.metrics['searches_performed'] > 0 else 0
                )
            }
        }
    
    async def save_to_disk(self, file_path: Optional[str] = None) -> bool:
        """Save vector store to disk."""
        try:
            save_path = file_path or self.storage_path
            
            # Prepare data for serialization
            data = {
                'documents': {},
                'vocabulary': self.embedder.vocabulary,
                'idf_scores': self.embedder.idf_scores,
                'metrics': self.metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Convert documents to serializable format
            for doc_id, doc in self.documents.items():
                data['documents'][doc_id] = {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'embedding': doc.embedding,
                    'timestamp': doc.timestamp.isoformat() if doc.timestamp else None
                }
            
            # Save to file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            secure_logger.info(f"Vector store saved to {save_path}")
            return True
            
        except Exception as e:
            secure_logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    async def load_from_disk(self, file_path: Optional[str] = None) -> bool:
        """Load vector store from disk."""
        try:
            load_path = file_path or self.storage_path
            
            if not Path(load_path).exists():
                secure_logger.warning(f"Vector store file not found: {load_path}")
                return False
            
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore documents
            self.documents = {}
            for doc_id, doc_data in data.get('documents', {}).items():
                timestamp = None
                if doc_data.get('timestamp'):
                    timestamp = datetime.fromisoformat(doc_data['timestamp'])
                
                doc = Document(
                    id=doc_data['id'],
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    embedding=doc_data.get('embedding'),
                    timestamp=timestamp
                )
                self.documents[doc_id] = doc
            
            # Restore embedder state
            self.embedder.vocabulary = data.get('vocabulary', {})
            self.embedder.idf_scores = data.get('idf_scores', {})
            
            # Restore metrics
            self.metrics.update(data.get('metrics', {}))
            
            secure_logger.info(f"Vector store loaded from {load_path}", extra={
                'documents_loaded': len(self.documents),
                'vocabulary_size': len(self.embedder.vocabulary)
            })
            return True
            
        except Exception as e:
            secure_logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            # Test basic operations
            test_doc = {
                'content': 'This is a test document for health check.',
                'metadata': {'test': True}
            }
            
            # Test add document
            doc_id = await self.add_document(test_doc)
            
            # Test search
            results = await self.search('test document', num_results=1)
            
            # Cleanup test document
            await self.delete_document(doc_id)
            
            return {
                'status': 'healthy',
                'test_successful': len(results) > 0,
                'stats': self.get_stats()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'stats': self.get_stats()
            }


# Export the main classes
__all__ = ['VectorStore', 'Document', 'SearchResult']
