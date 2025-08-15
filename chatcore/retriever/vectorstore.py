"""
Enterprise Vector Store Module

This module provides a comprehensive vector store implementation with support for
multiple backends (FAISS, Pinecone, ChromaDB) and advanced features like
indexing, search, and security.
"""

import json
import asyncio
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from functools import wraps

# Import from base module to avoid duplication
from .base import VectorStoreType, Document, SearchResult

# Custom imports with try/catch for optional dependencies
try:
    from ..security.encryptor import DataEncryptor
except ImportError:
    DataEncryptor = None

# Conditional imports with try/catch blocks
faiss = None
pinecone = None
chromadb = None
cohere = None

try:
    import faiss
except ImportError:
    pass

try:
    import pinecone
except ImportError:
    pass

try:
    import chromadb
except ImportError:
    pass

try:
    import cohere
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
secure_logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying operations on failure."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    secure_logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}")
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimension = config.get('dimension', 384)
        self.metric = config.get('metric', 'cosine')
        self.encryption_enabled = config.get('encryption_enabled', False)
        self.encryptor = DataEncryptor() if self.encryption_enabled and DataEncryptor else None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store."""
        pass
    
    @abstractmethod
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update a document in the vector store."""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Get the total number of documents."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    def _encrypt_content(self, content: str) -> str:
        """Encrypt content if encryption is enabled."""
        if self.encryptor:
            return self.encryptor.encrypt(content)
        return content
    
    def _decrypt_content(self, encrypted_content: str) -> str:
        """Decrypt content if encryption is enabled."""
        if self.encryptor:
            return self.encryptor.decrypt(encrypted_content)
        return encrypted_content


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index = None
        self.id_to_doc: Dict[str, Document] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        self.index_to_doc_id: Dict[int, str] = {}
        self.next_index = 0
        self.index_file = config.get('index_file', 'faiss_index.bin')
        self.metadata_file = config.get('metadata_file', 'faiss_metadata.json')
    
    async def initialize(self) -> None:
        """Initialize FAISS index."""
        if faiss is None:
            raise VectorStoreError("FAISS library not installed")
        
        try:
            # Create index based on metric type
            if self.metric == 'cosine':
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            elif self.metric == 'euclidean':
                self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
            else:
                raise VectorStoreError(f"Unsupported metric: {self.metric}")
            
            # Try to load existing index
            await self._load_index()
            
            secure_logger.info(f"FAISS vector store initialized with {self.get_document_count()} documents")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize FAISS: {str(e)}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to FAISS index."""
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
        
        doc_ids = []
        embeddings = []
        
        for doc in documents:
            if not doc.embedding:
                raise VectorStoreError(f"Document {doc.id} missing embedding")
            
            # Encrypt content if needed
            if self.encryption_enabled:
                doc.content = self._encrypt_content(doc.content)
            
            # Assign index
            index_id = self.next_index
            self.next_index += 1
            
            # Store mappings
            self.id_to_doc[doc.id] = doc
            self.doc_id_to_index[doc.id] = index_id
            self.index_to_doc_id[index_id] = doc.id
            
            embeddings.append(doc.embedding)
            doc_ids.append(doc.id)
        
        # Add to FAISS index
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine' and faiss is not None:
            faiss.normalize_L2(embedding_matrix)
        
        self.index.add(embedding_matrix)
        
        # Save index
        await self._save_index()
        
        secure_logger.info(f"Added {len(documents)} documents to FAISS index")
        return doc_ids
    
    async def search(self, query_embedding: List[float], k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search FAISS index for similar documents."""
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
        
        # Prepare query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == 'cosine' and faiss is not None:
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc_id = self.index_to_doc_id.get(idx)
            if not doc_id:
                continue
            
            doc = self.id_to_doc.get(doc_id)
            if not doc:
                continue
            
            # Apply filters if provided
            if filters and not self._matches_filters(doc.metadata, filters):
                continue
            
            # Decrypt content if needed
            if self.encryption_enabled:
                doc.content = self._decrypt_content(doc.content)
            
            # Convert score to similarity (FAISS returns distances/similarities)
            similarity_score = float(score) if self.metric == 'cosine' else 1.0 / (1.0 + float(score))
            
            results.append(SearchResult(
                document=doc,
                similarity_score=similarity_score,
                rank=rank + 1
            ))
        
        return results
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from FAISS index."""
        # FAISS doesn't support deletion directly, so we mark as deleted
        if doc_id in self.id_to_doc:
            index_id = self.doc_id_to_index.get(doc_id)
            if index_id is not None:
                # Remove from mappings
                del self.id_to_doc[doc_id]
                del self.doc_id_to_index[doc_id]
                del self.index_to_doc_id[index_id]
            
            await self._save_index()
            return True
        return False
    
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update document in FAISS index."""
        # For FAISS, we need to delete and re-add
        await self.delete_document(doc_id)
        await self.add_documents([document])
        return True
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.id_to_doc)
    
    async def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            if self.index and faiss is not None:
                faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            metadata = {
                'id_to_doc': {doc_id: {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'timestamp': doc.timestamp.isoformat() if doc.timestamp else None
                } for doc_id, doc in self.id_to_doc.items()},
                'doc_id_to_index': self.doc_id_to_index,
                'index_to_doc_id': {str(k): v for k, v in self.index_to_doc_id.items()},
                'next_index': self.next_index
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            secure_logger.error(f"Error saving FAISS index: {str(e)}")
    
    async def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            if Path(self.index_file).exists() and faiss is not None:
                self.index = faiss.read_index(self.index_file)
            
            # Load metadata
            if Path(self.metadata_file).exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Restore documents
                self.id_to_doc = {}
                for doc_id, doc_data in metadata.get('id_to_doc', {}).items():
                    timestamp = None
                    if doc_data.get('timestamp'):
                        timestamp = datetime.fromisoformat(doc_data['timestamp'])
                    
                    self.id_to_doc[doc_id] = Document(
                        id=doc_data['id'],
                        content=doc_data['content'],
                        metadata=doc_data['metadata'],
                        timestamp=timestamp or datetime.utcnow()
                    )
                
                # Restore mappings
                self.doc_id_to_index = metadata.get('doc_id_to_index', {})
                self.index_to_doc_id = {int(k): v for k, v in metadata.get('index_to_doc_id', {}).items()}
                self.next_index = metadata.get('next_index', 0)
            
        except Exception as e:
            secure_logger.error(f"Error loading FAISS index: {str(e)}")
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
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
    
    async def cleanup(self) -> None:
        """Cleanup FAISS resources."""
        self.index = None
        self.id_to_doc.clear()
        self.doc_id_to_index.clear()
        self.index_to_doc_id.clear()


class PineconeVectorStore(BaseVectorStore):
    """Pinecone-based vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index = None
        self.index_name = config.get('index_name', 'default-index')
        self.api_key = config.get('api_key')
        self.environment = config.get('environment')
        
        if not self.api_key:
            raise VectorStoreError("Pinecone API key required")
    
    async def initialize(self) -> None:
        """Initialize Pinecone index."""
        if pinecone is None:
            raise VectorStoreError("Pinecone library not installed")
        
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                # Create index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
                
                # Wait for index to be ready
                await asyncio.sleep(60)  # Pinecone needs time to initialize
            
            self.index = pinecone.Index(self.index_name)
            
            secure_logger.info(f"Pinecone vector store initialized: {self.index_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Pinecone index."""
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
        
        vectors = []
        for doc in documents:
            if not doc.embedding:
                raise VectorStoreError(f"Document {doc.id} missing embedding")
            
            # Encrypt content if needed
            content = self._encrypt_content(doc.content) if self.encryption_enabled else doc.content
            
            vector_data = {
                'id': doc.id,
                'values': doc.embedding,
                'metadata': {
                    'content': content,
                    'timestamp': doc.timestamp.isoformat() if doc.timestamp else None,
                    **doc.metadata
                }
            }
            vectors.append(vector_data)
        
        # Upsert in batches
        batch_size = 100
        doc_ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            doc_ids.extend([v['id'] for v in batch])
        
        secure_logger.info(f"Added {len(documents)} documents to Pinecone")
        return doc_ids
    
    async def search(self, query_embedding: List[float], k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search Pinecone index for similar documents."""
        if not self.index:
            raise VectorStoreError("Vector store not initialized")
        
        # Prepare query
        query_params = {
            'vector': query_embedding,
            'top_k': k,
            'include_metadata': True,
            'include_values': False
        }
        
        if filters:
            query_params['filter'] = filters
        
        # Search
        response = self.index.query(**query_params)
        
        results = []
        for rank, match in enumerate(response['matches']):
            # Extract document data
            metadata = match['metadata']
            content = metadata.pop('content', '')
            timestamp_str = metadata.pop('timestamp', None)
            
            # Decrypt content if needed
            if self.encryption_enabled:
                content = self._decrypt_content(content)
            
            # Parse timestamp
            timestamp = None
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            
            doc = Document(
                id=match['id'],
                content=content,
                metadata=metadata,
                timestamp=timestamp or datetime.utcnow()
            )
            
            results.append(SearchResult(
                document=doc,
                similarity_score=match['score'],
                rank=rank + 1
            ))
        
        return results
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from Pinecone index."""
        if not self.index:
            return False
        
        try:
            self.index.delete(ids=[doc_id])
            return True
        except Exception as e:
            secure_logger.error(f"Error deleting document from Pinecone: {str(e)}")
            return False
    
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update document in Pinecone index."""
        # Pinecone upsert will update if ID exists
        await self.add_documents([document])
        return True
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        if not self.index:
            return 0
        
        try:
            stats = self.index.describe_index_stats()
            return stats.get('total_vector_count', 0)
        except Exception:
            return 0
    
    async def cleanup(self) -> None:
        """Cleanup Pinecone resources."""
        pass  # Pinecone handles cleanup automatically


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.collection = None
        self.collection_name = config.get('collection_name', 'default-collection')
        self.persist_directory = config.get('persist_directory', './chroma_db')
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if chromadb is None:
            raise VectorStoreError("ChromaDB library not installed")
        
        try:
            # Initialize client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.dimension}
                )
            
            secure_logger.info(f"ChromaDB vector store initialized: {self.collection_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB collection."""
        if not self.collection:
            raise VectorStoreError("Vector store not initialized")
        
        ids = []
        embeddings = []
        metadatas = []
        documents_content = []
        
        for doc in documents:
            if not doc.embedding:
                raise VectorStoreError(f"Document {doc.id} missing embedding")
            
            # Encrypt content if needed
            content = self._encrypt_content(doc.content) if self.encryption_enabled else doc.content
            
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            documents_content.append(content)
            
            metadata = doc.metadata.copy()
            if doc.timestamp:
                metadata['timestamp'] = doc.timestamp.isoformat()
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_content
        )
        
        secure_logger.info(f"Added {len(documents)} documents to ChromaDB")
        return ids
    
    async def search(self, query_embedding: List[float], k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search ChromaDB collection for similar documents."""
        if not self.collection:
            raise VectorStoreError("Vector store not initialized")
        
        # Prepare query
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': k,
            'include': ['metadatas', 'documents', 'distances']
        }
        
        if filters:
            query_params['where'] = filters
        
        # Search
        results_data = self.collection.query(**query_params)
        
        if not results_data:
            return []
        
        # Extract data with safety checks
        ids_list = results_data.get('ids')
        distances_list = results_data.get('distances') 
        metadatas_list = results_data.get('metadatas')
        documents_list = results_data.get('documents')
        
        if not all([ids_list, distances_list, metadatas_list, documents_list]):
            return []
        
        ids = ids_list[0] if ids_list else []
        distances = distances_list[0] if distances_list else []
        metadatas = metadatas_list[0] if metadatas_list else []
        documents = documents_list[0] if documents_list else []
        
        results = []
        for rank, (doc_id, distance, metadata, content) in enumerate(zip(
            ids, distances, metadatas, documents
        )):
            # Decrypt content if needed
            if self.encryption_enabled:
                content = self._decrypt_content(content)
            
            # Parse timestamp and convert metadata to dict
            metadata_dict = dict(metadata) if metadata else {}
            timestamp = None
            timestamp_str = metadata_dict.pop('timestamp', None)
            if timestamp_str and isinstance(timestamp_str, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    pass  # Invalid timestamp format, use default
            
            doc = Document(
                id=doc_id,
                content=content,
                metadata=metadata_dict,
                timestamp=timestamp or datetime.utcnow()
            )
            
            # Convert distance to similarity score
            similarity_score = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
            
            results.append(SearchResult(
                document=doc,
                similarity_score=similarity_score,
                rank=rank + 1
            ))
        
        return results
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from ChromaDB collection."""
        if not self.collection:
            return False
        
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            secure_logger.error(f"Error deleting document from ChromaDB: {str(e)}")
            return False
    
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update document in ChromaDB collection."""
        # ChromaDB upsert will update if ID exists
        await self.add_documents([document])
        return True
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        if not self.collection:
            return 0
        
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    async def cleanup(self) -> None:
        """Cleanup ChromaDB resources."""
        pass  # ChromaDB handles persistence automatically


def _check_dependencies(store_type: VectorStoreType) -> None:
    """Check if required dependencies are installed."""
    if store_type == VectorStoreType.FAISS and faiss is None:
        raise VectorStoreError("FAISS library not installed. Run: pip install faiss-cpu")
    elif store_type == VectorStoreType.PINECONE and pinecone is None:
        raise VectorStoreError("Pinecone library not installed. Run: pip install pinecone-client")
    elif store_type == VectorStoreType.CHROMA and chromadb is None:
        raise VectorStoreError("ChromaDB library not installed. Run: pip install chromadb")

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create_vectorstore(vectorstore_type: VectorStoreType, config: Dict[str, Any]) -> BaseVectorStore:
        """Create vector store instance based on type."""
        _check_dependencies(vectorstore_type)
        
        vectorstore_map = {
            VectorStoreType.FAISS: FAISSVectorStore,
            VectorStoreType.PINECONE: PineconeVectorStore,
            VectorStoreType.CHROMA: ChromaVectorStore,
            # TODO: Add Milvus and Weaviate implementations
        }
        
        if vectorstore_type not in vectorstore_map:
            raise VectorStoreError(f"Unsupported vector store type: {vectorstore_type}")
        
        vectorstore_class = vectorstore_map[vectorstore_type]
        return vectorstore_class(config)


# Export the main classes
__all__ = [
    'BaseVectorStore',
    'FAISSVectorStore', 
    'PineconeVectorStore',
    'ChromaVectorStore',
    'VectorStoreFactory',
    'VectorStoreType',
    'VectorStoreError',
    'Document',
    'SearchResult'
]
