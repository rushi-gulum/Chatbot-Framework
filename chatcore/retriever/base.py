"""
Base Retriever Interface for Enterprise Chatbot Framework

This module defines the core interfaces and abstract classes for the retrieval system,
following SOLID principles with dependency injection and pluggable backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio
from pathlib import Path

# Type aliases for better readability
Embedding = List[float]
DocumentId = str
Metadata = Dict[str, Any]


class VectorStoreType(Enum):
    """Supported vector store backends."""
    FAISS = "faiss"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"


class EmbedderType(Enum):
    """Supported embedding model types."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"


class RerankerType(Enum):
    """Supported reranking strategies."""
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    SEMANTIC = "semantic"
    NONE = "none"


@dataclass
class Document:
    """Document representation with metadata and security features."""
    id: DocumentId = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Metadata = field(default_factory=dict)
    embedding: Optional[Embedding] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_level: str = "public"
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        if self.checksum is None:
            import hashlib
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class SearchQuery:
    """Search query with advanced filtering and security options."""
    text: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    access_level: str = "public"
    rerank: bool = True
    include_metadata: bool = True
    similarity_threshold: float = 0.0


@dataclass
class SearchResult:
    """Search result with comprehensive scoring and metadata."""
    document: Document
    similarity_score: float
    rerank_score: Optional[float] = None
    rank: int = 0
    retrieval_time_ms: float = 0.0
    explanation: Optional[str] = None


@dataclass
class IndexingConfig:
    """Configuration for document indexing operations."""
    batch_size: int = 100
    chunk_size: int = 1000
    overlap: int = 100
    enable_parallel: bool = True
    validate_embeddings: bool = True
    deduplicate: bool = True


@dataclass
class RetrieverConfig:
    """Comprehensive configuration for retriever components."""
    # Core settings
    embedder_type: EmbedderType = EmbedderType.SENTENCE_TRANSFORMERS
    vectorstore_type: VectorStoreType = VectorStoreType.FAISS
    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    
    # Performance settings
    max_concurrent_requests: int = 10
    embedding_cache_size: int = 1000
    request_timeout: float = 30.0
    
    # Security settings
    enable_access_control: bool = True
    encrypt_embeddings: bool = False
    audit_logging: bool = True
    
    # Indexing settings
    indexing_config: IndexingConfig = field(default_factory=IndexingConfig)
    
    # Backend-specific settings
    backend_config: Dict[str, Any] = field(default_factory=dict)


class BaseEmbedder(ABC):
    """Base class for embedding implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedder with configuration."""
        self.config = config
        self._metrics = {}
        self._cache = {}  # Simple in-memory cache
        self._validate_config(config)
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings."""
        pass
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Return embedder metrics."""
        return self._metrics
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self._cache.get(text_hash)
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self._cache[text_hash] = embedding
        
        # Simple cache size limit
        if len(self._cache) > 1000:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration."""
        # Base validation - can be overridden by subclasses
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._dimension: Optional[int] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: Embedding, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[DocumentId]) -> None:
        """Delete documents from the vector store."""
        pass
    
    @abstractmethod
    async def update_document(self, document: Document) -> None:
        """Update an existing document."""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and connections."""
        pass


class BaseReranker(ABC):
    """Abstract base class for reranking strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank search results for improved relevance."""
        pass
    
    @abstractmethod
    def get_reranker_type(self) -> RerankerType:
        """Return the type of reranker."""
        pass


class BaseRetriever(ABC):
    """
    Abstract base class for the retrieval system.
    
    Provides a unified interface for document indexing, embedding, retrieval,
    and reranking operations with pluggable backends.
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RetrieverConfig] = None
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.config = config or RetrieverConfig()
        self._initialized = False
        self._metrics = {}
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return
            
        await self.vectorstore.initialize()
        self._initialized = True
    
    @abstractmethod
    async def embed(self, text: str) -> Embedding:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    async def index(
        self, 
        documents: List[Union[str, Document]],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Index documents with batching and progress tracking."""
        pass
    
    @abstractmethod
    async def retrieve(self, query: SearchQuery) -> List[SearchResult]:
        """Retrieve relevant documents for a query."""
        pass
    
    @abstractmethod
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results if reranker is available."""
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[DocumentId]) -> Dict[str, Any]:
        """Delete documents by IDs."""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval system metrics."""
        return {
            "total_documents": await self.vectorstore.get_document_count(),
            "embedding_dimension": self.embedder.get_embedding_dimension(),
            "initialized": self._initialized,
            "config": self.config.__dict__,
            **self._metrics
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check embedder
            test_embedding = await self.embedder.embed_text("test")
            health["components"]["embedder"] = {
                "status": "healthy",
                "dimension": len(test_embedding)
            }
        except Exception as e:
            health["components"]["embedder"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        try:
            # Check vector store
            doc_count = await self.vectorstore.get_document_count()
            health["components"]["vectorstore"] = {
                "status": "healthy",
                "document_count": doc_count
            }
        except Exception as e:
            health["components"]["vectorstore"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        if self.reranker:
            health["components"]["reranker"] = {
                "status": "healthy",
                "type": self.reranker.get_reranker_type().value
            }
        
        return health
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        await self.vectorstore.cleanup()
        self._initialized = False


class RetrieverFactory:
    """Factory class for creating retriever instances with dependency injection."""
    
    @staticmethod
    def create_embedder(embedder_type: EmbedderType, config: Dict[str, Any]) -> BaseEmbedder:
        """Create embedder instance based on type."""
        # This will be implemented in concrete embedder classes
        raise NotImplementedError("Embedder factory not implemented")
    
    @staticmethod
    def create_vectorstore(vectorstore_type: VectorStoreType, config: Dict[str, Any]) -> BaseVectorStore:
        """Create vector store instance based on type."""
        # This will be implemented in concrete vectorstore classes
        raise NotImplementedError("VectorStore factory not implemented")
    
    @staticmethod
    def create_reranker(reranker_type: RerankerType, config: Dict[str, Any]) -> Optional[BaseReranker]:
        """Create reranker instance based on type."""
        if reranker_type == RerankerType.NONE:
            return None
        # This will be implemented in concrete reranker classes
        raise NotImplementedError("Reranker factory not implemented")


# Exception classes for better error handling
class RetrieverError(Exception):
    """Base exception for retriever operations."""
    pass


class EmbeddingError(RetrieverError):
    """Exception for embedding-related errors."""
    pass


class VectorStoreError(RetrieverError):
    """Exception for vector store operations."""
    pass


class RerankerError(RetrieverError):
    """Exception for reranking operations."""
    pass


class ConfigurationError(RetrieverError):
    """Exception for configuration-related errors."""
    pass
