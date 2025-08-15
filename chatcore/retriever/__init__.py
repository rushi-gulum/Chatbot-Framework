"""
Enterprise Retriever Module for Chatbot Framework

This module provides a comprehensive retrieval system with:
- Multiple embedding models (OpenAI, HuggingFace, Cohere)
- Multiple vector stores (FAISS, Pinecone, ChromaDB, Milvus, Weaviate)
- Advanced reranking strategies
- Batch document indexing
- Security and access control
- Performance optimization
- Comprehensive metrics
"""

from .base import (
    BaseRetriever,
    BaseEmbedder,
    BaseVectorStore,
    BaseReranker,
    Document,
    SearchQuery,
    SearchResult,
    RetrieverConfig,
    IndexingConfig,
    VectorStoreType,
    EmbedderType,
    RerankerType,
    RetrieverError,
    EmbeddingError,
    VectorStoreError,
    RerankerError,
    ConfigurationError
)

from .embedder import (
    OpenAIEmbedder,
    HuggingFaceEmbedder,
    CohereEmbedder,
    EmbedderFactory,
    cosine_similarity,
    normalize_embedding
)

from .vectorstore import (
    FAISSVectorStore,
    PineconeVectorStore,
    ChromaVectorStore,
    VectorStoreFactory
)

from .reranker import (
    CrossEncoderReranker,
    LLMReranker,
    SemanticReranker,
    HybridReranker,
    RerankerFactory,
    calculate_ndcg,
    calculate_map
)

from .indexer import (
    DocumentIndexer,
    TextChunker,
    TextPreprocessor,
    IndexingMetrics,
    IndexingProgress,
    estimate_indexing_time,
    calculate_optimal_batch_size
)

from .retriever import (
    EnterpriseRetriever,
    create_retriever,
    create_development_retriever,
    create_production_retriever
)

# Main public interface
__all__ = [
    # Main classes
    'EnterpriseRetriever',
    'Document',
    'SearchQuery',
    'SearchResult',
    'RetrieverConfig',
    
    # Factory functions
    'create_retriever',
    'create_development_retriever',
    'create_production_retriever',
    
    # Base classes
    'BaseRetriever',
    'BaseEmbedder',
    'BaseVectorStore',
    'BaseReranker',
    
    # Concrete implementations
    'OpenAIEmbedder',
    'HuggingFaceEmbedder',
    'CohereEmbedder',
    'FAISSVectorStore',
    'PineconeVectorStore',
    'ChromaVectorStore',
    'CrossEncoderReranker',
    'LLMReranker',
    'SemanticReranker',
    'HybridReranker',
    
    # Indexing
    'DocumentIndexer',
    'IndexingMetrics',
    'IndexingConfig',
    
    # Enums
    'VectorStoreType',
    'EmbedderType',
    'RerankerType',
    
    # Exceptions
    'RetrieverError',
    'EmbeddingError',
    'VectorStoreError',
    'RerankerError',
    'ConfigurationError',
    
    # Utility functions
    'cosine_similarity',
    'normalize_embedding',
    'calculate_ndcg',
    'calculate_map',
    'estimate_indexing_time',
    'calculate_optimal_batch_size'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Chatbot Framework Team'
__email__ = 'support@chatbotframework.com'

# Module-level convenience function for backward compatibility
async def create_simple_retriever(**kwargs):
    """
    Simple wrapper for creating a retriever with sensible defaults.
    
    This maintains compatibility with the original simple interface
    while providing access to the full enterprise features.
    """
    return await create_retriever(**kwargs)
