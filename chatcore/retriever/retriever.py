"""
Enterprise Retriever Implementation for Chatbot Framework

This module provides the main retriever implementation that combines:
- Embedder for text to vector conversion
- Vector store for storage and similarity search
- Reranker for improving relevance
- Indexer for document processing
- Security and access control
- Metrics and monitoring
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
from datetime import datetime

from .base import (
    BaseRetriever, BaseEmbedder, BaseVectorStore, BaseReranker,
    Document, DocumentId, SearchQuery, SearchResult, RetrieverConfig,
    Embedding, RetrieverError, EmbedderType, VectorStoreType, RerankerType
)
from .embedder import EmbedderFactory
from .vectorstore import VectorStoreFactory
from .reranker import RerankerFactory
from .indexer import DocumentIndexer, IndexingMetrics

# Configure logging
logger = logging.getLogger(__name__)


class EnterpriseRetriever(BaseRetriever):
    """
    Production-ready retriever implementation with enterprise features.
    
    Features:
    - Pluggable backends (embedders, vector stores, rerankers)
    - Batch processing and optimization
    - Security and access control
    - Comprehensive metrics and monitoring
    - Error handling and retries
    - Configuration management
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RetrieverConfig] = None
    ):
        super().__init__(embedder, vectorstore, reranker, config)
        self.indexer = DocumentIndexer(self.config.indexing_config)
        self._security_enabled = self.config.enable_access_control
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        
        # Rate limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        logger.info(f"Initialized EnterpriseRetriever with {type(embedder).__name__}, "
                   f"{type(vectorstore).__name__}, {type(reranker).__name__ if reranker else 'No reranker'}")
    
    async def embed(self, text: str) -> Embedding:
        """Generate embedding for text with error handling."""
        if not text or not text.strip():
            raise RetrieverError("Text cannot be empty")
        
        try:
            async with self._semaphore:
                start_time = time.time()
                embedding = await self.embedder.embed_text(text)
                
                # Update metrics
                self._request_count += 1
                self._total_latency += time.time() - start_time
                
                return embedding
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Embedding failed: {e}")
            raise RetrieverError(f"Failed to generate embedding: {e}")
    
    async def index(
        self, 
        documents: List[Union[str, Document, Dict[str, Any]]],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Index documents with comprehensive processing."""
        if not documents:
            return {"status": "success", "indexed_count": 0}
        
        start_time = time.time()
        
        try:
            # Override batch size if provided
            if batch_size:
                original_batch_size = self.config.indexing_config.batch_size
                self.config.indexing_config.batch_size = batch_size
            
            # Process documents through indexer
            metrics = await self.indexer.index_documents(
                documents=documents,
                embedder=self.embedder,
                vectorstore=self.vectorstore
            )
            
            # Restore original batch size
            if batch_size:
                self.config.indexing_config.batch_size = original_batch_size
            
            # Update retriever metrics
            self._update_metrics("index", time.time() - start_time, len(documents))
            
            return {
                "status": "success",
                "indexed_count": metrics.processed_documents,
                "failed_count": metrics.failed_documents,
                "total_chunks": metrics.total_chunks,
                "processing_time": metrics.processing_time_seconds,
                "errors": metrics.errors[:10]  # Limit error list
            }
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Indexing failed: {e}")
            raise RetrieverError(f"Failed to index documents: {e}")
    
    async def retrieve(self, query: SearchQuery) -> List[SearchResult]:
        """Retrieve relevant documents with reranking."""
        if not query.text or not query.text.strip():
            raise RetrieverError("Query text cannot be empty")
        
        start_time = time.time()
        
        try:
            async with self._semaphore:
                # Generate query embedding
                query_embedding = await self.embedder.embed_text(query.text)
                
                # Security check
                if self._security_enabled and not self._check_access(query):
                    logger.warning(f"Access denied for query: {query.text[:50]}...")
                    return []
                
                # Search vector store
                search_start = time.time()
                raw_results = await self.vectorstore.search(
                    query_embedding=query_embedding,
                    top_k=query.top_k * 2,  # Get more results for reranking
                    filters=query.filters
                )
                search_time = (time.time() - search_start) * 1000
                
                # Convert to SearchResult objects
                search_results = []
                for i, (document, score) in enumerate(raw_results):
                    # Security filter
                    if self._security_enabled and not self._check_document_access(document, query):
                        continue
                    
                    result = SearchResult(
                        document=document,
                        similarity_score=score,
                        rank=i + 1,
                        retrieval_time_ms=search_time
                    )
                    search_results.append(result)
                
                # Apply similarity threshold
                if query.similarity_threshold > 0:
                    search_results = [
                        r for r in search_results 
                        if r.similarity_score >= query.similarity_threshold
                    ]
                
                # Rerank if enabled and reranker available
                if query.rerank and self.reranker and search_results:
                    rerank_start = time.time()
                    search_results = await self.rerank(query.text, search_results)
                    rerank_time = (time.time() - rerank_start) * 1000
                    
                    # Update rerank timing
                    for result in search_results:
                        result.retrieval_time_ms += rerank_time
                
                # Limit to requested top_k
                search_results = search_results[:query.top_k]
                
                # Add explanations if requested
                if query.include_metadata:
                    for result in search_results:
                        result.explanation = self._generate_explanation(query, result)
                
                # Update metrics
                total_time = time.time() - start_time
                self._update_metrics("retrieve", total_time, len(search_results))
                
                logger.debug(f"Retrieved {len(search_results)} results for query: {query.text[:50]}...")
                return search_results
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Retrieval failed: {e}")
            raise RetrieverError(f"Failed to retrieve documents: {e}")
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results if reranker is available."""
        if not self.reranker or not results:
            return results
        
        try:
            return await self.reranker.rerank(query, results)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results on reranking failure
            return results
    
    async def delete(self, document_ids: List[DocumentId]) -> Dict[str, Any]:
        """Delete documents by IDs."""
        if not document_ids:
            return {"status": "success", "deleted_count": 0}
        
        start_time = time.time()
        
        try:
            # Get current count for metrics
            initial_count = await self.vectorstore.get_document_count()
            
            # Delete from vector store
            await self.vectorstore.delete_documents(document_ids)
            
            # Verify deletion
            final_count = await self.vectorstore.get_document_count()
            actual_deleted = initial_count - final_count
            
            # Update metrics
            self._update_metrics("delete", time.time() - start_time, actual_deleted)
            
            return {
                "status": "success",
                "requested_count": len(document_ids),
                "deleted_count": actual_deleted,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Deletion failed: {e}")
            raise RetrieverError(f"Failed to delete documents: {e}")
    
    def _check_access(self, query: SearchQuery) -> bool:
        """Check if query is allowed based on security settings."""
        if not self._security_enabled:
            return True
        
        # Implement access control logic
        # For now, just check access level
        return query.access_level in ["public", "internal", "restricted"]
    
    def _check_document_access(self, document: Document, query: SearchQuery) -> bool:
        """Check if document can be accessed by the query."""
        if not self._security_enabled:
            return True
        
        # Simple access level check
        access_levels = {
            "public": 0,
            "internal": 1,
            "restricted": 2,
            "confidential": 3
        }
        
        doc_level = access_levels.get(document.access_level, 0)
        query_level = access_levels.get(query.access_level, 0)
        
        return query_level >= doc_level
    
    def _generate_explanation(self, query: SearchQuery, result: SearchResult) -> str:
        """Generate explanation for why this result was returned."""
        explanation_parts = []
        
        # Similarity score
        explanation_parts.append(f"Similarity: {result.similarity_score:.3f}")
        
        # Rerank score if available
        if result.rerank_score is not None:
            explanation_parts.append(f"Rerank: {result.rerank_score:.3f}")
        
        # Query term matches
        query_terms = set(query.text.lower().split())
        content_terms = set(result.document.content.lower().split())
        matches = query_terms.intersection(content_terms)
        if matches:
            explanation_parts.append(f"Matches: {', '.join(list(matches)[:3])}")
        
        return " | ".join(explanation_parts)
    
    def _update_metrics(self, operation: str, duration: float, count: int):
        """Update internal metrics."""
        self._metrics[f"{operation}_count"] = self._metrics.get(f"{operation}_count", 0) + 1
        self._metrics[f"{operation}_total_time"] = self._metrics.get(f"{operation}_total_time", 0.0) + duration
        self._metrics[f"{operation}_total_items"] = self._metrics.get(f"{operation}_total_items", 0) + count
        
        # Calculate average latency
        if self._metrics[f"{operation}_count"] > 0:
            self._metrics[f"{operation}_avg_latency"] = (
                self._metrics[f"{operation}_total_time"] / self._metrics[f"{operation}_count"]
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval system metrics."""
        base_metrics = await super().get_metrics()
        
        # Add enterprise-specific metrics
        enterprise_metrics = {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "avg_latency_seconds": self._total_latency / max(self._request_count, 1),
            "concurrent_limit": self.config.max_concurrent_requests,
            "security_enabled": self._security_enabled,
        }
        
        # Add component metrics
        if hasattr(self.embedder, 'metrics'):
            try:
                enterprise_metrics["embedder_metrics"] = asdict(getattr(self.embedder, 'metrics'))
            except Exception:
                pass
        
        if self.reranker and hasattr(self.reranker, 'metrics'):
            try:
                enterprise_metrics["reranker_metrics"] = asdict(getattr(self.reranker, 'metrics'))
            except Exception:
                pass
        
        if hasattr(self.indexer, 'metrics'):
            try:
                enterprise_metrics["indexer_metrics"] = asdict(getattr(self.indexer, 'metrics'))
            except Exception:
                pass
        
        return {**base_metrics, **enterprise_metrics, **self._metrics}
    
    async def update_configuration(self, new_config: RetrieverConfig) -> None:
        """Update retriever configuration at runtime."""
        old_config = self.config
        self.config = new_config
        
        # Update semaphore if max_concurrent_requests changed
        if old_config.max_concurrent_requests != new_config.max_concurrent_requests:
            self._semaphore = asyncio.Semaphore(new_config.max_concurrent_requests)
        
        # Update indexer config
        self.indexer.config = new_config.indexing_config
        
        logger.info("Configuration updated successfully")
    
    async def rebuild_index(self, progress_callback=None) -> Dict[str, Any]:
        """Rebuild the entire index (useful for maintenance)."""
        logger.info("Starting index rebuild...")
        
        try:
            # Get current document count
            initial_count = await self.vectorstore.get_document_count()
            
            # Note: This is a placeholder implementation
            # In a real system, you'd need to maintain a source of truth
            # for all documents to enable rebuilding
            
            return {
                "status": "completed",
                "initial_count": initial_count,
                "message": "Index rebuild not implemented - requires document source of truth"
            }
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            raise RetrieverError(f"Failed to rebuild index: {e}")


# Factory function for creating retriever instances
async def create_retriever(
    embedder_type: str = "sentence_transformers",
    vectorstore_type: str = "faiss",
    reranker_type: str = "cross_encoder",
    config: Optional[Dict[str, Any]] = None
) -> EnterpriseRetriever:
    """
    Create a configured enterprise retriever instance.
    
    Args:
        embedder_type: Type of embedder ("openai", "huggingface", "cohere", etc.)
        vectorstore_type: Type of vector store ("faiss", "pinecone", "chroma", etc.)
        reranker_type: Type of reranker ("cross_encoder", "llm", "semantic", "none")
        config: Optional configuration dictionary
    
    Returns:
        Configured EnterpriseRetriever instance
    """
    # Create configuration
    retriever_config = RetrieverConfig()
    if config:
        # Update config with provided values
        for key, value in config.items():
            if hasattr(retriever_config, key):
                setattr(retriever_config, key, value)
    
    # Set component types
    retriever_config.embedder_type = EmbedderType(embedder_type)
    retriever_config.vectorstore_type = VectorStoreType(vectorstore_type)
    retriever_config.reranker_type = RerankerType(reranker_type)
    
    # Create components
    embedder_config = config.get('embedder', {}) if config else {}
    embedder = EmbedderFactory.create_embedder(
        retriever_config.embedder_type,
        embedder_config
    )
    
    vectorstore_config = config.get('vectorstore', {}) if config else {}
    vectorstore = VectorStoreFactory.create_vectorstore(
        retriever_config.vectorstore_type,
        vectorstore_config
    )
    
    reranker_config = config.get('reranker', {}) if config else {}
    reranker = RerankerFactory.create_reranker(
        retriever_config.reranker_type,
        reranker_config
    )
    
    # Create retriever
    retriever = EnterpriseRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        reranker=reranker,
        config=retriever_config
    )
    
    # Initialize
    await retriever.initialize()
    
    logger.info(f"Created enterprise retriever: {embedder_type}/{vectorstore_type}/{reranker_type}")
    return retriever


# Convenience functions for common configurations
async def create_development_retriever(config: Optional[Dict[str, Any]] = None) -> EnterpriseRetriever:
    """Create retriever optimized for development."""
    dev_config = {
        "max_concurrent_requests": 5,
        "enable_access_control": False,
        "embedder": {"cache_size": 100},
        "vectorstore": {"persist_directory": "./dev_vectorstore"},
        "indexing_config": {"batch_size": 10}
    }
    if config:
        dev_config.update(config)
    
    return await create_retriever(
        embedder_type="sentence_transformers",
        vectorstore_type="chroma",
        reranker_type="semantic",
        config=dev_config
    )


async def create_production_retriever(config: Dict[str, Any]) -> EnterpriseRetriever:
    """Create retriever optimized for production."""
    prod_config = {
        "max_concurrent_requests": 50,
        "enable_access_control": True,
        "audit_logging": True,
        "embedder": {"cache_size": 10000},
        "indexing_config": {"batch_size": 100, "enable_parallel": True}
    }
    prod_config.update(config)
    
    return await create_retriever(
        embedder_type=config.get("embedder_type", "openai"),
        vectorstore_type=config.get("vectorstore_type", "pinecone"),
        reranker_type=config.get("reranker_type", "cross_encoder"),
        config=prod_config
    )
