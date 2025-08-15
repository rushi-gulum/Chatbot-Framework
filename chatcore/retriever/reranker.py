"""
Enterprise Reranker Implementation for Chatbot Framework

This module provides reranking strategies for improving search relevance:
- Cross-encoder reranking
- LLM-based reranking
- Semantic similarity reranking
- Hybrid reranking approaches
- Performance optimization
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .base import BaseReranker, SearchResult, RerankerType, RerankerError
from .vectorstore import BaseVectorStore as VectorStoreBaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RerankerMetrics:
    """Metrics for reranking operations."""
    total_requests: int = 0
    total_results_reranked: int = 0
    avg_latency_ms: float = 0.0
    improvement_score: float = 0.0  # Average improvement in ranking quality
    error_count: int = 0


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranking for improved relevance."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.batch_size = config.get('batch_size', 32)
        self.max_length = config.get('max_length', 512)
        self.device = config.get('device', 'cpu')
        self.top_k = config.get('top_k', 100)  # Max results to rerank
        self.metrics = RerankerMetrics()
        
        # Initialize model
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
        except ImportError:
            raise RerankerError("sentence-transformers package is required for CrossEncoder")
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using cross-encoder."""
        if not results:
            return results
        
        start_time = time.time()
        
        try:
            # Limit number of results to rerank for performance
            results_to_rerank = results[:min(len(results), self.top_k)]
            
            # Prepare query-document pairs
            pairs = []
            for result in results_to_rerank:
                pairs.append([query, result.document.content])
            
            # Get cross-encoder scores
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self.model.predict(pairs, batch_size=self.batch_size)
            )
            
            # Update results with reranker scores
            for i, (result, score) in enumerate(zip(results_to_rerank, scores)):
                result.rerank_score = float(score)
            
            # Sort by reranker score (handle None values)
            reranked_results = sorted(
                results_to_rerank, 
                key=lambda x: x.rerank_score or 0.0, 
                reverse=True
            )
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            # Add remaining results (if any) at the end
            if len(results) > self.top_k:
                remaining_results = results[self.top_k:]
                for i, result in enumerate(remaining_results):
                    result.rank = len(reranked_results) + i + 1
                reranked_results.extend(remaining_results)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_results_reranked += len(results_to_rerank)
            
            latency = (time.time() - start_time) * 1000
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (self.metrics.total_requests - 1) + latency) 
                / self.metrics.total_requests
            )
            
            return reranked_results
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Return original results on error
            return results
    
    def get_reranker_type(self) -> RerankerType:
        """Return reranker type."""
        return RerankerType.CROSS_ENCODER


class LLMReranker(BaseReranker):
    """LLM-based reranking using language models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_provider = config.get('model_provider', 'openai')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.api_key = config.get('api_key')
        self.max_results = config.get('max_results', 20)  # Limit for cost control
        self.batch_size = config.get('batch_size', 5)
        self.temperature = config.get('temperature', 0.1)
        self.metrics = RerankerMetrics()
        
        # Initialize LLM client
        if self.model_provider == 'openai':
            if not self.api_key:
                raise RerankerError("OpenAI API key is required")
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise RerankerError("openai package is required")
        else:
            raise RerankerError(f"Unsupported model provider: {self.model_provider}")
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using LLM."""
        if not results:
            return results
        
        start_time = time.time()
        
        try:
            # Limit results for cost control
            results_to_rerank = results[:min(len(results), self.max_results)]
            
            # Process in batches
            reranked_results = []
            for i in range(0, len(results_to_rerank), self.batch_size):
                batch = results_to_rerank[i:i + self.batch_size]
                batch_scores = await self._rerank_batch(query, batch)
                
                for result, score in zip(batch, batch_scores):
                    result.rerank_score = score
                
                reranked_results.extend(batch)
            
            # Sort by LLM scores
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            # Add remaining results
            if len(results) > self.max_results:
                remaining_results = results[self.max_results:]
                for i, result in enumerate(remaining_results):
                    result.rank = len(reranked_results) + i + 1
                reranked_results.extend(remaining_results)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_results_reranked += len(results_to_rerank)
            
            latency = (time.time() - start_time) * 1000
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (self.metrics.total_requests - 1) + latency) 
                / self.metrics.total_requests
            )
            
            return reranked_results
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"LLM reranking failed: {e}")
            return results
    
    async def _rerank_batch(self, query: str, results: List[SearchResult]) -> List[float]:
        """Rerank a batch of results using LLM."""
        # Prepare prompt
        documents_text = ""
        for i, result in enumerate(results):
            content = result.document.content[:500]  # Limit content length
            documents_text += f"\nDocument {i+1}: {content}\n"
        
        prompt = f"""
Task: Rank the following documents by relevance to the query.

Query: {query}

Documents:{documents_text}

Instructions:
1. Rate each document's relevance to the query on a scale of 0-10
2. Consider semantic relevance, context, and specificity
3. Return only a JSON array of scores in order: [score1, score2, ...]

Scores:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=100
            )
            
            # Parse response (handle None content)
            content = response.choices[0].message.content
            scores_text = content.strip() if content else "[]"
            scores = json.loads(scores_text)
            
            # Validate scores
            if len(scores) != len(results):
                logger.warning("LLM returned wrong number of scores, using similarity scores")
                return [result.similarity_score for result in results]
            
            # Normalize scores to 0-1 range
            normalized_scores = [max(0, min(10, score)) / 10.0 for score in scores]
            return normalized_scores
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            # Fallback to similarity scores
            return [result.similarity_score for result in results]
    
    def get_reranker_type(self) -> RerankerType:
        """Return reranker type."""
        return RerankerType.LLM_BASED


class SemanticReranker(BaseReranker):
    """Semantic similarity-based reranking."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.query_weight = config.get('query_weight', 0.7)
        self.content_weight = config.get('content_weight', 0.3)
        self.use_metadata = config.get('use_metadata', True)
        self.metadata_boost = config.get('metadata_boost', 0.1)
        self.metrics = RerankerMetrics()
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using semantic features."""
        if not results:
            return results
        
        start_time = time.time()
        
        try:
            # Calculate enhanced scores
            for result in results:
                enhanced_score = await self._calculate_enhanced_score(query, result)
                result.rerank_score = enhanced_score
            
            # Sort by enhanced score (handle None values)
            reranked_results = sorted(results, key=lambda x: x.rerank_score or 0.0, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_results_reranked += len(results)
            
            latency = (time.time() - start_time) * 1000
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (self.metrics.total_requests - 1) + latency) 
                / self.metrics.total_requests
            )
            
            return reranked_results
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Semantic reranking failed: {e}")
            return results
    
    async def _calculate_enhanced_score(self, query: str, result: SearchResult) -> float:
        """Calculate enhanced relevance score."""
        base_score = result.similarity_score
        
        # Query term matching
        query_terms = set(query.lower().split())
        content_terms = set(result.document.content.lower().split())
        term_overlap = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
        
        # Content length penalty (prefer appropriate length)
        content_length = len(result.document.content)
        length_factor = 1.0
        if content_length < 50:
            length_factor = 0.8  # Too short
        elif content_length > 2000:
            length_factor = 0.9  # Too long
        
        # Metadata boost
        metadata_factor = 1.0
        if self.use_metadata and result.document.metadata:
            # Boost if query terms appear in metadata
            metadata_text = ' '.join(str(v) for v in result.document.metadata.values()).lower()
            metadata_overlap = len(query_terms.intersection(set(metadata_text.split())))
            if metadata_overlap > 0:
                metadata_factor = 1.0 + self.metadata_boost
        
        # Combine factors
        enhanced_score = (
            base_score * self.query_weight +
            term_overlap * self.content_weight
        ) * length_factor * metadata_factor
        
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def get_reranker_type(self) -> RerankerType:
        """Return reranker type."""
        return RerankerType.SEMANTIC


class HybridReranker(BaseReranker):
    """Hybrid reranker combining multiple strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rerankers: List[BaseReranker] = []
        self.weights: List[float] = config.get('weights', [0.5, 0.3, 0.2])
        self.metrics = RerankerMetrics()
        
        # Initialize sub-rerankers
        reranker_configs = config.get('rerankers', [])
        for reranker_config in reranker_configs:
            reranker_type = reranker_config.get('type')
            if reranker_type == 'cross_encoder':
                self.rerankers.append(CrossEncoderReranker(reranker_config))
            elif reranker_type == 'llm':
                self.rerankers.append(LLMReranker(reranker_config))
            elif reranker_type == 'semantic':
                self.rerankers.append(SemanticReranker(reranker_config))
        
        if not self.rerankers:
            # Default configuration
            self.rerankers = [
                CrossEncoderReranker(config.get('cross_encoder', {})),
                SemanticReranker(config.get('semantic', {}))
            ]
            self.weights = [0.7, 0.3]
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using hybrid approach."""
        if not results:
            return results
        
        start_time = time.time()
        
        try:
            # Get scores from all rerankers
            all_scores = []
            for reranker in self.rerankers:
                reranked_results = await reranker.rerank(query, results.copy())
                scores = [r.rerank_score if r.rerank_score is not None else r.similarity_score 
                         for r in reranked_results]
                all_scores.append(scores)
            
            # Combine scores with weights
            combined_scores = []
            for i in range(len(results)):
                weighted_score = 0.0
                total_weight = 0.0
                
                for j, scores in enumerate(all_scores):
                    if i < len(scores):
                        weight = self.weights[j] if j < len(self.weights) else 1.0
                        weighted_score += scores[i] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    combined_scores.append(weighted_score / total_weight)
                else:
                    combined_scores.append(results[i].similarity_score)
            
            # Update results with combined scores
            for result, score in zip(results, combined_scores):
                result.rerank_score = score
            
            # Sort by combined score (handle None values)
            reranked_results = sorted(results, key=lambda x: x.rerank_score or 0.0, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.total_results_reranked += len(results)
            
            latency = (time.time() - start_time) * 1000
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (self.metrics.total_requests - 1) + latency) 
                / self.metrics.total_requests
            )
            
            return reranked_results
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Hybrid reranking failed: {e}")
            return results
    
    def get_reranker_type(self) -> RerankerType:
        """Return reranker type."""
        return RerankerType.SEMANTIC  # Represents hybrid approach


class RerankerFactory:
    """Factory for creating reranker instances."""
    
    @staticmethod
    def create_reranker(reranker_type: RerankerType, config: Dict[str, Any]) -> Optional[BaseReranker]:
        """Create reranker instance based on type."""
        if reranker_type == RerankerType.NONE:
            return None
        
        reranker_map = {
            RerankerType.CROSS_ENCODER: CrossEncoderReranker,
            RerankerType.LLM_BASED: LLMReranker,
            RerankerType.SEMANTIC: SemanticReranker,
        }
        
        # Check for hybrid configuration
        if 'rerankers' in config:
            return HybridReranker(config)
        
        if reranker_type not in reranker_map:
            raise RerankerError(f"Unsupported reranker type: {reranker_type}")
        
        reranker_class = reranker_map[reranker_type]
        return reranker_class(config)


# Utility functions for reranking evaluation
def calculate_ndcg(relevance_scores: List[float], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if not relevance_scores:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, score in enumerate(relevance_scores[:k]):
        if i == 0:
            dcg += score
        else:
            dcg += score / (1 + i)  # Simplified NDCG formula
    
    # Calculate IDCG (perfect ranking)
    sorted_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(sorted_scores[:k]):
        if i == 0:
            idcg += score
        else:
            idcg += score / (1 + i)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_map(relevance_scores: List[float], threshold: float = 0.5) -> float:
    """Calculate Mean Average Precision."""
    if not relevance_scores:
        return 0.0
    
    relevant_positions = []
    for i, score in enumerate(relevance_scores):
        if score >= threshold:
            relevant_positions.append(i + 1)
    
    if not relevant_positions:
        return 0.0
    
    precision_sum = 0.0
    for i, pos in enumerate(relevant_positions):
        precision_at_k = (i + 1) / pos
        precision_sum += precision_at_k
    
    return precision_sum / len(relevant_positions)


# Example usage and testing functions
async def create_retriever(
    embedder_type: str = "openai",
    vectorstore_type: str = "faiss", 
    config_path: Optional[str] = None
):
    """
    Create retriever with default settings.
    
    This is the function referenced in the original reranker.py example.
    """
    from .base import RetrieverConfig, EmbedderType, VectorStoreType, RerankerType
    from .embedder import EmbedderFactory
    from .vectorstore import VectorStoreFactory
    
    # Load configuration
    config = RetrieverConfig()
    if config_path:
        # TODO: Load from config file
        pass
    
    # Create components
    embedder_config = config.backend_config.get('embedder', {})
    embedder = EmbedderFactory.create_embedder(
        EmbedderType(embedder_type), 
        embedder_config
    )
    
    vectorstore_config = config.backend_config.get('vectorstore', {})
    vectorstore = VectorStoreFactory.create_vectorstore(
        VectorStoreType(vectorstore_type),
        vectorstore_config
    )
    
    reranker_config = config.backend_config.get('reranker', {})
    reranker = RerankerFactory.create_reranker(
        config.reranker_type,
        reranker_config
    )
    
    # Create retriever implementation
    from .retriever import EnterpriseRetriever  # Will be implemented next
    
    # Type cast to resolve the conflict temporarily
    from typing import cast
    from .base import BaseVectorStore as BaseVS
    
    retriever = EnterpriseRetriever(
        embedder=embedder,
        vectorstore=cast(BaseVS, vectorstore),
        reranker=reranker,
        config=config
    )
    
    await retriever.initialize()
    return retriever