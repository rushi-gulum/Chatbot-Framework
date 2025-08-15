"""
Enterprise Document Indexer for Chatbot Framework

This module provides document indexing capabilities with:
- Batch processing for efficiency
- Text chunking and preprocessing
- Parallel processing
- Progress tracking
- Error handling and retries
- Deduplication
"""

import asyncio
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Union, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from pathlib import Path
import re

from .base import Document, DocumentId, IndexingConfig, RetrieverError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class IndexingMetrics:
    """Metrics for indexing operations."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    processing_time_seconds: float = 0.0
    documents_per_second: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class TextChunk:
    """Represents a chunk of text from a larger document."""
    id: str
    content: str
    parent_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class TextChunker:
    """Advanced text chunking with overlap and semantic boundaries."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        respect_sentence_boundaries: bool = True,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.min_chunk_size = min_chunk_size
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_endings = re.compile(r'\n\s*\n')
    
    def chunk_text(self, text: str, document_id: Optional[str] = None) -> List[TextChunk]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [TextChunk(
                id=str(uuid.uuid4()),
                content=text.strip(),
                parent_id=document_id,
                chunk_index=0,
                total_chunks=1
            )]
        
        chunks = []
        
        if self.respect_sentence_boundaries:
            chunks = self._chunk_by_sentences(text, document_id)
        else:
            chunks = self._chunk_by_length(text, document_id)
        
        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, document_id: Optional[str]) -> List[TextChunk]:
        """Chunk text respecting sentence boundaries."""
        # Split into sentences
        sentences = self.sentence_endings.split(text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                # Create chunk with current content
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        id=str(uuid.uuid4()),
                        content=current_chunk.strip(),
                        parent_id=document_id,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(TextChunk(
                id=str(uuid.uuid4()),
                content=current_chunk.strip(),
                parent_id=document_id,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _chunk_by_length(self, text: str, document_id: Optional[str]) -> List[TextChunk]:
        """Chunk text by fixed length with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Find word boundary if not at end
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                    chunk_text = text[start:end]
            
            if chunk_text.strip() and len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    id=str(uuid.uuid4()),
                    content=chunk_text.strip(),
                    parent_id=document_id,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap, end)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.overlap:
            return text
        
        # Find word boundary for overlap
        overlap_start = len(text) - self.overlap
        space_pos = text.find(' ', overlap_start)
        
        if space_pos > 0:
            return text[space_pos + 1:]
        else:
            return text[overlap_start:]


class TextPreprocessor:
    """Text preprocessing for better indexing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.normalize_whitespace = config.get('normalize_whitespace', True)
        self.remove_special_chars = config.get('remove_special_chars', False)
        self.lowercase = config.get('lowercase', False)
        self.min_length = config.get('min_length', 10)
        self.max_length = config.get('max_length', 50000)
    
    def preprocess(self, text: str) -> str:
        """Apply preprocessing to text."""
        if not text or not text.strip():
            return ""
        
        processed = text
        
        # Normalize whitespace
        if self.normalize_whitespace:
            processed = re.sub(r'\s+', ' ', processed)
            processed = processed.strip()
        
        # Remove special characters (optional)
        if self.remove_special_chars:
            processed = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', processed)
            processed = re.sub(r'\s+', ' ', processed)
        
        # Convert to lowercase (optional)
        if self.lowercase:
            processed = processed.lower()
        
        # Length validation
        if len(processed) < self.min_length or len(processed) > self.max_length:
            return ""
        
        return processed.strip()


class DocumentIndexer:
    """Enterprise document indexer with batch processing and optimization."""
    
    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap
        )
        self.preprocessor = TextPreprocessor({})
        self.metrics = IndexingMetrics()
        self._document_hashes: Dict[str, str] = {}  # For deduplication
    
    async def index_documents(
        self,
        documents: List[Union[str, Document, Dict[str, Any]]],
        embedder,
        vectorstore,
        progress_callback: Optional[Callable[[float, IndexingMetrics], Awaitable[None]]] = None
    ) -> IndexingMetrics:
        """Index documents with batch processing and progress tracking."""
        start_time = time.time()
        self.metrics = IndexingMetrics()
        self.metrics.total_documents = len(documents)
        
        # Convert inputs to Document objects
        processed_docs = await self._prepare_documents(documents)
        
        # Process in batches
        batch_size = self.config.batch_size
        total_batches = (len(processed_docs) + batch_size - 1) // batch_size
        
        all_chunks = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(processed_docs))
            batch = processed_docs[start_idx:end_idx]
            
            try:
                # Process batch
                if self.config.enable_parallel:
                    batch_chunks = await self._process_batch_parallel(batch)
                else:
                    batch_chunks = await self._process_batch_sequential(batch)
                
                all_chunks.extend(batch_chunks)
                self.metrics.processed_documents += len(batch)
                
                # Update progress
                if progress_callback:
                    progress = (batch_idx + 1) / total_batches
                    await progress_callback(progress, self.metrics)
                
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
                
            except Exception as e:
                self.metrics.failed_documents += len(batch)
                self.metrics.errors.append(f"Batch {batch_idx + 1} failed: {str(e)}")
                logger.error(f"Failed to process batch {batch_idx + 1}: {e}")
        
        # Generate embeddings and store
        if all_chunks:
            await self._embed_and_store_chunks(all_chunks, embedder, vectorstore)
        
        # Update final metrics
        self.metrics.processing_time_seconds = time.time() - start_time
        if self.metrics.processing_time_seconds > 0:
            self.metrics.documents_per_second = (
                self.metrics.processed_documents / self.metrics.processing_time_seconds
            )
        
        logger.info(f"Indexing completed: {self.metrics.processed_documents}/{self.metrics.total_documents} documents")
        return self.metrics
    
    async def _prepare_documents(self, documents: List[Union[str, Document, Dict[str, Any]]]) -> List[Document]:
        """Convert various input formats to Document objects."""
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                if isinstance(doc, str):
                    # String content
                    document = Document(
                        id=str(uuid.uuid4()),
                        content=doc,
                        metadata={'source': 'text_input', 'index': i}
                    )
                elif isinstance(doc, dict):
                    # Dictionary format
                    document = Document(
                        id=doc.get('id', str(uuid.uuid4())),
                        content=doc.get('content', ''),
                        metadata=doc.get('metadata', {}),
                        access_level=doc.get('access_level', 'public')
                    )
                elif isinstance(doc, Document):
                    # Already a Document
                    document = doc
                else:
                    raise ValueError(f"Unsupported document type: {type(doc)}")
                
                # Validate content
                if not document.content or not document.content.strip():
                    logger.warning(f"Skipping document {document.id}: empty content")
                    continue
                
                # Check for duplicates
                if self.config.deduplicate:
                    content_hash = hashlib.sha256(document.content.encode()).hexdigest()
                    if content_hash in self._document_hashes:
                        logger.warning(f"Skipping duplicate document {document.id}")
                        continue
                    self._document_hashes[content_hash] = document.id
                
                processed_docs.append(document)
                
            except Exception as e:
                self.metrics.errors.append(f"Failed to prepare document {i}: {str(e)}")
                logger.error(f"Failed to prepare document {i}: {e}")
        
        return processed_docs
    
    async def _process_batch_parallel(self, documents: List[Document]) -> List[Document]:
        """Process batch of documents in parallel."""
        tasks = [self._process_single_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        chunks = []
        for result in results:
            if isinstance(result, Exception):
                self.metrics.errors.append(str(result))
            elif isinstance(result, list):
                chunks.extend(result)
        
        return chunks
    
    async def _process_batch_sequential(self, documents: List[Document]) -> List[Document]:
        """Process batch of documents sequentially."""
        chunks = []
        for doc in documents:
            try:
                doc_chunks = await self._process_single_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                self.metrics.errors.append(f"Failed to process document {doc.id}: {str(e)}")
        
        return chunks
    
    async def _process_single_document(self, document: Document) -> List[Document]:
        """Process a single document into chunks."""
        # Preprocess content
        processed_content = self.preprocessor.preprocess(document.content)
        if not processed_content:
            logger.warning(f"Document {document.id} has no valid content after preprocessing")
            return []
        
        # Create chunks
        text_chunks = self.chunker.chunk_text(processed_content, document.id)
        
        # Convert to Document objects
        chunk_documents = []
        for chunk in text_chunks:
            chunk_doc = Document(
                id=chunk.id,
                content=chunk.content,
                metadata={
                    **document.metadata,
                    'parent_id': document.id,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'is_chunk': True
                },
                access_level=document.access_level
            )
            chunk_documents.append(chunk_doc)
        
        self.metrics.total_chunks += len(chunk_documents)
        return chunk_documents
    
    async def _embed_and_store_chunks(self, chunks: List[Document], embedder, vectorstore):
        """Generate embeddings and store chunks in vector store."""
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embedded_chunks = await embedder.embed_documents(chunks)
            
            # Validate embeddings
            if self.config.validate_embeddings:
                embedded_chunks = [doc for doc in embedded_chunks if doc.embedding is not None]
                if len(embedded_chunks) < len(chunks):
                    logger.warning(f"Some embeddings failed: {len(embedded_chunks)}/{len(chunks)}")
            
            # Store in vector store
            logger.info(f"Storing {len(embedded_chunks)} chunks in vector store")
            await vectorstore.add_documents(embedded_chunks)
            
        except Exception as e:
            self.metrics.errors.append(f"Failed to embed/store chunks: {str(e)}")
            raise RetrieverError(f"Failed to embed and store chunks: {e}")


class IndexingProgress:
    """Helper class for tracking indexing progress."""
    
    def __init__(self):
        self.current_progress = 0.0
        self.current_metrics: Optional[IndexingMetrics] = None
        self.callbacks: List[Callable[[float, IndexingMetrics], Awaitable[None]]] = []
    
    def add_callback(self, callback: Callable[[float, IndexingMetrics], Awaitable[None]]):
        """Add progress callback."""
        self.callbacks.append(callback)
    
    async def update(self, progress: float, metrics: IndexingMetrics):
        """Update progress and notify callbacks."""
        self.current_progress = progress
        self.current_metrics = metrics
        
        for callback in self.callbacks:
            try:
                await callback(progress, metrics)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")


# Utility functions
def estimate_indexing_time(
    document_count: int,
    avg_document_size: int,
    processing_rate: float = 10.0  # docs per second
) -> float:
    """Estimate indexing time in seconds."""
    return document_count / processing_rate


def calculate_optimal_batch_size(
    total_documents: int,
    available_memory_mb: int = 1024,
    avg_embedding_size: int = 1536
) -> int:
    """Calculate optimal batch size based on available memory."""
    # Rough estimation: embedding size * 4 bytes per float + overhead
    memory_per_doc = avg_embedding_size * 4 + 1024  # bytes
    max_docs_in_memory = (available_memory_mb * 1024 * 1024) // memory_per_doc
    
    # Conservative batch size
    optimal_batch = min(max_docs_in_memory // 4, 100, total_documents)
    return max(optimal_batch, 1)