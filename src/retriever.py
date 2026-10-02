"""Retriever module for semantic search over vector store chunks.

Takes user queries, generates query embeddings, performs ChromaDB similarity search,
and filters results based on a configurable relevance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from src.embeddings import EmbeddingService
from src.text_splitter import TextChunk
from src.vector_store import VectorStore
from config.settings import DEFAULT_TOP_K, DEFAULT_RELEVANCE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Encapsulates a retrieved chunk alongside its similarity score."""
    chunk: TextChunk
    score: float
    is_relevant: bool

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page_number(self) -> int | None:
        return self.chunk.page_number

    @property
    def content(self) -> str:
        return self.chunk.content


class Retriever:
    """Coordinates query embedding, ChromaDB search, and relevance threshold filtering."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        relevance_threshold: float | None = None,
        filter_document: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a user query.

        Args:
            query: Natural language question or search phrase.
            top_k: Optional override for number of chunks to fetch.
            relevance_threshold: Optional override for minimum similarity score.
            filter_document: Optional filename to constrain search to a single document.

        Returns:
            List of RetrievalResult objects meeting the relevance threshold.
        """
        k = top_k if top_k is not None else self.top_k
        threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else self.relevance_threshold
        )

        query = query.strip()
        if not query or self.vector_store.count() == 0:
            return []

        # Generate query embedding
        query_vec = self.embedding_service.embed_query(query)

        # Retrieve extra candidates if filtering by document
        fetch_k = k * 3 if filter_document else k
        raw_results = self.vector_store.search(query_vec, top_k=fetch_k)

        filtered_results: list[RetrievalResult] = []
        for chunk, score in raw_results:
            if filter_document and chunk.source != filter_document:
                continue

            # Note: Cosine similarity ranges from -1.0 to 1.0 (typical semantic match > 0.35)
            # Scores are NOT calibrated probabilities.
            is_rel = score >= threshold
            if is_rel:
                filtered_results.append(
                    RetrievalResult(chunk=chunk, score=score, is_relevant=True)
                )

            if len(filtered_results) >= k:
                break

        logger.info(
            "Retrieved %d relevant chunks (threshold: %.2f) for query: '%s'",
            len(filtered_results),
            threshold,
            query[:40],
        )

        return filtered_results
