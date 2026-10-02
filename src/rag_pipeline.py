"""End-to-end Retrieval-Augmented Generation (RAG) pipeline orchestrator.

Coordinates document ingestion, chunking, embedding generation, vector indexing,
relevance-filtered retrieval, prompt construction with context limits, and Groq LLM invocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_CONTEXT_CHARS,
)
from src.document_loader import Document, DocumentLoader, DocumentParsingError
from src.embeddings import EmbeddingService
from src.llm_service import LLMService, UNKNOWN_ANSWER_MESSAGE
from src.retriever import Retriever
from src.text_splitter import RecursiveTextSplitter, TextChunk
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Encapsulates the response from the RAG pipeline."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    context_used: str = ""
    has_sufficient_context: bool = True


class RAGPipeline:
    """Coordinates all components of the Smart Document Assistant."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.vector_store = vector_store or VectorStore(dimension=self.embedding_service.dimension)
        self.llm_service = llm_service or LLMService()
        self.text_splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )

        # Try to load existing persisted vector store on initialization
        self.vector_store.load()

    def process_document(
        self,
        file_obj_or_path: str | Path | BinaryIO,
        filename: str,
    ) -> dict:
        """Ingest, extract, chunk, embed, and index a single document.

        Args:
            file_obj_or_path: Document path or stream.
            filename: Document filename.

        Returns:
            Dictionary with processing metadata (filename, page count, chunk count, status).
        """
        # 1. Check for duplicates already in vector store
        existing_docs = self.vector_store.get_indexed_documents()
        if any(doc["filename"] == filename for doc in existing_docs):
            logger.info("Document '%s' already indexed. Skipping redundant embedding.", filename)
            doc_info = next(d for d in existing_docs if d["filename"] == filename)
            return {
                "filename": filename,
                "pages": doc_info.get("total_pages") or 1,
                "chunks": doc_info.get("chunk_count", 0),
                "status": "Already Indexed (Skipped)",
            }

        # 2. Extract text & metadata
        documents: list[Document] = DocumentLoader.load(file_obj_or_path, filename)
        total_pages = len(documents) if documents[0].file_type == ".pdf" else 1

        # 3. Recursive text chunking
        chunks: list[TextChunk] = self.text_splitter.split_documents(documents)
        if not chunks:
            raise DocumentParsingError(f"No chunks generated for document '{filename}'.")

        # 4. Generate dense embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # 5. Add to ChromaDB collection & persist
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()

        return {
            "filename": filename,
            "pages": total_pages,
            "chunks": len(chunks),
            "status": "Successfully Indexed",
        }

    def ask(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        relevance_threshold: float | None = None,
        filter_document: str | None = None,
    ) -> RAGResponse:
        """Execute the question-answering workflow.

        1. Retrieve relevant chunks from vector store.
        2. Filter by relevance threshold.
        3. If no relevant chunks, return unknown-answer message immediately without calling LLM.
        4. Construct bounded context.
        5. Invoke Groq LLM with strict grounding prompt.
        6. Return answer and formatted source citations.
        """
        question = question.strip()
        if not question:
            return RAGResponse(
                answer="Please enter a valid question.",
                sources=[],
                query="",
                has_sufficient_context=False,
            )

        if self.vector_store.count() == 0:
            return RAGResponse(
                answer=(
                    "No documents are currently indexed in the system. "
                    "Please upload and process at least one document before asking questions."
                ),
                sources=[],
                query=question,
                has_sufficient_context=False,
            )

        # 1 & 2: Retrieval with threshold filtering
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            filter_document=filter_document,
        )

        if not retrieval_results:
            logger.info("No chunks exceeded relevance threshold for query: '%s'", question)
            return RAGResponse(
                answer=UNKNOWN_ANSWER_MESSAGE,
                sources=[],
                query=question,
                context_used="",
                has_sufficient_context=False,
            )

        # 3. Context Construction with length bounding
        context_blocks: list[str] = []
        sources: list[dict] = []
        current_context_len = 0

        for idx, result in enumerate(retrieval_results, 1):
            chunk = result.chunk
            page_info = f", Page: {chunk.page_number}" if chunk.page_number is not None else ""
            block = f"[Source {idx}: {chunk.source}{page_info}]\n{chunk.content}\n"

            # Check context budget
            if current_context_len + len(block) > MAX_CONTEXT_CHARS:
                # Add truncated portion or stop adding further chunks
                break

            context_blocks.append(block)
            current_context_len += len(block)

            sources.append({
                "source": chunk.source,
                "page_number": chunk.page_number,
                "file_type": chunk.metadata.get("file_type", ""),
                "content": chunk.content,
                "score": result.score,
                "chunk_id": chunk.chunk_id,
            })

        full_context = "\n---\n".join(context_blocks)

        # 4. Generate grounded answer via Groq
        answer = self.llm_service.generate_rag_answer(
            question=question,
            context=full_context,
            conversation_history=conversation_history,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=full_context,
            has_sufficient_context=True,
        )
