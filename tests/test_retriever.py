"""Unit tests for VectorStore, Retriever, and ChromaDB similarity search."""

import pytest
from src.retriever import Retriever
from src.vector_store import VectorStore
from src.text_splitter import TextChunk


def test_vector_store_persistence(tmp_path, embedding_service, sample_chunks):
    """Verify ChromaDB collection and chunk metadata save and reload correctly."""
    persist_dir = tmp_path / "chroma_db"

    vs1 = VectorStore(dimension=embedding_service.dimension, persist_dir=persist_dir)
    embeddings = embedding_service.embed_documents([c.content for c in sample_chunks])
    vs1.add_chunks(sample_chunks, embeddings)
    vs1.save()

    assert persist_dir.exists()
    assert (persist_dir / "chroma.sqlite3").exists()

    vs2 = VectorStore(dimension=embedding_service.dimension, persist_dir=persist_dir)
    loaded = vs2.load()
    assert loaded is True
    assert vs2.count() == len(sample_chunks)
    assert vs2.chunks[0].content == sample_chunks[0].content


def test_retriever_semantic_match(populated_vector_store, embedding_service):
    """Verify query retrieves semantically relevant chunks."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_service=embedding_service,
        top_k=2,
        relevance_threshold=0.2,
    )

    results = retriever.retrieve("What are the core hours for remote work?")
    assert len(results) > 0
    top_result = results[0]
    assert "core collaborative hours" in top_result.content.lower()
    assert top_result.source == "company_policy.txt"
    assert top_result.score > 0.3


def test_retriever_threshold_filtering(populated_vector_store, embedding_service):
    """Verify that chunks with scores below relevance threshold are filtered out."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_service=embedding_service,
        top_k=5,
        relevance_threshold=0.99,  # Impossibly high threshold
    )

    results = retriever.retrieve("What are the core hours?")
    assert len(results) == 0


def test_retriever_filter_by_document(populated_vector_store, embedding_service):
    """Verify filtering retrieval by a specific document name."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_service=embedding_service,
        top_k=5,
        relevance_threshold=0.1,
    )

    # Search for passwords but restrict to leave_policy.txt
    results = retriever.retrieve("password policy", filter_document="leave_policy.txt")
    for r in results:
        assert r.source == "leave_policy.txt"


def test_delete_document(populated_vector_store, embedding_service):
    """Verify deleting a document removes its chunks from ChromaDB."""
    initial_count = populated_vector_store.count()
    removed = populated_vector_store.delete_document("company_policy.txt", embedding_service=embedding_service)

    assert removed == 2
    assert populated_vector_store.count() == initial_count - 2
    for chunk in populated_vector_store.chunks:
        assert chunk.source != "company_policy.txt"


def test_empty_vector_store_retrieval(tmp_path, embedding_service):
    """Verify empty vector store returns empty list gracefully without errors."""
    empty_vs = VectorStore(dimension=embedding_service.dimension, index_path=tmp_path / "empty.bin", metadata_path=tmp_path / "empty.json")
    retriever = Retriever(vector_store=empty_vs, embedding_service=embedding_service)

    results = retriever.retrieve("Any question")
    assert results == []
