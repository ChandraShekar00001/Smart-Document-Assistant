"""Unit tests for RAG pipeline and grounded question answering."""

import pytest
from src.llm_service import UNKNOWN_ANSWER_MESSAGE
from src.rag_pipeline import RAGPipeline
from src.source_formatter import SourceFormatter


def test_rag_pipeline_with_relevant_context(populated_vector_store, embedding_service, mock_llm_service):
    """Verify that a valid question retrieves sources and calls LLM to produce an answer."""
    pipeline = RAGPipeline(
        vector_store=populated_vector_store,
        embedding_service=embedding_service,
        llm_service=mock_llm_service,
        top_k=3,
        relevance_threshold=0.2,
    )

    response = pipeline.ask("What is the home office allowance?")
    assert response.has_sufficient_context is True
    assert len(response.sources) > 0
    assert response.sources[0]["source"] == "company_policy.txt"
    assert "home office equipment allowance" in response.sources[0]["content"].lower()
    assert response.answer == "This is a grounded mock answer."


def test_rag_pipeline_unknown_answer_on_low_relevance(populated_vector_store, embedding_service, mock_llm_service):
    """Verify that an unanswerable question triggers unknown-answer message when below threshold."""
    pipeline = RAGPipeline(
        vector_store=populated_vector_store,
        embedding_service=embedding_service,
        llm_service=mock_llm_service,
        top_k=3,
        relevance_threshold=0.85,  # Too high for absent topics
    )

    response = pipeline.ask("What is the policy regarding pet bereavement leave?")
    assert response.has_sufficient_context is False
    assert response.answer == UNKNOWN_ANSWER_MESSAGE
    assert response.sources == []


def test_rag_pipeline_empty_vector_store(tmp_path, embedding_service, mock_llm_service):
    """Verify that asking when no documents are uploaded informs the user clearly."""
    from src.vector_store import VectorStore
    empty_vs = VectorStore(dimension=embedding_service.dimension, persist_dir=tmp_path / "chroma_db")
    pipeline = RAGPipeline(
        vector_store=empty_vs,
        embedding_service=embedding_service,
        llm_service=mock_llm_service,
    )

    response = pipeline.ask("What is the vacation policy?")
    assert "No documents are currently indexed" in response.answer
    assert response.has_sufficient_context is False


def test_source_formatter_markdown_output():
    """Verify SourceFormatter generates clean markdown with disclaimer and without hallucinated fields."""
    sample_source = [{
        "source": "employee_handbook.pdf",
        "page_number": 2,
        "file_type": ".pdf",
        "content": "Passwords must be rotated every 90 days.",
        "score": 0.784,
    }]

    md = SourceFormatter.format_sources_markdown(sample_source)
    assert "employee_handbook.pdf" in md
    assert "Page 2" in md
    assert "0.78" in md
    assert "Passwords must be rotated every 90 days." in md
    assert "Similarity scores represent vector cosine similarity" in md


def test_source_formatter_txt_source_shows_na_page():
    """Verify plain text source clearly indicates page number is unavailable."""
    sample_source = [{
        "source": "notes.txt",
        "page_number": None,
        "file_type": ".txt",
        "content": "Plain text note without pages.",
        "score": 0.65,
    }]

    item = SourceFormatter.format_source_item(sample_source[0])
    assert "N/A" in item["page_label"]
    assert "TXT" in item["page_label"]
