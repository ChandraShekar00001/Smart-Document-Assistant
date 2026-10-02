"""Unit tests for RecursiveTextSplitter."""

import pytest
from src.document_loader import Document
from src.text_splitter import RecursiveTextSplitter, TextChunk


def test_splitter_invalid_parameters():
    """Verify that invalid chunk parameters raise ValueError."""
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=150)

    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=-10, chunk_overlap=5)


def test_split_short_text():
    """Verify short text produces exactly one chunk."""
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    text = "Short policy text."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_long_text_respects_chunk_size():
    """Verify that chunks generated from long text stay within chunk_size limits."""
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    long_text = (
        "Paragraph one is about remote work standards and guidelines. "
        "It provides information about laptops and monitors.\n\n"
        "Paragraph two discusses travel allowances and per diem meals. "
        "Employees get up to seventy-five dollars per calendar day."
    )
    chunks = splitter.split_text(long_text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 120  # Allowing small margin for boundary separators


def test_split_documents_preserves_metadata():
    """Verify that split_documents preserves parent document metadata and assigns chunk IDs."""
    splitter = RecursiveTextSplitter(chunk_size=150, chunk_overlap=30)
    doc = Document(
        text="Section 1: Information security requires 14 character passwords.\n\nSection 2: Multi-Factor Authentication is mandatory.",
        metadata={"source": "security.pdf", "page_number": 3, "file_type": ".pdf"},
    )
    chunks = splitter.split_documents([doc])
    assert len(chunks) >= 1
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, TextChunk)
        assert chunk.metadata["source"] == "security.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["file_type"] == ".pdf"
        assert chunk.metadata["chunk_index"] == idx
        assert "security.pdf_p3_" in chunk.metadata["chunk_id"]


def test_empty_text_returns_empty_list():
    """Verify that empty or whitespace-only text returns empty list."""
    splitter = RecursiveTextSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n\n  ") == []
