"""Unit tests for DocumentLoader and text cleaning."""

import pytest
from pathlib import Path
from src.document_loader import (
    DocumentLoader,
    clean_text,
    EmptyDocumentError,
    UnsupportedFormatError,
)


def test_clean_text():
    """Verify that text cleaning handles null bytes, extra spaces, and redundant newlines."""
    dirty = "Hello\x00 world!  \r\n\r\n\n\nThis is    a   test.   "
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "    " not in cleaned
    assert "Hello world!\n\nThis is a test." == cleaned


def test_load_txt_file(tmp_path):
    """Verify loading valid TXT files preserves content and metadata."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Company remote policy states 10am to 3pm core hours.", encoding="utf-8")

    docs = DocumentLoader.load(txt_file)
    assert len(docs) == 1
    assert "10am to 3pm" in docs[0].text
    assert docs[0].metadata["source"] == "test.txt"
    assert docs[0].metadata["file_type"] == ".txt"
    assert docs[0].metadata["page_number"] is None


def test_load_empty_txt_file(tmp_path):
    """Verify that an empty TXT file raises EmptyDocumentError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\n  \t ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        DocumentLoader.load(empty_file)


def test_load_unsupported_extension(tmp_path):
    """Verify that an unsupported file format raises UnsupportedFormatError."""
    fake_img = tmp_path / "photo.jpg"
    fake_img.write_bytes(b"dummy image bytes")

    with pytest.raises(UnsupportedFormatError):
        DocumentLoader.load(fake_img)


def test_load_pdf_file():
    """Verify loading real multi-page PDF extracts correct page count and page numbers."""
    pdf_path = Path("data/sample_documents/employee_handbook.pdf")
    if not pdf_path.exists():
        pytest.skip("Sample PDF not found.")

    docs = DocumentLoader.load(pdf_path)
    assert len(docs) > 0
    assert docs[0].metadata["file_type"] == ".pdf"
    assert docs[0].metadata["page_number"] == 1
    assert any("Chapter" in d.text for d in docs)


def test_load_docx_file():
    """Verify loading real DOCX file extracts paragraphs and tables."""
    docx_path = Path("data/sample_documents/employee_handbook.docx")
    if not docx_path.exists():
        pytest.skip("Sample DOCX not found.")

    docs = DocumentLoader.load(docx_path)
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == ".docx"
    assert "Employee Handbook" in docs[0].text
