"""Unit tests for DocumentSummarizer."""

import pytest
from unittest.mock import MagicMock
from src.llm_service import LLMService
from src.summarizer import DocumentSummarizer


class MockSummarizerClient:
    """Mock LLM client specifically recording summary generation calls."""
    def __init__(self):
        self.call_count = 0
        self.chat = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, messages, **kwargs):
        self.call_count += 1
        class Msg:
            content = "### Document Summary\n- Key policy point 1\n- Key policy point 2"
        class Choice:
            message = Msg()
        class Resp:
            choices = [Choice()]
        return Resp()


@pytest.fixture
def mock_summarizer_llm():
    client = MockSummarizerClient()
    return LLMService(api_key="mock_key", client=client), client


def test_summarize_short_text(mock_summarizer_llm):
    """Verify short text triggers single-pass summarization."""
    llm_service, client = mock_summarizer_llm
    summarizer = DocumentSummarizer(llm_service=llm_service, section_char_limit=2000)

    short_text = "This is a brief policy document regarding expense reports."
    summary, was_cached = summarizer.summarize_text(short_text, "brief_policy.txt")

    assert "### Document Summary" in summary
    assert was_cached is False
    assert client.call_count == 1


def test_summarizer_caching(mock_summarizer_llm):
    """Verify repeated calls for the same content return cached summary without calling LLM again."""
    llm_service, client = mock_summarizer_llm
    summarizer = DocumentSummarizer(llm_service=llm_service, section_char_limit=2000)

    text = "Company policy details for caching test."
    summary1, was_cached1 = summarizer.summarize_text(text, "doc.txt")
    assert was_cached1 is False
    assert client.call_count == 1

    summary2, was_cached2 = summarizer.summarize_text(text, "doc.txt")
    assert was_cached2 is True
    assert client.call_count == 1  # No additional LLM call!
    assert summary1 == summary2


def test_summarize_large_document_uses_local_compression(mock_summarizer_llm):
    """Verify large document exceeding token budget uses local compression and makes exactly ONE LLM call."""
    llm_service, client = mock_summarizer_llm
    # Set conservative safe input tokens threshold to trigger compression
    summarizer = DocumentSummarizer(llm_service=llm_service, safe_input_tokens=250)

    long_text = "\n\n".join(
        [f"Section {i} Compliance: Enterprise security regulation {i} requires mandatory audit logging."
         for i in range(1, 20)]
    )
    # Confirm text exceeds the safe input tokens budget
    assert summarizer.estimate_token_count(long_text) > 250

    summary, was_cached = summarizer.summarize_text(long_text, "long_handbook.txt")

    assert "### Document Summary" in summary
    assert was_cached is False
    # CRITICAL: Executes local compression with 0 extra LLM calls and performs exactly ONE Groq call
    assert client.call_count == 1


def test_summarize_empty_text(mock_summarizer_llm):
    """Verify empty text returns informative message without errors."""
    llm_service, client = mock_summarizer_llm
    summarizer = DocumentSummarizer(llm_service=llm_service)

    summary, was_cached = summarizer.summarize_text("", "empty.txt")
    assert "no extractable text" in summary.lower()
    assert client.call_count == 0


def test_summarize_medium_document_single_call(mock_summarizer_llm):
    """Verify an 18-page medium document (~50,000 characters) results in exactly ONE LLM call."""
    llm_service, client = mock_summarizer_llm
    summarizer = DocumentSummarizer(llm_service=llm_service)

    # 18-page document text (~48,000 characters)
    eighteen_page_text = "\n\n".join(
        [f"Page {p} Paragraph: Standard operational policies and guidelines for enterprise operations." * 30
         for p in range(1, 19)]
    )
    assert len(eighteen_page_text) >= 40000

    summary, was_cached = summarizer.summarize_text(eighteen_page_text, "18_page_report.pdf")

    assert "### Document Summary" in summary
    assert was_cached is False
    # Exactly ONE call for an 18-page document, not 13-19 calls!
    assert client.call_count == 1


def test_local_compression_preserves_distributed_content():
    """Verify local compression preserves content from the beginning, middle, and end without simple text[:N] truncation."""
    pages = [f"Page {p} Header: Crucial finding regarding metric {p * 10}%." for p in range(1, 21)]
    long_document = "\n\n".join(pages)

    # Compress to a small budget that forces significant reduction
    compressed = DocumentSummarizer._compress_document_locally(long_document, target_tokens=150)

    # Verify beginning, middle, and end page markers are all preserved
    assert "Page 1" in compressed
    assert "Page 10" in compressed or "Page 11" in compressed
    assert "Page 20" in compressed


def test_summarize_rate_limit_stops_immediately():
    """Verify summarization stops immediately on 429/rate-limit error without repeated calls."""
    call_count = 0

    class RateLimitMockClient:
        def __init__(self):
            self.chat = MagicMock()
            self.chat.completions.create = self._create

        def _create(self, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate Groq 429 RateLimitError
            raise Exception("Rate limit reached for model openai/gpt-oss-120b on tokens per minute (TPM): Limit 8000, Requested 10000 (code: rate_limit_exceeded)")

    mock_client = RateLimitMockClient()
    llm_service = LLMService(api_key="mock_key", client=mock_client)
    summarizer = DocumentSummarizer(llm_service=llm_service, safe_input_tokens=200)

    long_text = "\n\n".join([f"Paragraph {i}: Content exceeding threshold." for i in range(10)])
    summary, was_cached = summarizer.summarize_text(long_text, "massive_doc.txt")

    assert "Quota / Rate Limit Exceeded" in summary or "⚠️" in summary
    assert was_cached is False
    # CRITICAL: Stopped immediately after first rate-limited request, did NOT hammer API 10+ times!
    assert call_count == 1
