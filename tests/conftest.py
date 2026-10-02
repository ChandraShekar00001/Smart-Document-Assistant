"""Shared pytest fixtures and test configuration."""

import pytest
from unittest.mock import MagicMock
from src.embeddings import EmbeddingService
from src.llm_service import LLMService, UNKNOWN_ANSWER_MESSAGE
from src.text_splitter import TextChunk
from src.vector_store import VectorStore


class MockGroqMessage:
    def __init__(self, content: str):
        self.content = content


class MockGroqChoice:
    def __init__(self, content: str):
        self.message = MockGroqMessage(content)


class MockGroqCompletion:
    def __init__(self, content: str):
        self.choices = [MockGroqChoice(content)]


class MockGroqClient:
    """Mock Groq client for testing without API keys."""
    def __init__(self, default_response: str = "This is a grounded mock answer."):
        self.default_response = default_response
        self.calls = []
        self.chat = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, messages, **kwargs):
        self.calls.append(messages)
        # Check if contents indicate no info or asking about missing topic
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "pet bereavement" in content.lower():
                return MockGroqCompletion(UNKNOWN_ANSWER_MESSAGE)
        return MockGroqCompletion(self.default_response)


# Backwards compatibility alias
MockGeminiResponse = MockGroqCompletion
MockGeminiModel = MockGroqClient


@pytest.fixture
def mock_groq_client():
    return MockGroqClient()


@pytest.fixture
def mock_gemini_client(mock_groq_client):
    return mock_groq_client


@pytest.fixture
def mock_llm_service(mock_groq_client):
    return LLMService(api_key="mock_test_key_12345", client=mock_groq_client)


@pytest.fixture(scope="session")
def embedding_service():
    """Use the real embedding service once for fast local test embeddings."""
    return EmbeddingService.get_instance("all-MiniLM-L6-v2")


@pytest.fixture
def sample_chunks():
    """Sample pre-built chunks for testing retrieval."""
    return [
        TextChunk(
            content="Apex Global Technologies provides a home office equipment allowance of up to $1,200.",
            metadata={"source": "company_policy.txt", "page_number": None, "file_type": ".txt", "chunk_id": "c1"},
        ),
        TextChunk(
            content="Employees must be available during core collaborative hours from 10:00 AM to 3:00 PM EST.",
            metadata={"source": "company_policy.txt", "page_number": None, "file_type": ".txt", "chunk_id": "c2"},
        ),
        TextChunk(
            content="Full-time employees receive 22 days of annual paid time off (PTO) and 10 paid sick days.",
            metadata={"source": "leave_policy.txt", "page_number": 1, "file_type": ".pdf", "chunk_id": "c3"},
        ),
        TextChunk(
            content="Corporate passwords must be at least 14 characters long and rotated every 90 days with MFA.",
            metadata={"source": "employee_handbook.pdf", "page_number": 2, "file_type": ".pdf", "chunk_id": "c4"},
        ),
    ]


@pytest.fixture
def populated_vector_store(tmp_path, embedding_service, sample_chunks):
    """VectorStore populated with sample chunks and persisted in temp directory."""
    persist_dir = tmp_path / "test_chroma"
    vs = VectorStore(dimension=embedding_service.dimension, persist_dir=persist_dir)
    embeddings = embedding_service.embed_documents([c.content for c in sample_chunks])
    vs.add_chunks(sample_chunks, embeddings)
    vs.save()
    return vs
