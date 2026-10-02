"""Centralized configuration and environment settings for Smart Document Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
# Search from current file up to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path, override=False)

# Directory Paths
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Vector Store Persistence Files
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE_PATH = VECTOR_STORE_DIR / "metadata.json"

# LLM Configuration (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Text Splitting Defaults
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Defaults
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

# Safety and Prompt Limits
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# Groq Free-Tier Rate-Limit Aware Summarization Budget (8,000 TPM limit)
GROQ_FREE_TPM_LIMIT = int(os.getenv("GROQ_FREE_TPM_LIMIT", "8000"))
SUMMARY_SAFE_TOKEN_BUDGET = int(os.getenv("SUMMARY_SAFE_TOKEN_BUDGET", "6000"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "512"))
SUMMARY_PROMPT_OVERHEAD_TOKENS = int(os.getenv("SUMMARY_PROMPT_OVERHEAD_TOKENS", "350"))
SUMMARY_SAFE_INPUT_TOKENS = SUMMARY_SAFE_TOKEN_BUDGET - SUMMARY_MAX_OUTPUT_TOKENS - SUMMARY_PROMPT_OVERHEAD_TOKENS
SUMMARY_SECTION_CHAR_LIMIT = int(os.getenv("SUMMARY_SECTION_CHAR_LIMIT", "20000"))
SUMMARY_SINGLE_PASS_MAX_CHARS = int(os.getenv("SUMMARY_SINGLE_PASS_MAX_CHARS", "20000"))

# Supported File Extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def get_groq_api_key(override_key: str | None = None) -> str:
    """Retrieve the Groq API key, giving priority to a UI-supplied key if provided.

    Args:
        override_key: Optional explicit API key (e.g. from UI input).

    Returns:
        The active API key or an empty string.
    """
    if override_key and override_key.strip():
        return override_key.strip()
    return os.getenv("GROQ_API_KEY", "").strip()


def is_api_key_configured(override_key: str | None = None) -> bool:
    """Check if a non-empty Groq API key is available."""
    key = get_groq_api_key(override_key)
    return bool(key and len(key) > 5)


# Backward-compatibility aliases
DEFAULT_GEMINI_MODEL = DEFAULT_GROQ_MODEL
get_gemini_api_key = get_groq_api_key
