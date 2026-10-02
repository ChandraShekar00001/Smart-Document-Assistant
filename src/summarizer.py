"""Automatic Document Summarizer module.

Provides document summarization supporting both single-pass generation (for short documents)
and hierarchical section-based map-reduce summarization (for long documents).
Features automatic caching to prevent redundant API token consumption.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import BinaryIO

from config.settings import (
    GROQ_FREE_TPM_LIMIT,
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_SAFE_INPUT_TOKENS,
    SUMMARY_SAFE_TOKEN_BUDGET,
    SUMMARY_SECTION_CHAR_LIMIT,
)
from src.document_loader import Document, DocumentLoader, EmptyDocumentError
from src.llm_service import LLMService

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """Manages token-budget-aware document summarization, local compression, and caching."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        safe_input_tokens: int = SUMMARY_SAFE_INPUT_TOKENS,
        section_char_limit: int = SUMMARY_SECTION_CHAR_LIMIT,
        single_pass_max_chars: int | None = None,
    ):
        self.llm_service = llm_service or LLMService()
        if single_pass_max_chars is not None:
            self.safe_input_tokens = max(50, single_pass_max_chars // 4)
        else:
            self.safe_input_tokens = safe_input_tokens
        self.section_char_limit = section_char_limit
        self.single_pass_max_chars = single_pass_max_chars or (self.safe_input_tokens * 4)
        # In-memory summary cache: document_hash -> summary text
        self._summary_cache: dict[str, str] = {}

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of document text for caching."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Estimate token count for LLM budgeting (~1.35 tokens per word or ~1 token per 3.8 chars)."""
        if not text:
            return 0
        words = len(text.split())
        by_words = int(words * 1.35)
        by_chars = len(text) // 4
        return max(by_words, by_chars)

    @staticmethod
    def _is_error_response(text: str) -> bool:
        """Check if LLM response indicates an API error, rate limit, or empty failure."""
        if not text:
            return True
        s = text.strip()
        return s.startswith("⚠️")

    @classmethod
    def _compress_document_locally(cls, text: str, target_tokens: int) -> str:
        """Lightweight local text compression without LLM calls.

        Extracts headings, metrics, dates, and action items distributed across all pages/sections
        to ensure full-document representation within the safe token budget.
        """
        import re

        if not text or not text.strip():
            return ""

        # 1. Split document into section blocks / paragraphs / pages
        raw_blocks = [b.strip() for b in re.split(r"\n{2,}|\x0c", text) if b.strip()]
        if len(raw_blocks) < 4:
            raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]

        if not raw_blocks:
            raw_blocks = [text.strip()]

        extracted_sections: list[str] = []

        # Patterns for structural and high-value semantic content
        heading_pattern = re.compile(
            r"^(#+|\d+[\.\)]|\d+\.\d+|\b[A-Z\s]{4,}\b|[A-Z][a-zA-Z\s]{2,35}:)",
            re.MULTILINE,
        )
        metric_pattern = re.compile(
            r"(\b\d+(\.\d+)?%?|\$|€|£|¥|[\d]{4}\b|\b\d+\s+(days|hours|weeks|months|years|percent|USD)\b)",
            re.IGNORECASE,
        )
        action_pattern = re.compile(
            r"\b(must|shall|require|responsible|obligation|deadline|action|policy|prohibited|mandatory)\b",
            re.IGNORECASE,
        )

        for block in raw_blocks:
            # Split block into individual sentences or lines
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", block)
                if s.strip() and len(s.strip()) > 5
            ]
            if not sentences:
                continue

            block_chosen: list[str] = []

            # 1. Structural headings
            if heading_pattern.match(sentences[0]):
                block_chosen.append(sentences[0][:200])

            # 2. Key metrics and actionable obligations
            for sent in sentences:
                if sent in block_chosen:
                    continue
                if metric_pattern.search(sent) or action_pattern.search(sent):
                    block_chosen.append(sent[:250])
                    if len(block_chosen) >= 2:
                        break

            # 3. Topic sentence fallback if block had no metrics/action lines
            if not block_chosen:
                block_chosen.append(sentences[0][:250])

            extracted_sections.append("\n".join(block_chosen))

        compressed_text = "\n\n".join(extracted_sections)

        # If compressed text fits within target token budget, return it
        if cls.estimate_token_count(compressed_text) <= target_tokens:
            return compressed_text

        # Second-pass refinement if still over budget: sample sections evenly across entire document
        current_tokens = cls.estimate_token_count(compressed_text)
        if current_tokens > target_tokens and extracted_sections:
            ratio = current_tokens / target_tokens
            step = max(2, int(ratio) + 1)
            sampled = [extracted_sections[i] for i in range(0, len(extracted_sections), step)]
            # Ensure the final section is retained so conclusion/later pages are never dropped
            if (len(extracted_sections) - 1) % step != 0:
                sampled.append(extracted_sections[-1])
            compressed_text = "\n\n".join(sampled)

        # Final safety boundary: if still slightly above due to long lines, keep lines up to budget
        if cls.estimate_token_count(compressed_text) > target_tokens:
            lines = [l for l in compressed_text.splitlines() if l.strip()]
            accumulated: list[str] = []
            running_tokens = 0
            for line in lines:
                l_tokens = cls.estimate_token_count(line)
                if running_tokens + l_tokens > target_tokens:
                    break
                accumulated.append(line)
                running_tokens += l_tokens
            if accumulated:
                compressed_text = "\n".join(accumulated)

        return compressed_text

    def summarize_text(self, text: str, doc_name: str) -> tuple[str, bool]:
        """Summarize raw document text with token-budget awareness and caching.

        Args:
            text: Extracted plain text of the document.
            doc_name: Filename or document title.

        Returns:
            Tuple of (summary_markdown, was_cached: bool).
        """
        if not text or not text.strip():
            return f"_Document '{doc_name}' contains no extractable text to summarize._", False

        text_hash = self._compute_hash(text)
        if text_hash in self._summary_cache:
            logger.info("Retrieved summary for '%s' from cache.", doc_name)
            return self._summary_cache[text_hash], True

        # Evaluate token budget against safe threshold
        est_tokens = self.estimate_token_count(text)

        if est_tokens <= self.safe_input_tokens:
            # Case 1: Document fits comfortably within safe single-call token budget
            logger.info(
                "Document '%s' (~%d tokens) fits within safe single-call budget (%d tokens). Executing 1 Groq LLM call...",
                doc_name,
                est_tokens,
                self.safe_input_tokens,
            )
            summary = self.llm_service.generate_document_summary(text, doc_name)

        else:
            # Case 2: Document exceeds safe single-call budget.
            # Perform lightweight LOCAL compression across all pages/sections (0 extra LLM calls).
            logger.info(
                "Document '%s' (~%d tokens) exceeds safe budget (%d tokens). Compressing locally without LLM calls...",
                doc_name,
                est_tokens,
                self.safe_input_tokens,
            )
            compressed_text = self._compress_document_locally(text, self.safe_input_tokens)
            comp_tokens = self.estimate_token_count(compressed_text)

            if comp_tokens <= self.safe_input_tokens:
                logger.info(
                    "Document '%s' locally compressed from ~%d to ~%d tokens. Executing 1 final Groq LLM call...",
                    doc_name,
                    est_tokens,
                    comp_tokens,
                )
                summary = self.llm_service.generate_document_summary(compressed_text, doc_name)
            else:
                # Case 3: Document is exceptionally massive and still exceeds budget after compression.
                logger.warning(
                    "Document '%s' (~%d tokens) exceeds budget even after local compression (~%d tokens). Aborting to protect Groq quota.",
                    doc_name,
                    est_tokens,
                    comp_tokens,
                )
                return (
                    f"⚠️ **Document Too Large**: '{doc_name}' is too large to summarize within the Groq Free-Tier "
                    f"token quota (8,000 TPM limit) even after local compression. Please summarize specific sections "
                    f"or upload a shorter document.",
                    False,
                )

        # Cache the result if valid (do not cache errors)
        if not self._is_error_response(summary):
            self._summary_cache[text_hash] = summary

        return summary, False

    def summarize_document(
        self,
        file_obj_or_path: str | Path | BinaryIO,
        filename: str,
    ) -> tuple[str, bool]:
        """Load document and generate its summary.

        Args:
            file_obj_or_path: Document stream or path.
            filename: Document filename.

        Returns:
            Tuple of (summary_markdown, was_cached: bool).
        """
        try:
            docs: list[Document] = DocumentLoader.load(file_obj_or_path, filename)
            combined_text = "\n\n".join(d.text for d in docs)
            return self.summarize_text(combined_text, filename)
        except EmptyDocumentError as e:
            return f"⚠️ **Cannot Summarize**: {str(e)}", False
        except Exception as e:
            logger.error("Error summarizing '%s': %s", filename, e)
            return f"⚠️ **Error Summarizing '{filename}'**: {str(e)}", False

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        self._summary_cache.clear()

    @staticmethod
    def _split_into_sections(text: str, max_chars: int) -> list[str]:
        """Split long document text into semantically logical sections."""
        paragraphs = text.split("\n\n")
        sections: list[str] = []
        current_section: list[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > max_chars and current_section:
                sections.append("\n\n".join(current_section))
                current_section = []
                current_len = 0

            current_section.append(p)
            current_len += p_len + 2

        if current_section:
            sections.append("\n\n".join(current_section))

        return sections
