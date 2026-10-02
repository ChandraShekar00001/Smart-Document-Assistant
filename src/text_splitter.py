"""Recursive text chunking implementation with configurable chunk size and overlap.

Splits document text into semantically cohesive chunks by hierarchically checking
separators (paragraphs -> sentences -> words -> characters) while preserving
and enriching document metadata (source filename, page number, chunk index).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from src.document_loader import Document


@dataclass
class TextChunk:
    """Represents a chunk of text derived from a document with associated metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page_number(self) -> int | None:
        return self.metadata.get("page_number")

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TextChunk:
        return cls(content=data["content"], metadata=data.get("metadata", {}))


class RecursiveTextSplitter:
    """Splits text recursively by paragraphs, lines, sentences, words, and characters."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_into_atomic_fragments(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text into fragments where each fragment is <= chunk_size."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Find first matching separator
        chosen_sep = None
        remaining_seps = []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen_sep = ""
                remaining_seps = []
                break
            if sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1:]
                break

        if chosen_sep is None or chosen_sep == "":
            # Character slice fallback
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

        # Split on chosen separator
        raw_parts = text.split(chosen_sep)
        fragments: list[str] = []
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            # Reattach sentence punctuation if separator was punctuation-based
            if chosen_sep in {". ", "? ", "! "}:
                part += chosen_sep.strip()

            if len(part) <= self.chunk_size:
                fragments.append(part)
            else:
                fragments.extend(self._split_into_atomic_fragments(part, remaining_seps))

        return fragments

    def _merge_fragments(self, fragments: list[str]) -> list[str]:
        """Combine atomic fragments into overlapping chunks up to chunk_size."""
        if not fragments:
            return []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for frag in fragments:
            frag_len = len(frag)
            added_sep_len = 1 if current_chunk else 0

            if current_len + added_sep_len + frag_len > self.chunk_size and current_chunk:
                # Emit current chunk
                merged = " ".join(current_chunk).strip()
                if merged:
                    chunks.append(merged)

                # Keep trailing fragments for overlap
                while current_chunk and current_len > self.chunk_overlap:
                    removed = current_chunk.pop(0)
                    current_len -= len(removed) + (1 if current_chunk else 0)

            current_chunk.append(frag)
            current_len += frag_len + (1 if len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = " ".join(current_chunk).strip()
            if merged and (not chunks or merged != chunks[-1]):
                chunks.append(merged)

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split a raw text string into chunks."""
        if not text or not text.strip():
            return []
        fragments = self._split_into_atomic_fragments(text.strip(), self.separators)
        return self._merge_fragments(fragments)

    def split_documents(self, documents: list[Document]) -> list[TextChunk]:
        """Split a list of Document objects into enriched TextChunk objects."""
        chunks: list[TextChunk] = []

        for doc in documents:
            text_splits = self.split_text(doc.text)
            for idx, split in enumerate(text_splits):
                chunk_metadata = dict(doc.metadata)
                chunk_id = (
                    f"{doc.source}_p{doc.page_number or 'NA'}_c{idx}_"
                    f"{uuid.uuid4().hex[:6]}"
                )
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "char_count": len(split),
                })
                chunks.append(TextChunk(content=split, metadata=chunk_metadata))

        return chunks
