"""Source attribution and citation formatting utilities.

Formats retrieved document chunks into clean, informative citations displaying
original filenames, page numbers, relevance metrics, and text excerpts.
Guarantees source fidelity without hallucinated references.
"""

from __future__ import annotations

from typing import Any


class SourceFormatter:
    """Provides methods to format retrieved source chunks for UI rendering and text export."""

    @staticmethod
    def format_page_label(page_number: int | None, file_type: str = "") -> str:
        """Return a human-readable page label."""
        if page_number is not None:
            return f"Page {page_number}"
        if file_type in {".txt", ".docx"}:
            return f"N/A ({file_type.upper().replace('.', '')} Document)"
        return "N/A"

    @classmethod
    def format_source_item(cls, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize a source dictionary into a standard display format.

        Expected input keys:
            source: str
            page_number: int | None
            content: str
            score: float
            chunk_id: str (optional)
            file_type: str (optional)
        """
        source = item.get("source", "Unknown Document")
        page_number = item.get("page_number")
        file_type = item.get("file_type", "")
        content = item.get("content", "").strip()
        score = item.get("score", 0.0)

        page_label = cls.format_page_label(page_number, file_type)
        score_label = f"{score:.2f} (Cosine Similarity)"

        # Truncate content for preview if very long
        preview = content if len(content) <= 300 else content[:297] + "..."

        return {
            "source": source,
            "page_number": page_number,
            "page_label": page_label,
            "score": score,
            "score_label": score_label,
            "content": content,
            "preview": preview,
            "chunk_id": item.get("chunk_id", ""),
        }

    @classmethod
    def format_sources_markdown(cls, sources: list[dict[str, Any]]) -> str:
        """Format a list of sources into GitHub Flavored Markdown."""
        if not sources:
            return "_No supporting sources were retrieved._"

        lines = ["### Supporting Sources\n"]
        for idx, item in enumerate(sources, 1):
            formatted = cls.format_source_item(item)
            lines.append(f"**[{idx}] {formatted['source']}** — *{formatted['page_label']}*")
            lines.append(f"- **Relevance Score:** `{formatted['score_label']}`")
            lines.append(f"- **Excerpt:**\n> {formatted['content']}\n")

        lines.append(
            "> [!NOTE]\n"
            "> *Similarity scores represent vector cosine similarity against the query embedding "
            "and are not calibrated statistical probabilities.*"
        )
        return "\n".join(lines)
