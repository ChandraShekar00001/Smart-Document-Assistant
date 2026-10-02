"""Document extraction and loading module.

Supports PDF (PyMuPDF / pypdf fallback), TXT (UTF-8 / Latin-1), and DOCX (python-docx).
Extracts text page-by-page where possible, performs robust text cleaning,
and preserves comprehensive document metadata.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from config.settings import SUPPORTED_EXTENSIONS


class DocumentParsingError(Exception):
    """Raised when an error occurs while parsing a document."""
    pass


class UnsupportedFormatError(DocumentParsingError):
    """Raised when an uploaded file format is not supported."""
    pass


class EmptyDocumentError(DocumentParsingError):
    """Raised when a document contains no extractable text."""
    pass


@dataclass
class Document:
    """Represents an extracted unit of a document (e.g., a page or single file)."""
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page_number(self) -> int | None:
        return self.metadata.get("page_number")

    @property
    def file_type(self) -> str:
        return self.metadata.get("file_type", "")


def clean_text(raw_text: str) -> str:
    """Normalize and clean raw extracted text.

    - Replaces null bytes and non-printable control characters.
    - Normalizes carriage returns and multiple consecutive spaces.
    - Preserves paragraph breaks while collapsing 3+ blank lines into 2.
    - Strips leading and trailing whitespace.
    """
    if not raw_text:
        return ""

    # Remove null bytes
    text = raw_text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace strange vertical whitespace or form feeds
    text = re.sub(r"[\x0b\x0c]", "\n", text)

    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3 or more newlines into 2 (paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


class DocumentLoader:
    """Loads and extracts text from various document formats."""

    @staticmethod
    def load_txt(file_obj_or_path: str | Path | BinaryIO, filename: str) -> list[Document]:
        """Load text from a .txt file."""
        content: str = ""
        if isinstance(file_obj_or_path, (str, Path)):
            path = Path(file_obj_or_path)
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            # File-like stream
            raw_bytes = file_obj_or_path.read()
            if isinstance(raw_bytes, str):
                content = raw_bytes
            else:
                for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                    try:
                        content = raw_bytes.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue

        cleaned = clean_text(content)
        if not cleaned:
            raise EmptyDocumentError(f"Document '{filename}' contains no extractable text.")

        metadata = {
            "source": filename,
            "file_type": ".txt",
            "page_number": None,
            "total_pages": 1,
            "char_count": len(cleaned),
        }
        return [Document(text=cleaned, metadata=metadata)]

    @staticmethod
    def load_pdf(file_obj_or_path: str | Path | BinaryIO, filename: str) -> list[Document]:
        """Load text from a .pdf file page by page using PyMuPDF (fitz) or pypdf fallback."""
        docs: list[Document] = []
        raw_bytes: bytes | None = None

        if hasattr(file_obj_or_path, "read"):
            raw_bytes = file_obj_or_path.read()

        # Try PyMuPDF (fitz) first
        try:
            import fitz

            if raw_bytes is not None:
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
            else:
                doc = fitz.open(str(file_obj_or_path))

            total_pages = len(doc)
            if total_pages == 0:
                raise EmptyDocumentError(f"PDF '{filename}' has 0 pages.")

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_text = clean_text(page.get_text("text"))
                if page_text:
                    docs.append(Document(
                        text=page_text,
                        metadata={
                            "source": filename,
                            "file_type": ".pdf",
                            "page_number": page_idx + 1,
                            "total_pages": total_pages,
                            "char_count": len(page_text),
                        }
                    ))
            doc.close()

        except Exception as fitz_err:
            # Fallback to pypdf if fitz had an unexpected failure
            try:
                import pypdf

                if raw_bytes is not None:
                    stream = io.BytesIO(raw_bytes)
                else:
                    stream = open(str(file_obj_or_path), "rb")

                reader = pypdf.PdfReader(stream)
                total_pages = len(reader.pages)
                docs = []
                for page_idx, page in enumerate(reader.pages):
                    text = clean_text(page.extract_text() or "")
                    if text:
                        docs.append(Document(
                            text=text,
                            metadata={
                                "source": filename,
                                "file_type": ".pdf",
                                "page_number": page_idx + 1,
                                "total_pages": total_pages,
                                "char_count": len(text),
                            }
                        ))
                if hasattr(stream, "close"):
                    stream.close()
            except Exception as pypdf_err:
                raise DocumentParsingError(
                    f"Failed to parse PDF '{filename}': PyMuPDF error: {fitz_err}; pypdf error: {pypdf_err}"
                )

        if not docs:
            raise EmptyDocumentError(
                f"PDF '{filename}' contains no extractable text. It may be a scanned image or empty."
            )

        return docs

    @staticmethod
    def load_docx(file_obj_or_path: str | Path | BinaryIO, filename: str) -> list[Document]:
        """Load text from a .docx file using python-docx."""
        try:
            import docx

            if hasattr(file_obj_or_path, "read"):
                raw_bytes = file_obj_or_path.read()
                file_stream = io.BytesIO(raw_bytes)
                doc = docx.Document(file_stream)
            else:
                doc = docx.Document(str(file_obj_or_path))

            paragraphs_text = [p.text for p in doc.paragraphs if p.text.strip()]

            # Also extract table cells
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs_text.append(row_text)

            full_text = clean_text("\n\n".join(paragraphs_text))
            if not full_text:
                raise EmptyDocumentError(f"DOCX '{filename}' contains no extractable text.")

            metadata = {
                "source": filename,
                "file_type": ".docx",
                "page_number": None,
                "total_pages": 1,
                "char_count": len(full_text),
            }
            return [Document(text=full_text, metadata=metadata)]

        except EmptyDocumentError:
            raise
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse DOCX '{filename}': {str(e)}")

    @classmethod
    def load(cls, file_obj_or_path: str | Path | BinaryIO, filename: str | None = None) -> list[Document]:
        """Dispatch document loading based on file extension.

        Args:
            file_obj_or_path: Path string, Path object, or file-like stream (BytesIO).
            filename: Name of the file. Required if file_obj_or_path is a stream.

        Returns:
            List of extracted Document objects with metadata.
        """
        if filename is None:
            if isinstance(file_obj_or_path, (str, Path)):
                filename = Path(file_obj_or_path).name
            else:
                filename = getattr(file_obj_or_path, "name", "unknown_file")

        ext = Path(filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if ext == ".txt":
            return cls.load_txt(file_obj_or_path, filename)
        elif ext == ".pdf":
            return cls.load_pdf(file_obj_or_path, filename)
        elif ext == ".docx":
            return cls.load_docx(file_obj_or_path, filename)
        else:
            raise UnsupportedFormatError(f"Unsupported extension: {ext}")
