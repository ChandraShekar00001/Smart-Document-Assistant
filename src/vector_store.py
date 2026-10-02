"""Local ChromaDB vector store with persistent storage and metadata synchronization.

Uses ChromaDB PersistentClient with cosine distance metric, providing
reliable on-disk persistence and metadata management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np

from config.settings import CHROMA_PERSIST_DIR
from src.text_splitter import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages persistent ChromaDB vector index and corresponding chunk metadata."""

    def __init__(
        self,
        dimension: int = 384,
        persist_dir: Path | str | None = None,
        collection_name: str = "documents",
        index_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
    ):
        self.dimension = dimension
        self.collection_name = collection_name

        # Determine persistence directory
        if persist_dir is not None:
            self.persist_dir = Path(persist_dir)
        elif index_path is not None:
            # If a file path was passed, use its directory for Chroma persistence
            p = Path(index_path)
            self.persist_dir = p.parent / "chroma_db" if p.suffix else p
        else:
            self.persist_dir = Path(CHROMA_PERSIST_DIR)

        self._client = None
        self._collection = None
        self.chunks: list[TextChunk] = []

    @property
    def client(self):
        """Lazy-initialize ChromaDB PersistentClient."""
        if self._client is None:
            import chromadb

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self):
        """Lazy-initialize or get ChromaDB collection with cosine metric."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        """Return total number of chunks currently stored in the index."""
        try:
            return self.collection.count()
        except Exception:
            return len(self.chunks)

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their pre-computed embeddings to ChromaDB.

        Args:
            chunks: List of TextChunk objects.
            embeddings: 2D numpy array of shape (len(chunks), dimension).
        """
        if not chunks or len(chunks) == 0:
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.chunk_id or f"{chunk.source}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk.content)

            # Chroma metadata values must be str, int, float, or bool
            clean_meta: dict[str, Any] = {}
            for k, v in chunk.metadata.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )

        self.chunks.extend(chunks)
        logger.info(
            "Added %d chunks to ChromaDB collection '%s'. Total count: %d",
            len(chunks),
            self.collection_name,
            self.count(),
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[tuple[TextChunk, float]]:
        """Search ChromaDB for the most similar chunks to the query embedding.

        Args:
            query_embedding: 2D or 1D numpy array of shape (1, dimension) or (dimension,).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of tuples: (TextChunk, similarity_score).
        """
        if self.count() == 0:
            return []

        k = min(top_k, self.count())
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        if isinstance(query_list, list) and len(query_list) > 0 and not isinstance(query_list[0], list):
            query_list = [query_list]

        results = self.collection.query(
            query_embeddings=query_list,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        output: list[tuple[TextChunk, float]] = []
        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return output

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

        for doc_text, meta, dist, chk_id in zip(docs, metas, dists, ids):
            restored_meta = dict(meta)
            # Restore None for empty string page_number
            if restored_meta.get("page_number") == "":
                restored_meta["page_number"] = None
            elif isinstance(restored_meta.get("page_number"), (int, float)):
                restored_meta["page_number"] = int(restored_meta["page_number"])

            if "chunk_id" not in restored_meta and chk_id:
                restored_meta["chunk_id"] = chk_id

            chunk = TextChunk(content=doc_text, metadata=restored_meta)
            # Cosine similarity = 1.0 - Cosine distance
            similarity = float(1.0 - dist)
            output.append((chunk, similarity))

        return output

    def delete_document(self, filename: str, embedding_service: Any = None) -> int:
        """Delete all chunks originating from a specified document.

        Args:
            filename: Name of the document to remove.
            embedding_service: Ignored, preserved for backward compatibility.

        Returns:
            Number of chunks removed.
        """
        existing = self.collection.get(where={"source": filename})
        matching_ids = existing.get("ids", [])

        if not matching_ids:
            return 0

        self.collection.delete(ids=matching_ids)
        removed_count = len(matching_ids)
        self.chunks = [c for c in self.chunks if c.source != filename]

        logger.info("Removed document '%s' (%d chunks) from ChromaDB.", filename, removed_count)
        return removed_count

    def clear(self) -> None:
        """Clear all indexed documents and reset the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        self.chunks.clear()
        logger.info("ChromaDB vector store cleared completely.")

    def get_indexed_documents(self) -> list[dict]:
        """Return aggregated information for each unique indexed document."""
        doc_stats: dict[str, dict] = {}

        # Ensure chunks are loaded if empty but collection has data
        if not self.chunks and self.count() > 0:
            self.load()

        for chunk in self.chunks:
            source = chunk.source
            if source not in doc_stats:
                doc_stats[source] = {
                    "filename": source,
                    "file_type": chunk.metadata.get("file_type", ""),
                    "total_pages": chunk.metadata.get("total_pages"),
                    "chunk_count": 0,
                    "total_chars": 0,
                }
            doc_stats[source]["chunk_count"] += 1
            doc_stats[source]["total_chars"] += len(chunk.content)

        return list(doc_stats.values())

    def save(self, index_path: Path | None = None, metadata_path: Path | None = None) -> None:
        """Persist vector store state.

        ChromaDB PersistentClient automatically flushes and persists to disk.
        Method preserved for interface compatibility.
        """
        pass

    def load(self, index_path: Path | None = None, metadata_path: Path | None = None) -> bool:
        """Load persisted collection and chunk metadata from disk into memory.

        Returns:
            True if chunks exist and were loaded, False otherwise.
        """
        try:
            total = self.count()
            if total == 0:
                self.chunks = []
                return False

            records = self.collection.get(include=["documents", "metadatas"])
            loaded: list[TextChunk] = []

            if records and records.get("documents"):
                docs = records["documents"]
                metas = records["metadatas"] or [{}] * len(docs)
                ids = records["ids"] or [""] * len(docs)

                for doc_text, meta, chk_id in zip(docs, metas, ids):
                    restored_meta = dict(meta)
                    if restored_meta.get("page_number") == "":
                        restored_meta["page_number"] = None
                    elif isinstance(restored_meta.get("page_number"), (int, float)):
                        restored_meta["page_number"] = int(restored_meta["page_number"])

                    if "chunk_id" not in restored_meta and chk_id:
                        restored_meta["chunk_id"] = chk_id

                    loaded.append(TextChunk(content=doc_text, metadata=restored_meta))

            self.chunks = loaded
            logger.info("Successfully loaded %d chunks from ChromaDB.", len(self.chunks))
            return True

        except Exception as e:
            logger.error("Failed to load vector store from ChromaDB: %s", e)
            return False
