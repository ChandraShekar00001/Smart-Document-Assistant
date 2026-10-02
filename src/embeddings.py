"""Dense vector embedding generation using sentence-transformers.

Generates normalized vector embeddings using a local pretrained model (default: all-MiniLM-L6-v2).
Embeddings are L2-normalized for cosine similarity in vector search.
"""

from __future__ import annotations

import logging
from typing import ClassVar
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages the sentence-transformers model instance and embedding generation."""

    _instances: ClassVar[dict[str, EmbeddingService]] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
        """Get or create a cached instance for the specified model name."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    @property
    def model(self):
        """Lazy load the SentenceTransformer model on first access."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # Infer dimension
            test_emb = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = int(test_emb.shape[1])
            logger.info(
                "Model %s loaded successfully. Embedding dimension: %d",
                self.model_name,
                self._dimension,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if self._dimension is None:
            _ = self.model  # Trigger load
        return self._dimension or 384

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate L2-normalized embeddings for a list of document strings.

        Args:
            texts: List of text chunks to embed.
            batch_size: Batch size for model inference.

        Returns:
            2D numpy array of shape (len(texts), dimension) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # encode returns numpy array or tensor
        raw_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Built-in L2 normalization
        )

        return np.asarray(raw_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an L2-normalized embedding for a single query string.

        Args:
            query: The user query string.

        Returns:
            2D numpy array of shape (1, dimension) with float32 dtype.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return np.zeros((1, self.dimension), dtype=np.float32)

        raw_embedding = self.model.encode(
            [cleaned_query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(raw_embedding, dtype=np.float32)
