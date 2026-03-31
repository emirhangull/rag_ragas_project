from __future__ import annotations

from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers package is required for embedding."
            ) from exc
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        if isinstance(vectors, np.ndarray):
            return vectors.tolist()
        return [np.asarray(vector).tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return np.asarray(vector).tolist()
