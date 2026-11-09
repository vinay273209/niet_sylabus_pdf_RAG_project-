from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str, expected_dimension: int = 384) -> None:
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_embedding_dimension()
        if self.dimension != expected_dimension:
            raise ValueError(
                f"Embedding model '{model_name}' returns {self.dimension} dimensions; "
                f"Qdrant collection expects {expected_dimension}."
            )

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()
