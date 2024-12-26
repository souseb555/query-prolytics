import logging
from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)


class EmbeddingModelsConfig:
    """Configuration for embedding models."""
    model_type: str = "openai"
    dims: int = 512
    context_length: int = 512
    batch_size: int = 32


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @classmethod
    def create(cls, config: EmbeddingModelsConfig) -> "EmbeddingModel":
        """
        Factory method to create an embedding model instance based on the configuration.
        """
        if config.model_type == "openai":
            from .models import OpenAIEmbeddings
            return OpenAIEmbeddings(config)
        else:
            raise ValueError(f"Unsupported embedding model type: {config.model_type}")

    @abstractmethod
    def embedding_fn(self) -> Callable[[List[str]], List[List[float]]]:
        """Returns a function to compute embeddings."""
        pass

    @property
    @abstractmethod
    def embedding_dims(self) -> int:
        """Returns the dimensions of the embeddings."""
        pass

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        try:
            embeddings = self.embedding_fn()([text1, text2])
            emb1, emb2 = np.array(embeddings[0]), np.array(embeddings[1])
            similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
