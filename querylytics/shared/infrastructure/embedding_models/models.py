import os
import time
from typing import Callable, List
from openai import OpenAI
import tiktoken
from querylytics.shared.infrastructure.embedding_models.base import EmbeddingModelsConfig, EmbeddingModel
from itertools import islice

from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")

def batched(iterable: Iterable[T], n: int) -> Iterable[Sequence[T]]:
    """
    Batch data into tuples of length n. The last batch may be shorter.

    Args:
        iterable (Iterable[T]): Input iterable.
        n (int): Batch size.

    Yields:
        Iterable[Sequence[T]]: Batches of data.
    """
    if n < 1:
        raise ValueError("Batch size must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    """Configuration specific to OpenAI embeddings."""
    model_type: str = "openai"
    model_name: str = "text-embedding-ada-002"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    dims: int = 1536
    context_length: int = 8192
    batch_size: int = 16


class OpenAIEmbeddings(EmbeddingModel):
    """Implementation of OpenAI embeddings."""

    def __init__(self, config: OpenAIEmbeddingsConfig):
        super().__init__()
        self.config = config
        self.config.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.config.api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set in the environment or passed in the config."
            )
        self.client = OpenAI(api_key=self.config.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)

    def truncate_texts(self, texts: List[str]) -> List[str]:
        """Truncate texts to fit the embedding model's context length."""
        return [
            self.tokenizer.decode(
                self.tokenizer.encode(text)[: self.config.context_length]
            )
            for text in texts
        ]

    def embedding_fn(self) -> Callable[[List[str]], List[List[float]]]:
        """
        Returns a function to compute embeddings using OpenAI, with batching and retries.

        Returns:
            Callable[[List[str]], List[List[float]]]: A callable function for embedding computation.
        """
        def compute_embeddings(texts: List[str]) -> List[List[float]]:
            # Truncate texts to the model's maximum context length
            tokenized_texts = self.truncate_texts(texts)
            print(tokenized_texts)
            embeddings = []

            # Process texts in batches
            for batch in batched(tokenized_texts, self.config.batch_size):
                retry_attempts = 3
                while retry_attempts > 0:
                    try:
                        # Call OpenAI API to generate embeddings
                        response = self.client.embeddings.create(
                            input=batch, model=self.config.model_name
                        )
                        print("response", response)
                        batch_embeddings = [data["embedding"] for data in response["data"]]
                        embeddings.extend(batch_embeddings)
                        break 
                    except Exception as e:
                        retry_attempts -= 1
                        if retry_attempts == 0:
                            raise RuntimeError(f"Failed to generate embeddings: {e}")
                        time.sleep(2 ** (3 - retry_attempts))  # Exponential backoff

            return embeddings

        return compute_embeddings

    @property
    def embedding_dims(self) -> int:
        return self.config.dims
