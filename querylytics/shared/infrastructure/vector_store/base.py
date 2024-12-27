from querylytics.shared.infrastructure.pydantic_v1 import BaseSettings
from abc import ABC


import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Sequence

from querylytics.shared.infrastructure.embedding_models.models import OpenAIEmbeddingsConfig
from querylytics.shared.infrastructure.embedding_models.base import EmbeddingModelsConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VectorStoreConfig(BaseSettings):
    
    collection_name: str = "reports-store"
    replace_collection: bool = False
    storage_path: str = "chromadb/reports/"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    batch_size: int = 200
    timeout: int = 60
    host: str = "127.0.0.1"
    port: int = 6379


class Document:
    """Simple representation of a document."""
    def __init__(self, content: str, metadata: Optional[dict] = None):
        self.content = content
        self.metadata = metadata or {}


class VectorStore(ABC):
    """Abstract base class for a vector store."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    @staticmethod
    def create(config: VectorStoreConfig) -> Optional["VectorStore"]:
        """Factory method to create a VectorStore instance."""
        from querylytics.shared.infrastructure.vector_store.chromadb import ChromaDB, ChromaDBConfig
        if isinstance(config, ChromaDBConfig):
            return ChromaDB(config)

    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[dict], ids: List[str]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document contents
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
        """
        pass

    @abstractmethod
    def similar_texts_with_scores(
        self, text: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Find similar documents to a query."""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections in the vector store."""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        """Create a collection."""
        pass
    
    def set_collection(self, collection_name: str, replace: bool = False) -> None:
        """
        Set the current collection to the given collection name.
        Args:
            collection_name (str): Name of the collection.
            replace (bool, optional): Whether to replace the collection if it
                already exists. Defaults to False.
        """

        self.config.collection_name = collection_name
        self.config.replace_collection = replace

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        pass
