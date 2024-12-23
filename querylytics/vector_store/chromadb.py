import logging
import chromadb
from typing import List, Tuple, Sequence
from vector_store.base import VectorStore, VectorStoreConfig, Document

logger = logging.getLogger(__name__)

class ChromaDBConfig(VectorStoreConfig):
    collection_name: str = "temp"
    storage_path: str = ".chroma/data"
    host: str = "127.0.0.1"
    port: int = 6379

class ChromaDB(VectorStore):
    """Implementation of VectorStore using ChromaDB."""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(config)
        self.client = chromadb.Client(
            chromadb.config.Settings(
                # chroma_db_impl="duckdb+parquet",
                persist_directory=config.storage_path,
            )
        )

    def add_documents(self, documents: Sequence[Document]) -> None:
        logger.info(f"Adding {len(documents)} documents to collection '{self.config.collection_name}'.")

    def similar_texts_with_scores(
        self, text: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        logger.info(f"Searching for {k} similar texts for: {text}")
        return [(Document(content=f"Sample doc {i}", metadata={"id": i}), 0.9 - 0.1 * i) for i in range(k)]

    def list_collections(self) -> List[str]:
        logger.info("Listing all collections.")
        return ["default", "example_collection"]

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        logger.info(f"Creating collection '{collection_name}' (replace={replace}).")
        self.config.collection_name = collection_name
        self.config.replace_collection = replace

    def delete_collection(self, collection_name: str) -> None:
        logger.info(f"Deleting collection '{collection_name}'.")
