import logging
import chromadb
from typing import List, Tuple, Sequence
from querylytics.shared.infrastructure.vector_store.base import VectorStore, VectorStoreConfig, Document
from querylytics.shared.infrastructure.embedding_models.base import EmbeddingModelsConfig
from querylytics.shared.infrastructure.embedding_models.models import OpenAIEmbeddingsConfig
from chromadb import PersistentClient, Settings

logger = logging.getLogger(__name__)

class ChromaDBConfig(VectorStoreConfig):
    collection_name: str = "temp"
    storage_path: str = ".chroma/data"
    embedding: EmbeddingModelsConfig = OpenAIEmbeddingsConfig()
    host: str = "127.0.0.1"
    port: int = 6379

class ChromaDB(VectorStore):
    """Implementation of VectorStore using ChromaDB."""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(config)
        self.config = config
        self.client = PersistentClient(
            path=config.storage_path,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], metadatas: List[dict], ids: List[str]) -> None:
        """Add documents to the ChromaDB collection.
        
        Args:
            documents: List of document contents
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
        """
        logger.info(f"Adding {len(documents)} documents to collection '{self.config.collection_name}'.")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

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
        
    def query(
        self, 
        query: str, 
        top_k: int = 5, 
        include_metadata: bool = True,
        where: dict = None,
        alpha: float = 0.5
    ) -> List[dict]:
        """
        Query the vector store for similar documents.

        Args:
            query: The input query string
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            where: Optional filter conditions for metadata
            alpha: Balance between semantic (1.0) and keyword search (0.0)

        Returns:
            List of dictionaries containing 'text', 'metadata', and 'similarity_score'
        """
        logger.info(f"Querying for top {top_k} similar documents for: {query}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        formatted_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted_results.append({
                'text': doc,
                'metadata': metadata,
                'similarity_score': (1 - distance/2)  # Convert cosine distance to similarity score
            })
        return formatted_results
