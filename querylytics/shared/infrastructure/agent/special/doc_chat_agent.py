from typing import List, Optional
from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig
from querylytics.shared.infrastructure.vector_store.base import VectorStore, VectorStoreConfig
from querylytics.shared.infrastructure.vector_store.chromadb import ChromaDB, ChromaDBConfig
from querylytics.shared.infrastructure.embedding_models.models import OpenAIEmbeddingsConfig
from querylytics.shared.infrastructure.utils.chunks import chunk_text
import logging

logger = logging.getLogger(__name__)

class DocChatAgentConfig(AgentConfig):
    """Configuration for DocChatAgent"""
    retrieval_k: int = 5
    chunk_size: int = 500
    overlap: int = 50
    vecdb_config: VectorStoreConfig = ChromaDBConfig(
        collection_name="doc-chat-store",
        replace_collection=True,
        storage_path=".chromadb/data/",
        embedding=OpenAIEmbeddingsConfig()
    )

class DocChatAgent(Agent):
    """Agent for document search and Q&A"""

    def __init__(self, config: DocChatAgentConfig):
        super().__init__(config)
        self.config = config
        self.vector_store = VectorStore.create(config.vecdb_config)
        logger.info("DocChatAgent initialized with vector store")

    def ingest_documents(self, documents: List[str]) -> None:
        """Ingest documents into vector store"""
        logger.info(f"Ingesting {len(documents)} documents")
        chunks = []
        for doc in documents:
            chunks.extend(chunk_text(
                doc, 
                self.config.chunk_size, 
                self.config.overlap
            ))
        
        self.vector_store.add_documents(chunks)
        logger.info(f"Ingested {len(chunks)} chunks")

    def search(self, query: str) -> List[dict]:
        """Search for relevant documents"""
        logger.info(f"Searching for: {query}")
        results = self.vector_store.query(
            query=query,
            top_k=self.config.retrieval_k
        )
        return results

    def handle_message(self, message: str) -> str:
        """Handle a user query"""
        # Get relevant documents
        results = self.search(message)
        print("results from doc chat agent", results)
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Format context for LLM
        context = "\n\n".join(r["text"] for r in results)
        
        # Generate response using LLM
        prompt = (
            f"Based on the following information, answer the question:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {message}"
        )
        print("doc chat agent prompt", prompt)
        response = self.llm.generate(prompt)
        print("doc chat agent response", response)
        return response

    def get_collection_info(self) -> dict:
        """Get information about the current vector store collection"""
        return {
            "collection": self.config.vecdb_config.collection_name,
            "storage_path": self.config.vecdb_config.storage_path
        }
