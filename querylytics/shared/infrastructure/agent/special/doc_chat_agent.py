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
        collection_name="reports-store",
        replace_collection=True,
        storage_path="./../.chromadb/reports/",
        embedding=OpenAIEmbeddingsConfig()
    )

class DocChatAgent(Agent):
    """Agent for document search and Q&A"""

    def __init__(self, config: DocChatAgentConfig):
        super().__init__(config)
        self.config = config
        self.vector_store = VectorStore.create(config.vecdb_config)
        logger.info("DocChatAgent initialized with vector store")

    def ingest_documents(self, reports: List[dict]) -> None:
        """Ingest reports into vector store
        
        Args:
            reports: List of dictionaries containing:
                - title: Report title
                - description: Report description
                - metadata: Optional additional metadata
        """
        logger.info(f"Ingesting {len(reports)} reports")
        chunks = []
        metadatas = []
        ids = []
        
        for idx, report in enumerate(reports):
            # Create chunks from the description
            report_chunks = chunk_text(
                report['description'], 
                self.config.chunk_size, 
                self.config.overlap
            )
            
            # Add chunks with their metadata
            for chunk_idx, chunk in enumerate(report_chunks):
                chunks.append(chunk)
                chunk_metadata = {
                    'title': report['title'],
                    'description': report['description'],
                    'chunk_index': chunk_idx,
                    **report.get('metadata', {})
                }
                metadatas.append(chunk_metadata)
                ids.append(f"report_{idx}_chunk_{chunk_idx}")
        
        self.vector_store.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Ingested {len(chunks)} chunks from {len(reports)} reports")

    def search(self, query: str, filter_dict: Optional[dict] = None) -> List[dict]:
        """Optimized search for business reports
        
        Args:
            query: Search query
            filter_dict: Optional metadata filters for report categories/dates
        """
        logger.info(f"Searching for: {query}")
        
        # Primary search: Hybrid search with metadata filtering
        results = self.vector_store.query(
            query=query,
            top_k=self.config.retrieval_k,
            include_metadata=True,
            where=filter_dict,
            alpha=0.7  # Bias towards semantic search (70%) while keeping some keyword matching (30%)
        )
        # Apply distance threshold to ensure quality
        filtered_results = [
            result for result in results 
            if result.get('similarity_score', 0) >= 0.6  # Only keep high-confidence matches
        ]
        
        return filtered_results

    def handle_message(self, message: str) -> str:
        """Handle a user query with business context"""
        # Extract potential time period and report type from query
        # Example: "What was our CAC in Q2?" or "Show me recent MRR trends"
        filters = self._extract_query_filters(message)
        
        results = self.search(message, filters)
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Format context with business-specific structure
        contexts = []
        for r in results:
            context = (
                f"Report: {r['metadata']['title']}\n"
                f"Period: {r['metadata'].get('period', 'N/A')}\n"
                f"Department: {r['metadata'].get('department', 'N/A')}\n"
                f"Content: {r['text']}\n"
            )
            contexts.append(context)
        
        full_context = "\n\n".join(contexts)
        
        prompt = (
            f"You are a business analyst. Based on the following report sections, "
            f"provide a clear and concise answer with relevant metrics and insights:\n\n"
            f"Context:\n{full_context}\n\n"
            f"Question: {message}"
        )
        response = self.llm.generate(prompt)
        return response

    def _extract_query_filters(self, query: str) -> Optional[dict]:
        """Extract relevant filters from the query"""
        filters = {}
        
        # Example time period detection
        time_periods = ['Q1', 'Q2', 'Q3', 'Q4', '2023', '2024']
        for period in time_periods:
            if period.lower() in query.lower():
                filters['period'] = period
        
        # Example report type detection
        report_types = {
            'win loss': 'Sales Performance',
            'cac': 'Customer Acquisition',
            'mrr': 'Revenue',
            'lead': 'Marketing',
        }
        for key, value in report_types.items():
            if key.lower() in query.lower():
                filters['department'] = value
        
        return filters if filters else None

    def inspect_database(self) -> dict:
        """Inspect the contents of the vector database
        
        Returns:
            Dictionary containing collection stats and sample entries
        """
        collection = self.vector_store.client.get_collection(
            name=self.config.vecdb_config.collection_name
        )
        
        result = collection.get()
        
        return {
            "total_documents": len(result['ids']),
            "sample_entries": [
                {
                    "id": id,
                    "metadata": metadata,
                    "content": content[:200] + "..." 
                }
                for id, metadata, content in zip(
                    result['ids'][:5],
                    result['metadatas'][:5],
                    result['documents'][:5]
                )
            ]
        }
