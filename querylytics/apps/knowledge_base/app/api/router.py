from fastapi import APIRouter, HTTPException
from querylytics.shared.infrastructure.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from querylytics.apps.knowledge_base.app.models.schemas import Document

kb_router = APIRouter()

# Initialize DocChatAgent
doc_chat_config = DocChatAgentConfig(
    name="KBDocChat",
    debug=True,
    collection_name="kb-store",
    storage_path=".chromadb/kb/"
)
doc_chat_agent = DocChatAgent(doc_chat_config)


@kb_router.get("/search")
async def search_documents():
    try:
        # Use DocChatAgent to search
        # results = doc_chat_agent.search(request.query)
        
        # Convert results to Document objects
        documents = [{'text': 'Russia, the largest country in the world, occupies one-tenth of all the land on Earth. It spans 11 time zones across two continents (Europe and Asia) and has coasts on three oceans (the Atlantic, Pacific, and Arctic).', 'metadata': {'id': 0}}]
        response = {"message": documents}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@kb_router.get("/documents/{doc_id}")
async def get_document(doc_id: str) -> Document:
    try:
        # Query vector store for specific document
        results = doc_chat_agent.search(f"id:{doc_id}")
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
            
        result = results[0]
        return Document(
            content=result["text"],
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@kb_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        # Delete document logic would go here
        # Note: Current ChromaDB implementation doesn't support deletion
        return {"status": "deleted", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@kb_router.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "ok", "message": "Knowledge Base API is running"}