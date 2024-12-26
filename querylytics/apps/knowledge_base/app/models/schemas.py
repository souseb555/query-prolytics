from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    """
    Represents a document in the knowledge base
    
    Attributes:
        content: The main text content of the document
        metadata: Additional metadata about the document (e.g., id, title, timestamp)
    """
    content: str
    metadata: Dict[str, Any] = {}

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    limit: Optional[int] = 10

class SearchResponse(BaseModel):
    results: List[dict] 