from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    limit: Optional[int] = 10

class SearchResponse(BaseModel):
    results: List[dict] 