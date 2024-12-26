from typing import Dict, Optional

from pydantic import BaseModel


class ToolMessage(BaseModel):
    """
    Defines the structure of a tool message from an LLM to an agent.
    Represents intents such as:
    - Requesting data or information.
    - Calling a specific function or method.
    """

    request: str
    purpose: str = ""
    id: str = "" 
    requires_function_call: bool = False
    function_schema: Optional[Dict] = None

    @classmethod
    def get_request_type(cls) -> str:
        """Get the request type for this tool"""
        request = getattr(cls, 'request', None)
        if not request:
            request = cls.__name__.lower().replace('tool', '')
        return request
