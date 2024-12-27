from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

class LLMConfig(BaseModel):
    """Base configuration for all LLMs."""
    model_type: str = "openai"
    timeout: int = 20
    max_tokens: int = 1024
    temperature: float = 0.7

class LanguageModel(ABC):
    """Abstract base class for Language Models."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @staticmethod
    def create(config: Optional[LLMConfig]) -> Optional["LanguageModel"]:
        """
        Factory method to create a LanguageModel instance.
        Args:
            config: LLMConfig object specifying the configuration.
        Returns:
            An instance of a specific LanguageModel subclass or None.
        """
        from querylytics.shared.infrastructure.language_models.openai_gpt import OpenAIGPT
        if config is None:
            return None

        if config.model_type == "openai":
            return OpenAIGPT(config)

        raise ValueError(f"Unsupported model type: {config.model_type}")

    @abstractmethod
    def generate(
        self, 
        prompt: str,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Dict] = None
    ) -> dict:
        pass

class Role(str, Enum):
    """
    Possible roles for a message in a chat.
    """

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class LLMMessage(BaseModel):
    """
    Represents a message in the interaction history sent to the LLM API.
    It can be a user message, an assistant response, or a tool/function call.
    """

    role: str  # Role can be "user", "assistant", or "system".
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def api_dict(self, has_system_role: bool = True) -> Dict[str, Any]:
        """
        Convert the message to a dictionary for API calls, excluding irrelevant fields.

        Args:
            has_system_role (bool): Treat "system" role as "user" if False.
        Returns:
            Dict[str, Any]: Dictionary representation for API usage.
        """
        d = self.dict(exclude_none=True)
        if not has_system_role and d.get("role") == "system":
            d["role"] = "user"
            d["content"] = f"[SYSTEM MESSAGE]:\n{d['content']}"

        # Ensure API compatibility
        if "function_call" in d:
            d["function_call"]["arguments"] = json.dumps(d["function_call"]["arguments"])
        if "tool_calls" in d:
            for tc in d["tool_calls"]:
                if "arguments" in tc.get("function", {}):
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])

        # Drop unnecessary fields
        d.pop("timestamp", None)
        d.pop("tool_call_id", None)
        return d

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        content = (
            f"FUNC: {json.dumps(self.function_call)}"
            if self.function_call
            else self.content
        )
        name_info = f" ({self.name})" if self.name else ""
        return f"{self.role}{name_info}: {content}"