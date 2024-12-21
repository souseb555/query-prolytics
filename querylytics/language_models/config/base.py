from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, Optional

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
        from language_models.openai_gpt import OpenAIGPT
        if config is None:
            return None

        if config.model_type == "openai":
            return OpenAIGPT(config)

        raise ValueError(f"Unsupported model type: {config.model_type}")

    @abstractmethod
    def generate(self, prompt: str) -> dict:
        pass