import logging
from abc import ABC
from typing import Optional, List, Dict, Type, Union, Callable
from pydantic import BaseModel

from language_models.config.base import LLMConfig, LanguageModel
from language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT
from vector_store.base import VectorStoreConfig, VectorStore
from agent.tool_message import ToolMessage
from agent.chat_document import ChatDocument

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for the agent."""
    name: str = "LLM-Agent"
    debug: bool = False
    llm: Optional[LLMConfig] = OpenAIGPTConfig()
    vecdb_config: Optional[VectorStoreConfig] = None

class Agent(ABC):
    """A simplified agent class to interact with LLM and VectorStore."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LanguageModel.create(config.llm)
        self.vecdb = self._init_vecdb(config.vecdb_config)
        if self.config.debug:
            logger.setLevel(logging.DEBUG)

    def _init_vecdb(self, vecdb_config: Optional[VectorStoreConfig]) -> Optional[VectorStore]:
        """Initialize the vector store."""
        if not vecdb_config:
            logger.warning("No VectorStore configuration provided.")
            return None
        logger.info("Initializing VectorStore.")
        return VectorStore(config=vecdb_config)

    def respond(self, prompt: str) -> str:
        """Generate a response for a given prompt."""
        if not self.llm:
            logger.error("LLM not initialized. Cannot generate a response.")
            return "Error: LLM not available."

        try:
            response = self.llm.generate(prompt=prompt)
            logger.debug(f"Generated response: {response}")
            return response.get("message", "Error: No message in response.")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error: Could not generate a response."
