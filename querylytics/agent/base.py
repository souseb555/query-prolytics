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
    llm_config: Optional[LLMConfig] = OpenAIGPTConfig()
    vecdb_config: Optional[VectorStoreConfig] = None

class Agent(ABC):
    """A simplified agent class to interact with LLM and VectorStore."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = self._init_llm(config.llm_config)
        self.vecdb = self._init_vecdb(config.vecdb_config)
        if self.config.debug:
            logger.setLevel(logging.DEBUG)

    def _init_llm(self, llm_config: Optional[LLMConfig]) -> Optional[LanguageModel]:
        """Initialize the language model."""
        if not llm_config:
            logger.warning("No LLM configuration provided.")
            return None
        logger.info("Initializing LLM.")
        return OpenAIGPT(config=llm_config)

    def _init_vecdb(self, vecdb_config: Optional[VectorStoreConfig]) -> Optional[VectorStore]:
        """Initialize the vector store."""
        if not vecdb_config:
            logger.warning("No VectorStore configuration provided.")
            return None
        logger.info("Initializing VectorStore.")
        return VectorStore(config=vecdb_config)

    def handle_message(self, msg: str) -> str:
        """Process a message and return a response."""
        if not self.llm:
            logger.error("LLM not initialized. Cannot process message.")
            return "LLM not available."
        logger.info(f"Processing message: {msg}")
        return "Response to: " + msg

    def respond(self, msg: str) -> str:
        """Generate a response for a given message."""
        response = self.handle_message(msg)
        logger.debug(f"Generated response: {response}")
        return response
