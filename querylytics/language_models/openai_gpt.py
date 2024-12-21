import logging
import os

from typing import Optional, Dict
from enum import Enum

from language_models.config.base import LLMConfig, LanguageModel
from openai import OpenAI, OpenAIError

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")



class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT4 = "gpt-4"
    GPT4_32K = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4o = "gpt-4o"
    GPT4o_MINI = "gpt-4o-mini"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"

class OpenAIGPTConfig(LLMConfig):
    type: str = "openai"
    api_key: str = OPENAI_API_KEY 
    organization: str = ""
    api_base: Optional[str] = None
    timeout: int = 20
    temperature: float = 0.2
    seed: Optional[int] = 42
    chat_model: str = OpenAIChatModel.GPT4_TURBO
    completion_model: str = OpenAIChatModel.GPT4_TURBO
    supports_json_schema: Optional[bool] = None


class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    def __init__(self, config: OpenAIGPTConfig):
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key)  # Initialize the client

    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate a response for a given prompt.
        Args:
            prompt: The input prompt for the model.
        Returns:
            A dictionary containing the response text and additional metadata.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            if hasattr(response.choices[0], "message"):
                message = response.choices[0].message.content.strip()
            else:
                message = response.choices[0].text.strip()
            return {"message": message, "usage": getattr(response, "usage", {})}
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return {"message": "Error: OpenAI API error occurred."}
        except TimeoutError as e:
            logger.error(f"Request timed out: {e}")
            return {"message": "Error: Request timed out."}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"message": "Error: An unexpected error occurred."}
    
    