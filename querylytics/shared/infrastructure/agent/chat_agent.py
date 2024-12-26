import copy
import json
from typing import Any, Dict, List, Optional, Type, Union

from agent.base import Agent, AgentConfig
from agent.tool_message import ToolMessage
from language_models.config.base import LLMMessage, Role

import logging
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChatAgentConfig(AgentConfig):
    """
    Configuration for ChatAgent
    Attributes:
        system_message: Default system message to initialize conversations.
        use_tools: Enable or disable tool usage.
    """
    system_message: str = "You are a helpful assistant."
    use_tools: bool = True
    tools: List[Type[ToolMessage]] = []


class ChatAgent(Agent):
    """
    Chat Agent for interacting with LLMs in a conversational style.
    """

    def __init__(
        self, config: ChatAgentConfig = ChatAgentConfig(), task: Optional[List[LLMMessage]] = None
    ):
        super().__init__(config)
        self.config = config
        self.message_history: List[LLMMessage] = []
        self.system_message = config.system_message
        self.llm_functions_map = {}  # For managing tool functions
        self.llm_tools_known = set()
        self.llm_tools_handled = set()
        self.llm_tools_usable = set()
        self.enabled_tools = set()

        self.init_state()

        if task:
            self.message_history.extend(task)

    def init_state(self) -> None:
        """Initialize agent state."""
        self.message_history = []
        self.clear_history()

    def clear_history(self) -> None:
        """Clear the message history."""
        self.message_history = []

    def add_message(self, role: Role, content: str) -> None:
        """Add a new message to the history."""
        self.message_history.append(LLMMessage(role=role, content=content))

    def respond(self, message: str) -> str:
        """Generate a response for a given user message."""
        self.add_message(Role.USER, message)

        response = self.llm_response()
        self.add_message(Role.ASSISTANT, response)
        return response

    def llm_response(self) -> str:
        """
        Placeholder method for LLM response.
        Replace with actual LLM call logic.
        """
        if not self.llm:
            return "LLM not configured."
        return "LLM response here."

    def enable_message(
        self,
        message_class: Optional[Union[Type[ToolMessage], List[Type[ToolMessage]]]],
        use: bool = True,
        handle: bool = True,
        force: bool = False,
        require_recipient: bool = False,
        include_defaults: bool = True,
    ) -> None:
        """
        Add a tool (message class) to the agent, enabling usage and/or handling.

        Args:
            message_class: The ToolMessage class or list of such classes to enable.
            use: Allow the agent to use this tool.
            handle: Allow the agent to handle this tool.
            force: Force the agent to prioritize this tool.
            require_recipient: Require a recipient when using the tool.
            include_defaults: Include fields with default values in tool schema.
        """
        if message_class is None:
            logger.warning("No tool class provided to enable.")
            return

        if isinstance(message_class, list):
            for mc in message_class:
                self.enable_message(
                    mc,
                    use=use,
                    handle=handle,
                    force=force,
                    require_recipient=require_recipient,
                    include_defaults=include_defaults,
                )
            return

        if require_recipient:
            message_class = message_class.require_recipient()

        if handle:
            self.enabled_tools.add(message_class)
            logger.info(f"Enabled handling for tool: {message_class.__name__}")

        if use:
            if hasattr(message_class, "allow_llm_use") and not message_class.allow_llm_use:
                logger.warning(
                    f"Tool {message_class.__name__} does not allow LLM use."
                )
            else:
                self.config.tools.append(message_class)
                logger.info(f"Enabled usage for tool: {message_class.__name__}")

        if force:
            self.config.tools.insert(0, message_class)
            logger.info(f"Forced priority for tool: {message_class.__name__}")

        if include_defaults:
            logger.debug(f"Including default values for {message_class.__name__}.")

    def disable_tool(self, tool: Type[ToolMessage]) -> None:
        """Disable a specific tool."""
        tool_name = tool.default_value("request")
        self.llm_tools_known.discard(tool_name)
        self.llm_tools_handled.discard(tool_name)
        self.llm_tools_usable.discard(tool_name)
        logger.info(f"Disabled tool: {tool_name}")
