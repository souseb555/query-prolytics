import logging
from abc import ABC
from typing import Optional, List, Dict, Type, Any
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass, field
from colorama import Fore, Style, init

from querylytics.shared.infrastructure.language_models.config.base import LLMConfig, LanguageModel
from querylytics.shared.infrastructure.language_models.openai_gpt import OpenAIGPTConfig
from querylytics.shared.infrastructure.vector_store.base import VectorStoreConfig, VectorStore
from querylytics.shared.infrastructure.agent.tool_message import ToolMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init(autoreset=True)  # Initialize colorama

class AgentState(Enum):
    """Agent states as an Enum"""
    IDLE = "idle"
    PROCESSING = "processing"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    DONE = "done"
    RETRIEVING = "retrieving"
    WAITING_FEEDBACK = "waiting_feedback"
    PROBING = "probing"

@dataclass
class AgentContext:
    current_tool: Optional[ToolMessage] = None
    tool_history: List[ToolMessage] = field(default_factory=list)
    conversation_history: List[tuple] = field(default_factory=list)
    error_count: int = 0
    capabilities: List[str] = field(default_factory=list)

class AgentConfig(BaseModel):
    name: str = "LLM-Agent"
    debug: bool = False
    llm: Optional[LLMConfig] = OpenAIGPTConfig()
    vecdb_config: Optional[VectorStoreConfig] = None
    capabilities: List[str] = []

class Agent(ABC):
    # State colors mapping
    STATE_COLORS = {
        AgentState.IDLE: Fore.WHITE,
        AgentState.PROCESSING: Fore.YELLOW,
        AgentState.TOOL_EXECUTION: Fore.BLUE,
        AgentState.ERROR: Fore.RED,
        AgentState.DONE: Fore.GREEN,
        AgentState.RETRIEVING: Fore.CYAN,
        AgentState.WAITING_FEEDBACK: Fore.MAGENTA,
        AgentState.PROBING: Fore.LIGHTBLUE_EX
    }

    # Emoji indicators for different states
    STATE_EMOJI = {
        AgentState.IDLE: "⚪",
        AgentState.PROCESSING: "⚡",
        AgentState.TOOL_EXECUTION: "🔧",
        AgentState.ERROR: "❌",
        AgentState.DONE: "✅",
        AgentState.RETRIEVING: "🔍",
        AgentState.WAITING_FEEDBACK: "⏳",
        AgentState.PROBING: "🔄"
    }

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LanguageModel.create(config.llm)
        self.tools: Dict[str, Type[ToolMessage]] = {}
        self.vecdb = self._init_vecdb(config.vecdb_config)
        self._state = AgentState.IDLE
        self.context = AgentContext()
        self.context.capabilities = config.capabilities

    @property
    def state(self) -> AgentState:
        """Get current state"""
        return self._state

    def transition_to(self, new_state: AgentState) -> None:
        """Transition to a new state with color visualization"""
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state type: {type(new_state)}. Must be AgentState enum.")
        
        old_state = self._state
        self._state = new_state
        
        # Get colors and emojis for old and new states
        old_color = self.STATE_COLORS.get(old_state, Fore.WHITE)
        new_color = self.STATE_COLORS.get(new_state, Fore.WHITE)
        old_emoji = self.STATE_EMOJI.get(old_state, "")
        new_emoji = self.STATE_EMOJI.get(new_state, "")
        
        # Log the transition with colors and emojis
        logger.info(
            f"{old_color}{old_emoji} {self.config.name}: "
            f"{old_state.value}{Style.RESET_ALL} → "
            f"{new_color}{new_emoji} {new_state.value}"
        )

    def _init_vecdb(self, vecdb_config: Optional[VectorStoreConfig]) -> Optional[VectorStore]:
        if not vecdb_config:
            return None
        return VectorStore.create(config=vecdb_config)

    def register_tool(self, tool_class: Type[ToolMessage]) -> None:
        """Register a tool with the agent"""
        if self.state != AgentState.IDLE:
            logger.warning("Registering tool while not IDLE")
        
        # Get the request type with better error handling
        try:
            tool_name = tool_class.default_value("request")
            if not tool_name:
                # Fallback to class name if request is empty
                tool_name = tool_class.__name__.lower()
                logger.warning(f"Tool {tool_class.__name__} has no request value, using class name")
        except Exception as e:
            tool_name = tool_class.__name__.lower()
            logger.error(f"Error getting request type for {tool_class.__name__}: {e}")
        
        logger.info(f"Registering tool {tool_class.__name__} with name '{tool_name}'")
        self.tools[tool_name] = tool_class
        logger.info(f"Current tools: {list(self.tools.keys())}")

    def handle_message(self, message: str) -> str:
        """Main message handling flow with color indicators"""
        try:
            self.transition_to(AgentState.PROCESSING)
            
            # Get LLM's initial response
            logger.info(f"{Fore.YELLOW}⚡ Processing message with LLM...{Style.RESET_ALL}")
            llm_response = self.llm_response(message)
            
            # Try to execute any tools
            logger.info(f"{Fore.BLUE}🔧 Checking for tool execution...{Style.RESET_ALL}")
            tool_result = self._execute_tools(llm_response)
            if tool_result:
                logger.info(f"{Fore.GREEN}✅ Tool execution successful{Style.RESET_ALL}")
                return self.llm_response(f"Tool result: {tool_result}")
            
            return llm_response
            
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Error in message handling: {str(e)}{Style.RESET_ALL}")
            self.transition_to(AgentState.ERROR)
            return f"Error in message handling: {str(e)}"

    def _execute_tools(self, llm_response: str) -> Optional[str]:
        """Tool execution with color indicators"""
        try:
            specific_tool = self._get_specific_tool(llm_response)
            if specific_tool:
                logger.info(f"{Fore.CYAN}🔧 Using tool: {specific_tool.__name__}{Style.RESET_ALL}")
                result = self._try_tool(llm_response, specific_tool)
                if result:
                    return result

            for tool_class in self.tools.values():
                logger.info(f"{Fore.CYAN}🔧 Trying tool: {tool_class.__name__}{Style.RESET_ALL}")
                result = self._try_tool(llm_response, tool_class)
                if result:
                    return result
            
            return None
            
        except Exception as e:
            logger.warning(f"{Fore.RED}❌ Error in tool execution: {e}{Style.RESET_ALL}")
            return None

    def llm_response(self, message: str) -> str:
        """Get response from LLM"""
        try:
            if not self.llm:
                logger.warning("LLM not configured")
                return "LLM not configured."
                
            tool_context = self.get_tool_context()
            augmented_message = f"{message}\n\nAvailable tools:\n{tool_context}"
            print("augmented message", augmented_message)
            llm_result = self.llm.generate(augmented_message)
            print("llm result", llm_result)
            if not llm_result:
                logger.warning("No response from LLM")
                return "Failed to get LLM response"
                
            # Extract message from response dictionary
            response_text = llm_result.get("message", "No response generated")
            
            # Update conversation history
            self.context.conversation_history.append(("user", message))
            self.context.conversation_history.append(("assistant", response_text))
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in llm_response: {e}")
            return f"Error getting LLM response: {str(e)}"

    def parse_tools(self, llm_response: str) -> List[ToolMessage]:
        """Parse all potential tool requests from LLM response"""
        if not llm_response:
            return []
        
        tool_requests = []
        
        for tool_name, tool_class in self.tools.items():
            if tool_name.lower() in llm_response.lower():
                try:
                    params = self._extract_tool_params(llm_response, tool_class)
                    tool = tool_class(**params)
                    tool_requests.append(tool)
                except Exception as e:
                    logger.warning(f"Failed to parse tool {tool_name}: {e}")
            
        return tool_requests

    def process_tool_result(self, result: Any) -> str:
        """Process the result of a tool execution"""
        if not result:
            return None
        return self.llm_response(f"Tool result: {result}")

    def get_tool_context(self) -> str:
        """Get formatted list of available tools"""
        tool_descriptions = []
        logger.info(f"Getting tool context for {len(self.tools)} tools")
        for name, tool in self.tools.items():
            desc = tool.__doc__ or "No description available"
            desc = desc.strip()  # Remove any extra whitespace
            tool_descriptions.append(f"- {name}: {desc}")
            logger.info(f"Added tool to context: {name}")
        return "\n".join(tool_descriptions)

    def get_state_info(self) -> dict:
        """Get current state information when needed"""
        return {
            "state": self.state.value,
            "tool_history": [t.__class__.__name__ for t in self.context.tool_history],
            "error_count": self.context.error_count
        }

    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.context.capabilities

    def add_capability(self, capability: str) -> None:
        """Add a new capability to the agent"""
        if capability not in self.context.capabilities:
            self.context.capabilities.append(capability)
            logger.info(f"Added capability: {capability}")
    
    def _get_specific_tool(self, message: str) -> Optional[Type[ToolMessage]]:
        """Check if message specifically requests a tool"""
        message_lower = message.lower()
        for tool_class in self.tools.values():
            # Get the request type from the tool class
            tool_name = tool_class.get_request_type().lower()
            if tool_name in message_lower:
                return tool_class
        return None

    def _try_tool(self, message: str, tool_class: Type[ToolMessage]) -> Optional[str]:
        """Try using a specific tool"""
        try:
            params = self._extract_tool_params(message, tool_class)
            tool = tool_class(**params)
            
            handler_name = tool_class.__name__.lower()
            handler = getattr(self, handler_name, None)
            
            if not handler:
                logger.warning(f"No handler for {tool_class.__name__}")
                return None
            
            # If tool requires function calling, use that format
            if tool.requires_function_call:
                response = self.llm.generate(
                    prompt=[{
                        "role": "user",
                        "content": message
                    }],
                    functions=[tool.function_schema],
                    function_call={"name": tool.function_schema["name"]}
                )
                result = handler(tool, response)
            else:
                # Standard tool execution
                result = handler(tool)
                
            if self._is_valid_result(result):
                logger.info(f"Successfully got result from {tool_class.__name__}")
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Error executing {tool_class.__name__}: {e}")
            return None

    def _is_valid_result(self, result: Any) -> bool:
        """Check if the result is valid"""
        if result is None:
            return False
        
        if isinstance(result, str):
            error_patterns = [
                "error",
                "not found",
                "no relevant",
                "couldn't find",
                "not configured"
            ]
            return not any(pattern in result.lower() for pattern in error_patterns)
        
        return bool(result)
    
