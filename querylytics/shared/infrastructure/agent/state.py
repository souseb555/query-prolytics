from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Any, Set
from agent.tool_message import ToolMessage

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    TOOL_EXECUTION = "tool_execution"
    ERROR = "error"
    DONE = "done"

@dataclass
class AgentContext:
    current_tool: Optional[ToolMessage] = None
    tool_history: List[ToolMessage] = field(default_factory=list)
    conversation_history: List[tuple] = field(default_factory=list)
    tried_tools: Set[str] = field(default_factory=set)
    last_result: Any = None
    error_count: int = 0
    total_llm_token_cost: float = 0.0
    total_llm_token_usage: int = 0