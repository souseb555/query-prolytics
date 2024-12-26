from typing import Any, List

from agent.chat_agent import ChatAgent  
from agent.chat_document import ChatDocument 
from agent.tool_message import ToolMessage 


class AgentDoneTool(ToolMessage):
    """Signal that the current task is done with optional content and tools."""

    request: str = "agent_done_tool"
    content: Any = None
    tools: List[ToolMessage] = []

    def response(self, agent: ChatAgent) -> ChatDocument:
        """Create a response indicating the task is done."""
        return agent.create_agent_response(
            content=str(self.content or ""),
            content_any=self.content,
            tool_messages=[self] + self.tools,
        )


class DoneTool(ToolMessage):
    """Signal the task is completed with a result."""

    request: str = "done_tool"
    content: str = ""

    def response(self, agent: ChatAgent) -> ChatDocument:
        """Return a completion response."""
        return agent.create_agent_response(
            content=self.content,
            content_any=self.content,
            tool_messages=[self],
        )


class ResultTool(ToolMessage):
    """Wrap arbitrary results and signal task completion."""

    request: str = "result_tool"

    def handle(self) -> AgentDoneTool:
        """Wrap result in AgentDoneTool."""
        return AgentDoneTool(tools=[self])


class FinalResultTool(ToolMessage):
    """Wrap results and signal task completion for all parent tasks."""

    request: str = "final_result_tool"


class PassTool(ToolMessage):
    """Pass the current message to be handled by another agent."""

    request: str = "pass_tool"

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        """Pass the current task to another agent."""
        return agent.forward_task(chat_doc)


class ForwardTool(PassTool):
    """Forward the current message to a specified agent."""

    request: str = "forward_tool"
    agent: str

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        """Forward the task to a specified recipient."""
        forwarded_doc = super().response(agent, chat_doc)
        forwarded_doc.metadata.recipient = self.agent
        return forwarded_doc


class SendTool(ToolMessage):
    """Send a message to a specified agent."""

    request: str = "send_tool"
    to: str
    content: str = ""

    def response(self, agent: ChatAgent) -> ChatDocument:
        """Send a message to the specified agent."""
        return agent.create_agent_response(
            content=self.content,
            recipient=self.to,
        )


class DonePassTool(PassTool):
    """Signal task completion and pass the current message."""

    request: str = "done_pass_tool"

    def response(self, agent: ChatAgent, chat_doc: ChatDocument) -> ChatDocument:
        """Combine DoneTool and PassTool behavior."""
        new_doc = super().response(agent, chat_doc)
        return AgentDoneTool(content=new_doc.content, tools=agent.get_tool_messages(new_doc))
