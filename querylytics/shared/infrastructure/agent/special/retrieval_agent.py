from typing import List, Dict, Any, Optional
from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig, AgentState
from querylytics.shared.infrastructure.agent.tool_message import ToolMessage
from querylytics.shared.infrastructure.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from querylytics.shared.infrastructure.agent.special.doc_chat_agent import DocChatAgentConfig, DocChatAgent
import logging
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class APISearchTool(ToolMessage):
    """Search external API for data"""
    request: str = "api_search"
    endpoint: str
    query_params: Dict[str, Any]
    requires_function_call: bool = True
    function_schema: Dict = {
        "name": "search_documents",
        "description": "Search external documents",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }

class VectorSearchTool(ToolMessage):
    """Search vector database for relevant documents"""
    request: str = "vector_search"
    query: str
    top_k: int = 5
    requires_function_call: bool = False


class RetrievalAgentConfig(AgentConfig):
    """Configuration for RetrievalAgent"""
    api_base_url: str = ""
    api_key: str = ""
    retrieval_k: int = 5
    min_relevance_score: float = 0.7
    doc_chat_config: Optional[DocChatAgentConfig] = DocChatAgentConfig()
    system_message: str = """
    You are a retrieval agent with access to two tools:
    - vectorsearchtool: For searching internal documents
    - apisearchtool: For searching external sources

    Before using any tool:
    1. Explain which tool you're choosing and why, for example:
       "Let me search our internal documents for this information..."
       or 
       "This seems like external information, I'll check external sources..."

    2. Then use the chosen tool by clearly stating its name and parameters.

    If you're asked to try a different tool, explain your new approach before using it.

    All responses from tools must be formatted as:
    SOURCE: [source_name]
    EXTRACT: [First 3 words ... last 3 words]
    """
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        chat_model=OpenAIChatModel.GPT4,
        temperature=0.7
    )

class RetrievalAgent(Agent):
    """Agent that combines RAG capabilities with API calls"""
    
    def __init__(self, config: RetrievalAgentConfig):
        super().__init__(config)
        self.config = config
        self.doc_chat_agent = DocChatAgent(config.doc_chat_config) if config.doc_chat_config else None
        
        # Register tools
        self.register_tool(VectorSearchTool)
        self.register_tool(APISearchTool)

    def _extract_tool_params(self, message: str, tool_class) -> dict:
        """Extract tool parameters from message"""
        if tool_class == VectorSearchTool:
            return {"query": message, "top_k": self.config.retrieval_k}
        elif tool_class == APISearchTool:
            return {
                "endpoint": "http://0.0.0.0:8000/kb/search",
                "query_params": {"q": message}
            }

    def vectorsearchtool(self, tool: VectorSearchTool) -> str:
        """Execute vector store search using DocChat agent"""
        if not self.doc_chat_agent:
            return "DocChat agent not configured"
            
        result = self.doc_chat_agent.search(tool.query)
        return result

    def apisearchtool(self, tool: APISearchTool) -> str:
        """Execute API search through HTTP request"""
        try:
            logger.info(f"Starting API search with tool: {tool}")
            
            # Make the API request using the query from the function call
            response = requests.get(
                tool.endpoint,
                params=tool.query_params,
                headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else None
            )
            
            response.raise_for_status()
            return response.json().get("message") or str(response.json())

        except Exception as e:
            logger.error(f"Error in API search: {e}")
            return None

    def llm_response(self, message: str) -> str:
        """Override llm_response to handle tool results and retries"""
        try:
            if not self.llm:
                return "LLM not configured."

            # First attempt
            tool_result = self._try_get_answer(message)
            if tool_result:
                return tool_result

            # If first attempt failed, try other tool
            logger.info("First tool attempt failed, trying alternative tool...")
            retry_message = (
                f"The previous tool didn't provide useful results for: '{message}'. "
                "Please try the other available tool and explain your new approach."
            )
            alternative_result = self._try_get_answer(retry_message)
            if alternative_result:
                return alternative_result

            # If both tools failed
            return (
                "I've tried both internal and external searches but couldn't find "
                "relevant information to answer your question."
            )

        except Exception as e:
            logger.error(f"Error in llm_response: {e}")
            return f"Error getting response: {str(e)}"

    def _try_get_answer(self, message: str) -> Optional[str]:
        """Helper method to try getting an answer using tools"""
        # Get the function schema from our tool
        functions = [APISearchTool.function_schema]
        
        # Call LLM with the function schema
        llm_result = self.llm.generate(
            message,
            functions=functions,  # Pass the available functions
            function_call={"name": "search_documents"}  # Force it to use our function
        )
        
        if not llm_result:
            return None
            
        # Get the function call details from LLM
        function_call = llm_result.get("function_call")
        if not function_call:
            return None
            
        # Extract the query from the function call
        try:
            query = json.loads(function_call["arguments"])["query"]
            
            # Create and execute the tool with the query
            tool = APISearchTool(
                endpoint="http://0.0.0.0:8000/kb/search",
                query_params={"q": query}
            )
            
            return self.apisearchtool(tool)
            
        except Exception as e:
            logger.error(f"Error processing function call: {e}")
            return None
        
    async def handle_message_parallel(self, message: str) -> str:
        try:
            if not self.llm:
                logger.error("LLM not configured")
                return "LLM not configured"

            self.transition_to(AgentState.PROCESSING)
            
            # Step 1: Gather results from both tools in parallel
            executor = ThreadPoolExecutor()
            try:
                vector_task = asyncio.create_task(
                    self._async_tool_execution(executor, VectorSearchTool, message)
                )
                api_task = asyncio.create_task(
                    self._async_tool_execution(executor, APISearchTool, message)
                )

                results = await asyncio.gather(
                    vector_task, 
                    api_task, 
                    return_exceptions=True
                )
            finally:
                executor.shutdown(wait=False)

            # Step 2: Process and structure the results
            structured_results = {
                "vector_search": self._process_result(results[0], "vector_search"),
                "api_search": self._process_result(results[1], "api_search")
            }
            print("structured_results", structured_results)

            # Step 3: Ask LLM to evaluate and select the most relevant information
            selection_prompt = self._create_selection_prompt(message, structured_results)
            print("selection_prompt", selection_prompt)
            selection_result = self.llm.generate(selection_prompt)
            print("selection_result", selection_result)
            if not selection_result:
                return "Failed to evaluate search results."
            
            # Extract just the answer from the selection result
            if isinstance(selection_result, dict) and "message" in selection_result:
                message = selection_result["message"]
                # Find the ANSWER section
                if "ANSWER:" in message:
                    answer = message.split("ANSWER:")[1].strip()
                    return answer
                return message  # Return full message if no ANSWER section found
            
            return "No relevant information found."

        except Exception as e:
            logger.error(f"Error in parallel message handling: {str(e)}")
            self.transition_to(AgentState.ERROR)
            return f"Error in message handling: {str(e)}"

    def _process_result(self, result: Any, source: str) -> Dict:
        """Process and structure results from different tools"""
        if isinstance(result, Exception):
            return {"status": "error", "source": source, "data": str(result)}
            
        if not result:
            return {"status": "empty", "source": source, "data": None}

        # Convert result to structured format
        if isinstance(result, dict):
            data = result.get("message") or result
        elif isinstance(result, list):
            data = result
        else:
            data = str(result)

        return {
            "status": "success",
            "source": source,
            "data": data
        }

    def _create_selection_prompt(self, original_query: str, results: Dict) -> str:
        """Create a prompt for LLM to evaluate and select the most relevant information"""
        prompt = f"""
Given the following search results for the query: "{original_query}"

Vector Search Results:
{self._format_results(results['vector_search'])}

API Search Results:
{self._format_results(results['api_search'])}

Please:
1. Evaluate the relevance of each source to the original query
2. Select the most relevant information
3. If both sources contain complementary information, combine them
4. If the information conflicts, explain which source seems more reliable and why
5. Provide a final answer based on the most relevant and reliable information

Format your response as:
EVALUATION: [Brief evaluation of sources]
SELECTED SOURCE(S): [Which source(s) you're using]
ANSWER: [Your synthesized answer]
"""
        return prompt

    def _format_results(self, result: Dict) -> str:
        """Format results for the prompt"""
        if result["status"] == "error":
            return f"[Error occurred: {result['data']}]"
        elif result["status"] == "empty":
            return "[No results found]"
        else:
            return f"[Found: {str(result['data'])}]"

    async def _async_tool_execution(self, executor: ThreadPoolExecutor, tool_class, message: str) -> Optional[str]:
        """Execute tool asynchronously"""
        try:
            logger.info(f"Starting async execution for tool: {tool_class.__name__}")
            params = self._extract_tool_params(message, tool_class)
            tool = tool_class(**params)
            
            # Execute tool in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            logger.debug(f"Executing {tool_class.__name__} with params: {params}")
            result = await loop.run_in_executor(
                executor,
                lambda: getattr(self, tool_class.__name__.lower())(tool)
            )
            logger.info(f"Tool {tool_class.__name__} execution completed with result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in async tool execution: {e}")
            return None

    # Add method to support both sync and async usage
    def handle_message(self, message: str) -> str:
        """Synchronous wrapper for handle_message_parallel"""
        return asyncio.run(self.handle_message_parallel(message))
        