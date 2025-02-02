from pydantic import BaseModel
from typing import Optional
import logging


from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig, AgentState
from querylytics.shared.infrastructure.agent.special.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from querylytics.shared.infrastructure.agent.special.probing_agent import ProbingAgent, ProbingAgentConfig
from querylytics.shared.infrastructure.agent.special.notification_agent import NotificationAgent, NotificationAgentConfig



logger = logging.getLogger(__name__)

class MainAgentConfig(AgentConfig):
    """Configuration for MainAgent"""
    retrieval_config: Optional[RetrievalAgentConfig] = None
    probing_config: Optional[ProbingAgentConfig] = ProbingAgentConfig()
    notification_config: Optional[NotificationAgentConfig] = None
    system_message: str = """
    You are a helpful assistant that can:
    1. Answer questions directly
    2. Use retrieval for detailed information
    3. Handle feedback and improve responses
    """
    max_probe_attempts: int = 3

class MainAgent(Agent):
    def __init__(self, config: MainAgentConfig):
        super().__init__(config)
        
        if not config.retrieval_config:
            logger.warning("No retrieval config provided, using default")
            self.retrieval_config = RetrievalAgentConfig(
                name="DefaultRetrieval",
                debug=True,
                llm=config.llm
            )
        else:
            self.retrieval_config = config.retrieval_config
        self.retrieval_agent = RetrievalAgent(self.retrieval_config)
        
        if not config.probing_config:
            logger.warning("No probing config provided, using default")
            self.probing_config = ProbingAgentConfig(
                name="DefaultProbing",
                debug=True,
                llm=config.llm
            )
        else:
            self.probing_config = config.probing_config
        self.probing_agent = ProbingAgent(self.probing_config)
        
        if not config.notification_config:
            logger.warning("No notification config provided, using default")
            self.notification_config = NotificationAgentConfig(
                name="DefaultNotification",
                debug=True,
                llm=config.llm
            )
        else:
            self.notification_config = config.notification_config
        
        self.context.current_query = None
        self.context.current_answer = None
        self.context.probe_count = 0
        
        logger.info(f"MainAgent initialized with config: {config}")

    def handle_message(self, message: str) -> str:
        """Main message handler"""
        try:
            logger.debug(f"Handling message in state {self.state}: {message}")
            
            if self.state == AgentState.IDLE:
                return self._handle_new_query(message)
                
            elif self.state == AgentState.WAITING_FEEDBACK:
                return self._handle_feedback(message)
                
            elif self.state == AgentState.PROBING:
                response = self.probing_agent.handle_message(message)
                
                # If we get a dict back, probing is complete
                if isinstance(response, dict):
                    self.context.probe_results = response
                    self.transition_to(AgentState.NOTIFYING)
                    return self._handle_notification()
                
                # Otherwise, continue with next question
                return response
            # elif self.state == AgentState.NOTIFYING:
            #     self.transition_to(AgentState.DONE)
            #     self._reset()
            #     return "Thank you for your feedback. I'll use it to improve future responses."

            elif self.state == AgentState.ERROR:
                self.transition_to(AgentState.IDLE)
                return "Error occurred. Please try your question again."
                
            else:
                self.transition_to(AgentState.ERROR)
                return "System error. Please try again."

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            self.transition_to(AgentState.ERROR)
            return f"Error: {str(e)}"

    def _handle_new_query(self, query: str) -> str:
        """Handle new query by determining appropriate response method"""
        logger.debug(f"Handling new query: {query}")
        self.context.current_query = query
        
        query_type = self._classify_query(query)
        logger.debug(f"Query classified as: {query_type}")
        
        try:
            if query_type == "general_chat":
                answer = self._handle_general_chat(query)
                answer = answer.get("message")
                return answer 
            else:
                self.transition_to(AgentState.RETRIEVING)
                answer = self.retrieval_agent.handle_message(query)
            
            logger.debug(f"Got answer: {answer}...")
            self.context.current_answer = answer
            self.transition_to(AgentState.WAITING_FEEDBACK)
            return f"{answer}\nAre you satisfied with this answer? (yes/no)"
            
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}", exc_info=True)
            self.transition_to(AgentState.ERROR)
            return f"Failed to get answer: {str(e)}"

    def _classify_query(self, query: str) -> str:
        """Classify the type of query to determine appropriate handling"""
        classification_prompt = f"""
        Classify the following query as either 'general_chat' or 'needs_retrieval':
        Query: {query}
        
        general_chat: General conversation, greetings, opinions, or simple questions
        needs_retrieval: Questions requiring specific information, facts, or detailed knowledge
        
        Classification:"""
        result = self.llm.generate(classification_prompt).get("message")
        return "general_chat" if "general_chat" in result else "needs_retrieval"

    def _handle_general_chat(self, query: str) -> str:
        """Handle general chat queries directly with LLM"""
        chat_prompt = f"""
        {self.config.system_message}
        
        User: {query}
        Assistant:"""
        
        return self.llm.generate(chat_prompt)

    def _handle_feedback(self, feedback: str) -> str:
        """Handle user feedback on query responses"""
        logger.info(f"Handling feedback: {feedback}")
        feedback = feedback.lower().strip()
        
        FEEDBACK_SATISFIED = {"yes", "good", "correct", "perfect", "thanks"}
        FEEDBACK_UNSATISFIED = {"no", "wrong", "incorrect", "bad", "not helpful"}
        
        if feedback in FEEDBACK_SATISFIED:
            self.transition_to(AgentState.DONE)
            self._reset()
            return "Great! Let me know if you have any other questions."
            
        elif feedback in FEEDBACK_UNSATISFIED:
            logger.info("Starting probing session for unsatisfied feedback")
            
            self.context.feedback_type = "unsatisfied"
            
            self.probing_agent = ProbingAgent(self.probing_config)
            self.transition_to(AgentState.PROBING)
            
            # Start probing with initial feedback
            return self.probing_agent.handle_message(feedback)
            
        else:
            return "Please indicate if you're satisfied with the answer. You can say things like 'yes', 'no', 'good', or 'not helpful'."

    def _reset(self):
        """Reset agent state and context"""
        self.transition_to(AgentState.IDLE)
        self.context.current_query = None
        self.context.current_answer = None
        self.context.probe_count = 0

    def _handle_notification(self) -> str:
        """Handle notification after probing is complete"""
        try:
            notification_agent = NotificationAgent(self.notification_config)
            notification_agent.handle_message({
                "feedback_type": self.context.feedback_type,
                "original_query": self.context.current_query,
                "probe_summary": self.context.probe_results
            })
            self.transition_to(AgentState.DONE)
            self._reset()
            return "Thank you for your feedback. I'll use it to improve future responses."
        except Exception as e:
            logger.error(f"Error in notification: {str(e)}")
            self.transition_to(AgentState.ERROR)
            return f"Error processing feedback: {str(e)}" 