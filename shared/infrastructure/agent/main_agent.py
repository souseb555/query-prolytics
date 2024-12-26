from pydantic import BaseModel
from typing import Optional
import logging


from .base import Agent, AgentConfig, AgentState
from .special.retrieval_agent import RetrievalAgent, RetrievalAgentConfig
from .special.probing_agent import ProbingAgent, ProbingAgentConfig
from .special.notification_agent import NotificationAgent, NotificationAgentConfig
from .task import Task



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
                try:
                    probing_task = self._create_probe_task()
                    next_question = probing_task.run(message)
                    
                    if isinstance(next_question, dict):
                        notification_task = self._create_notification_task()
                        notification_task.run({
                            "feedback_type": self.context.feedback_type,
                            "original_query": self.context.current_query,
                            "probe_summary": next_question
                        })
                        
                        self.transition_to(AgentState.DONE)
                        self._reset()
                        return "Thank you for your feedback. I'll use it to improve future responses."
                    
                    return next_question
                    
                except Exception as e:
                    logger.error(f"Error in probing session: {str(e)}")
                    self.transition_to(AgentState.ERROR)
                    return f"Error during feedback session: {str(e)}"
                
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
        
        # First, determine query type
        query_type = self._classify_query(query)
        logger.debug(f"Query classified as: {query_type}")
        
        try:
            if query_type == "general_chat":
                # Use direct LLM response for simple queries
                answer = self._handle_general_chat(query)
                answer = answer.get("message")
                return answer  # Return directly without asking for feedback
            else:
                # Use retrieval for information-seeking queries
                self.transition_to(AgentState.RETRIEVING)
                retrieval_task = self._create_retrieval_task()
                answer = retrieval_task.run(query)
            
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
        # Simple classification prompt for the LLM
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
        
        # Define semantic feedback options
        FEEDBACK_SATISFIED = {"yes", "good", "correct", "perfect", "thanks"}
        FEEDBACK_UNSATISFIED = {"no", "wrong", "incorrect", "bad", "not helpful"}
        
        if feedback in FEEDBACK_SATISFIED:
            self.transition_to(AgentState.DONE)
            self._reset()
            return "Great! Let me know if you have any other questions."
            
        elif feedback in FEEDBACK_UNSATISFIED:
            logger.info("Starting probing session for unsatisfied feedback")
            
            # Record feedback for analytics
            self.context.feedback_type = "unsatisfied"
            
            # Start interactive probing session
            probing_task = self._create_probe_task()
            probe_args = {
                "query": self.context.current_query,
                "previous_answer": self.context.current_answer
            }
            
            # Get first question directly
            self.context.probe_session = probing_task.run(probe_args)
            first_question = self.context.probe_session
            
            logger.debug(f"First probing question: {first_question}")
            self.transition_to(AgentState.PROBING)
            
            if first_question is None:
                logger.error("Probing agent returned None for first question")
                return "I apologize, but I'm having trouble starting the feedback session. Could you try again?"
                
            return first_question
            
        else:
            return "Please indicate if you're satisfied with the answer. You can say things like 'yes', 'no', 'good', or 'not helpful'."

    def _reset(self):
        """Reset agent state and context"""
        self.transition_to(AgentState.IDLE)
        self.context.current_query = None
        self.context.current_answer = None
        self.context.probe_count = 0

    def _create_retrieval_task(self) -> Task:
        """Create a task for retrieval"""
        logger.debug("Creating retrieval task")
        task = Task(
            agent=self.retrieval_agent,
            name="retrieval_task",
            single_round=True,
            max_steps=3
        )
        # Ensure retrieval agent starts in correct state
        self.retrieval_agent.transition_to(AgentState.IDLE)
        return task

    def _create_probe_task(self) -> Task:
        """Create a probing task"""
        return Task(
            agent=self.probing_agent,  # Use the pre-initialized probing agent
            name="probing_task",
            single_round=False,
            max_steps=self.config.max_probe_attempts  # Add max steps limit
        )

    def _create_notification_task(self) -> Task:
        """Create a notification task"""
        return Task(
            agent=NotificationAgent(self.notification_config),
            name="notification_task",
            single_round=True
        ) 