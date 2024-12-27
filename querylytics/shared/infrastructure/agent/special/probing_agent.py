from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig, AgentState
import logging

logger = logging.getLogger(__name__)

class ProbingAgentConfig(AgentConfig):
    """Configuration for ProbingAgent"""
    max_questions: int = 3
    system_message: str = """
    You are a probing agent that asks clarifying questions to understand why a user was not satisfied with a previous answer.
    Ask focused, relevant questions one at a time.
    Be empathetic and constructive in your questioning.
    """

class ProbingAgent(Agent):
    def __init__(self, config: ProbingAgentConfig):
        super().__init__(config)
        self.collected_responses = []
        self.question_count = 0

    def handle_message(self, message: str) -> str | dict:
        """Handle incoming messages and manage the probing conversation"""
        logger.info(f"Handling message in probing. Question count: {self.question_count}")
        
        # Store the response
        self.collected_responses.append(message)
        
        # Check if we should finish probing
        if self.question_count >= self.config.max_questions:
            return self._create_summary()
            
        # Generate next question
        self.question_count += 1
        return self._generate_next_question()

    def _generate_next_question(self) -> str:
        """Generate the next contextual question based on previous responses"""
        conversation_history = "\n".join(self.collected_responses)
        
        prompt = f"""
        Previous responses:
        {conversation_history}

        Question {self.question_count} of {self.config.max_questions}:
        Generate a focused question to better understand the user's concerns.
        Respond with just the question, no additional text.
        """
        
        response = self.llm.generate(prompt)
        return response.get('message', 'What specific improvements would you like to see?')

    def _create_summary(self) -> dict:
        """Create a summary of the probing session"""
        conversation_history = "\n".join(self.collected_responses)
        
        prompt = f"""
        Based on this conversation:
        {conversation_history}
        
        Summarize:
        1. Main issues identified
        2. Specific improvements needed
        3. Key action items
        
        Format as JSON with these keys: issues, improvements, actions
        """
        
        try:
            response = self.llm.generate(prompt)
            return {
                "status": "complete",
                "findings": response.get('content'),
                "responses": self.collected_responses
            }
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return {
                "status": "error",
                "findings": "Error generating summary",
                "responses": self.collected_responses
            } 