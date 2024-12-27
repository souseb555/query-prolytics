from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig, AgentState
import logging
from typing import Union

logger = logging.getLogger(__name__)

class ProbingAgentConfig(AgentConfig):
    """Configuration for ProbingAgent"""
    max_questions: int = 5
    system_message: str = """
    You are a probing agent that asks clarifying questions to understand why a user was not satisfied with a previous answer.
    Ask focused, relevant questions one at a time.
    Be empathetic and constructive in your questioning.
    """

class ProbingAgent(Agent):
    def __init__(self, config: ProbingAgentConfig):
        super().__init__(config)
        # Initialize with a list to store Q&A pairs
        self.collected_responses = []
        self.question_counter = 0
        self.current_question = None
        
    def _store_response(self, question: str, answer: str):
        """Store question and answer as a dictionary in the list"""
        self.question_counter += 1
        qa_pair = {
            "question_id": f"question_{self.question_counter}",
            "question": question,
            "answer_id": f"answer_{self.question_counter}",
            "answer": answer
        }
        self.collected_responses.append(qa_pair)

    def handle_message(self, message: str) -> Union[str, dict]:
        """Handle incoming messages and manage the probing conversation"""
        logger.info(f"Handling message in probing. Question count: {self.question_counter}")
        
        if self.current_question:
            self._store_response(self.current_question, message)
        
        # Check if we should finish probing
        if self.question_counter >= self.config.max_questions:
            return self._create_summary()
            
        # Generate and store next question
        self.current_question = self._generate_next_question()
        return self.current_question

    def _generate_next_question(self) -> str:
        """Generate the next contextual question based on previous responses"""
        # Format the conversation history
        conversation_history = ""
        for qa in self.collected_responses:
            conversation_history += f"Q: {qa['question']}\nA: {qa['answer']}\n"
        
        prompt = f"""
        Previous responses:
        {conversation_history}

        Question {self.question_counter + 1} of {self.config.max_questions}:
        Generate a focused question to better understand the user's concerns.
        Respond with just the question, no additional text.
        """
        
        response = self.llm.generate(prompt)
        return response.get('message', 'What specific improvements would you like to see?')

    def _create_summary(self) -> dict:
        """Create a summary of the probing session"""
        # Format the conversation history
        conversation_history = ""
        for qa in self.collected_responses:
            conversation_history += f"Q: {qa['question']}\nA: {qa['answer']}\n"
        
        prompt = f"""
        Based on this conversation:
        {conversation_history}
        
        Summarize:
        1. Main issues identified
        2. Specific improvements needed
        3. Key action items
        
        Format as JSON with these keys: issues, improvements, actions
        Return only the JSON object, no markdown formatting.
        """
        
        try:
            response = self.llm.generate(prompt)
            logger.info("Generated probing agent summary")
            
            # Get the message content
            message = response.get('message', '{}')
            
            # If it's already a dict, use it directly
            if isinstance(message, dict):
                findings = message
            else:
                # Try to parse if it's a string
                try:
                    import json
                    # Clean up markdown if present
                    if '```' in message:
                        message = message.split('```json\n')[1].split('\n```')[0]
                    findings = json.loads(message)
                except (json.JSONDecodeError, IndexError):
                    logger.warning("Failed to parse response")
                    findings = {
                        "issues": ["Unable to parse response"],
                        "improvements": [],
                        "actions": []
                    }
            
            return {
                "status": "complete",
                "findings": findings,
                "responses": self.collected_responses
            }
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return {
                "status": "error",
                "findings": "Error generating summary",
                "responses": self.collected_responses
            } 