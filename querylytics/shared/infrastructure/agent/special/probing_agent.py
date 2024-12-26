from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig, AgentState
from querylytics.shared.infrastructure.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from typing import List, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Instead of extending AgentState, we'll use the existing one
# Add our new states to the existing AgentState if needed
AgentState.QUESTIONING = "questioning"
AgentState.SUMMARIZING = "summarizing"

class ProbingAgentConfig(AgentConfig):
    """Configuration for ProbingAgent"""
    max_questions: int = 3
    system_message: str = """
    You are a probing agent that asks clarifying questions to understand why a user was not satisfied with a previous answer.
    Ask focused, relevant questions one at a time.
    Be empathetic and constructive in your questioning.
    """
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        chat_model=OpenAIChatModel.GPT4,
        temperature=0.7
    )

class ProbingAgent(Agent):
    def __init__(self, config: ProbingAgentConfig):
        super().__init__(config)
        self.collected_responses = []
        self.question_count = 0
        self.current_question = None

    def handle_message(self, message: str) -> str:
        """
        Handle the probing session with exactly 3 questions before generating summary.
        """
        try:
            # If this is a user response to a previous question, store it
            if self.current_question and self.question_count > 0:
                self.collected_responses.append((f"response_{self.question_count}", message))

            # Generate next question or summary based on question count
            if self.question_count < 3:
                self.question_count += 1
                self.current_question = self._generate_question(
                    question_number=self.question_count,
                    previous_responses=self.collected_responses
                )
                return self.current_question
            
            # Generate and return summary after collecting all responses
            summary = self._generate_summary()
            self.question_count = 0  # Reset for next session
            self.current_question = None
            return summary

        except Exception as e:
            logger.error(f"Error in probing: {str(e)}")
            return {"feedback_type": "error", "findings": str(e)}

    def _generate_question(self, question_number: int, previous_responses: List) -> str:
        """Generate contextual questions based on conversation progress."""
        context = "\n".join([f"{type}: {response}" for type, response in previous_responses])
        
        prompt = f"""
        You are conducting a probing session to understand user dissatisfaction with a previous answer.
        This is question {question_number} out of 3.
        
        Previous responses:
        {context if previous_responses else "No previous responses"}
        
        Generate a clear, focused question that:
        - For first question: Identifies the core issue with the previous answer
        - For follow-up questions: Builds on previous responses to get more specific details
        - For final question: Confirms the most important improvements needed
        
        Provide only the question text without any additional context or explanation.
        """

        try:
            response = self.llm.generate(prompt)
            print(response)
            return response.get('message')
        except Exception as e:
            logger.error(f"Failed to generate question: {str(e)}")
            return "What specific improvements would you like to see?"

    def _generate_summary(self) -> dict:
        responses_text = "\n".join([f"{type}: {response}" for type, response in self.collected_responses])
        
        prompt = f"""
        Based on these user responses:
        {responses_text}
        
        Provide a concise summary of:
        1. The main issue with the original answer
        2. Specific improvements needed
        3. Action items for improvement
        """
        
        try:
            response = self.llm.generate(prompt)
            return {
                "feedback_type": "probing_complete",
                "findings": response.get('content'),
                "responses": self.collected_responses
            }
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return {
                "feedback_type": "probing_complete",
                "findings": "Error generating summary",
                "responses": self.collected_responses
            } 