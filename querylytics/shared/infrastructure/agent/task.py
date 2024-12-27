from typing import List, Optional
from datetime import datetime
import logging
from querylytics.shared.infrastructure.agent.base import AgentState

logger = logging.getLogger(__name__)

class Task:
    def __init__(
        self, 
        agent, 
        name: str = "", 
        single_round: bool = False,
        max_steps: int = 10
    ):
        self.agent = agent
        self.name = name
        self.single_round = single_round
        self.max_steps = max_steps
        
        # Task state
        self.pending_message = None
        self.step_count = 0
        self.created_at = datetime.now()
        self.sub_tasks: List[Task] = []

    def add_sub_task(self, task: 'Task') -> None:
        self.sub_tasks.append(task)

    def run(self, initial_message: str) -> str:
        try:
            self.pending_message = initial_message
            print("pending_message", self.pending_message)
            response = None
            generator = None
            print("self.is_done()", self.is_done())
            print("self.step_count", self.step_count)
            print("self.max_steps", self.max_steps)
            while not self.is_done() and self.step_count < self.max_steps:
                response = self.step()
                print("response", response)
                if response is None:
                    break
                    
                self.step_count += 1
            
            return response if response else "Task completed without response"
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            self.agent.transition_to(AgentState.ERROR)
            return f"Task failed: {str(e)}"

    def step(self) -> Optional[str]:
        if self.pending_message is None:
            return None

        try:
            # Remove debug prints
            response = self.agent.handle_message(self.pending_message)
            
            # Process sub-tasks if agent not in error state
            if self.sub_tasks and self.agent.state != AgentState.ERROR:
                for sub_task in self.sub_tasks:
                    sub_response = sub_task.run(response)
                    if sub_response:
                        response = self.agent.handle_message(sub_response)

            self.pending_message = response

            # Handle single round tasks
            if self.single_round:
                self.agent.transition_to(AgentState.DONE)

            return response

        except Exception as e:
            logger.error(f"Step execution failed: {e}", exc_info=True)
            self.agent.transition_to(AgentState.ERROR)
            return None

    def is_done(self) -> bool:
        """Check if task is complete"""
        print("self.agent.state", self.agent.state)
        return self.agent.state in [AgentState.DONE, AgentState.ERROR]

    def reset(self) -> None:
        """Reset task and agent state"""
        self.pending_message = None
        self.step_count = 0
        self.agent.transition_to(self.agent.state.IDLE)
        for sub_task in self.sub_tasks:
            sub_task.reset()