from shared.infrastructure.agent.base import Agent, AgentConfig
from shared.infrastructure.tools.slack_tool import SlackTool
from shared.infrastructure.tools.mongodb_tool import MongoDBTool
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
from os import getenv

logger = logging.getLogger(__name__)

class NotificationAgentConfig(AgentConfig):
    """Configuration for NotificationAgent"""
    slack_webhook_url: Optional[str] = getenv('SLACK_WEBHOOK_URL')
    slack_channel: str = getenv('SLACK_CHANNEL', '#feedback-alerts')
    mongodb_uri: str = getenv('MONGODB_URI', 'mongodb://localhost:27017')
    mongodb_db: str = getenv('MONGODB_DB', 'querylytics')
    mongodb_collection: str = getenv('MONGODB_COLLECTION', 'feedback')

class NotificationAgent(Agent):
    def __init__(self, config: NotificationAgentConfig):
        super().__init__(config)
        
        # Initialize tools
        self.slack_tool = SlackTool(
            webhook_url=config.slack_webhook_url,
            default_channel=config.slack_channel
        )
        
        self.mongodb_tool = MongoDBTool(
            uri=config.mongodb_uri,
            database=config.mongodb_db,
            collection=config.mongodb_collection
        )

    def handle_message(self, message: dict) -> str:
        """
        Handle the notification process:
        1. Store feedback in MongoDB
        2. Send Slack notification for important feedback
        """
        try:
            # Extract data from message
            feedback_type = message.get("feedback_type")
            original_query = message.get("original_query")
            probe_summary = message.get("probe_summary", {})
            
            # Prepare document for MongoDB
            feedback_document = {
                "timestamp": datetime.utcnow(),
                "feedback_type": feedback_type,
                "original_query": original_query,
                "findings": probe_summary.get("findings"),
                "responses": probe_summary.get("responses", []),
                "status": "new"
            }
            
            # Store in MongoDB
            try:
                result = self.mongodb_tool.insert_document(feedback_document)
                logger.info(f"Stored feedback in MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                logger.error(f"Failed to store feedback in MongoDB: {str(e)}")
            
            # Prepare and send Slack notification
            slack_message = self._format_slack_message(feedback_document)
            try:
                self.slack_tool.send_message(
                    message=slack_message,
                    channel=self.config.slack_channel
                )
                logger.info("Sent feedback notification to Slack")
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {str(e)}")
            
            return "Notification processed successfully"
            
        except Exception as e:
            logger.error(f"Error processing notification: {str(e)}")
            return f"Error: {str(e)}"

    def _format_slack_message(self, feedback: dict) -> str:
        """Format feedback data for Slack notification"""
        # Add default values and handle None cases with explicit string conversion
        findings = str(feedback.get("findings", "No findings available"))
        responses = feedback.get("responses", [])
        query = str(feedback.get("original_query", "N/A"))
        feedback_type = str(feedback.get("feedback_type", "N/A"))
        
        # Create a formatted message for Slack
        message = [
            "*New Feedback Alert* 🔔",
            f"*Query:* {query}",
            f"*Type:* {feedback_type}",
            "",
            "*Summary of Findings:*",
            findings,  # Already converted to string above
            "",
            "*Detailed Responses:*"
        ]
        
        # Add responses if available
        if responses:
            try:
                for response_type, response in responses:
                    message.append(f"• *{str(response_type)}:* {str(response)}")
            except (TypeError, ValueError):
                message.append("Invalid response format")
        else:
            message.append("No detailed responses available")
        
        return "\n".join(message) 