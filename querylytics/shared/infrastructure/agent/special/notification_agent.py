from querylytics.shared.infrastructure.agent.base import Agent, AgentConfig
from querylytics.shared.infrastructure.tools.slack_tool import SlackTool
from querylytics.shared.infrastructure.tools.mongodb_tool import MongoDBTool
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime, timezone
from os import getenv

logger = logging.getLogger(__name__)

class NotificationAgentConfig(AgentConfig):
    """Configuration for NotificationAgent"""
    slack_webhook_url: Optional[str] = getenv('SLACK_WEBHOOK_URL')
    slack_channel: str = getenv('SLACK_CHANNEL', '#feedback-alerts')
    mongodb_username: Optional[str] = getenv('MONGODB_USERNAME')
    mongodb_password: Optional[str] = getenv('MONGODB_PASSWORD')
    mongodb_cluster_url: str = getenv('MONGODB_CLUSTER_URL', 'localhost:27017')
    print("mongodb_cluster_url", mongodb_cluster_url)
    mongodb_database: str = getenv('MONGODB_DATABASE', 'querylytics')
    mongodb_collection: str = getenv('MONGODB_COLLECTION', 'feedback')

class NotificationAgent(Agent):
    def __init__(self, config: NotificationAgentConfig):
        super().__init__(config)
        
        self.slack_tool = SlackTool(
            webhook_url=config.slack_webhook_url,
            default_channel=config.slack_channel
        )
        
        self.mongodb_tool = MongoDBTool(
            username=config.mongodb_username,
            password=config.mongodb_password,
            cluster_url=config.mongodb_cluster_url,
            database=config.mongodb_database,
            collection=config.mongodb_collection
        )

    def handle_message(self, message: dict) -> str:
        """
        Handle the notification process:
        1. Store feedback in MongoDB
        2. Send Slack notification for important feedback
        """
        logger.info("Starting handle_message with input: %s", message)
        try:
            feedback_type = message.get("feedback_type")
            original_query = message.get("original_query")
            probe_summary = message.get("probe_summary", {})
            
            # Prepare document for MongoDB
            feedback_document = {
                "timestamp": datetime.now(timezone.utc),
                "feedback_type": feedback_type,
                "original_query": original_query,
                "findings": probe_summary.get("findings"),
                "responses": probe_summary.get("responses", []),
                "status": "new"
            }
            
            logger.info("Prepared feedback document: %s", feedback_document)
            
            try:
                logger.info("Attempting to insert into MongoDB...")
                result = self.mongodb_tool.insert_document(feedback_document)
                logger.info(f"Successfully stored feedback in MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                logger.error(f"Failed to store feedback in MongoDB: {str(e)}", exc_info=True)
                raise
            
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
            findings,
            "",
            "*Detailed Responses:*"
        ]
        
        if responses:
            try:
                for response_type, response in responses:
                    message.append(f"• *{str(response_type)}:* {str(response)}")
            except (TypeError, ValueError):
                message.append("Invalid response format")
        else:
            message.append("No detailed responses available")
        
        return "\n".join(message) 