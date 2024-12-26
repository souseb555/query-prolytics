import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SlackTool:
    def __init__(self, webhook_url: str, default_channel: str = "#general"):
        self.webhook_url = webhook_url
        self.default_channel = default_channel

    def send_message(self, message: str, channel: Optional[str] = None) -> bool:
        """
        Send a message to Slack channel
        Returns True if successful, False otherwise
        """
        try:
            payload = {
                "channel": channel or self.default_channel,
                "text": message,
                "mrkdwn": True  # Enable markdown-style formatting
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {str(e)}")
            return False 