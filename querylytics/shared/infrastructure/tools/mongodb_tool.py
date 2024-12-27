from pymongo import MongoClient
from typing import Dict, Any
import logging
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class MongoDBTool:
    def __init__(self, username: str, password: str, cluster_url: str, database: str, collection: str):
        if not all([username, password, cluster_url]):
            raise ValueError("MongoDB connection requires username, password, and cluster_url")
        # Construct the MongoDB Atlas connection string
        escaped_username = quote_plus(username)
        escaped_password = quote_plus(password)
        connection_string = f"mongodb+srv://{escaped_username}:{escaped_password}@{cluster_url}/?retryWrites=true&w=majority"
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]
        logger.info(f"Initialized MongoDB connection to {database}.{collection}")

    def insert_document(self, document: Dict[str, Any]):
        """Insert a document into the collection"""
        logger.info("Starting insert_document with document: %s", document)
        try:
            logger.info("Attempting MongoDB insertion...")
            result = self.collection.insert_one(document)
            logger.info("MongoDB insertion successful with ID: %s", result.inserted_id)
            return result
        except Exception as e:
            logger.error("MongoDB insertion failed: %s", str(e), exc_info=True)
            raise

    def find_documents(self, query: Dict[str, Any]):
        """Find documents matching the query"""
        return self.collection.find(query)

    def update_document(self, query: Dict[str, Any], update: Dict[str, Any]):
        """Update documents matching the query"""
        return self.collection.update_one(query, {"$set": update}) 