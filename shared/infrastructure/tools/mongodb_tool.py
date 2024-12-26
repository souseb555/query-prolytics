from pymongo import MongoClient
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MongoDBTool:
    def __init__(self, uri: str, database: str, collection: str):
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.collection = self.db[collection]
        logger.info(f"Initialized MongoDB connection to {database}.{collection}")

    def insert_document(self, document: Dict[str, Any]):
        """Insert a document into the collection"""
        return self.collection.insert_one(document)

    def find_documents(self, query: Dict[str, Any]):
        """Find documents matching the query"""
        return self.collection.find(query)

    def update_document(self, query: Dict[str, Any], update: Dict[str, Any]):
        """Update documents matching the query"""
        return self.collection.update_one(query, {"$set": update}) 