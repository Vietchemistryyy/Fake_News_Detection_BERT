"""MongoDB database connection and operations"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import config

logger = logging.getLogger(__name__)

class Database:
    """MongoDB database manager"""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.users = None
        self.queries = None
        self.connected = False
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(
                config.MONGODB_URL,
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[config.MONGODB_DB_NAME]
            self.users = self.db.users
            self.queries = self.db.queries
            
            # Create indexes
            self._create_indexes()
            
            self.connected = True
            logger.info(f"✓ Connected to MongoDB: {config.MONGODB_DB_NAME}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
    
    def _create_indexes(self):
        """Create database indexes"""
        try:
            # Users collection indexes
            self.users.create_index("username", unique=True)
            self.users.create_index("email", unique=True)
            
            # Queries collection indexes
            self.queries.create_index("user_id")
            self.queries.create_index("timestamp")
            self.queries.create_index([("user_id", DESCENDING), ("timestamp", DESCENDING)])
            
            logger.info("✓ Database indexes created")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
    # ==================== User Operations ====================
    
    def create_user(self, username: str, email: str, password_hash: str) -> Optional[str]:
        """Create new user"""
        try:
            user_doc = {
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "created_at": datetime.utcnow(),
                "last_login": None
            }
            result = self.users.insert_one(user_doc)
            logger.info(f"User created: {username}")
            return str(result.inserted_id)
        except DuplicateKeyError:
            logger.warning(f"User already exists: {username}")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            user = self.users.find_one({"username": username})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = self.users.find_one({"email": email})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def update_last_login(self, username: str):
        """Update user's last login time"""
        try:
            self.users.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
    
    # ==================== Query Operations ====================
    
    def save_query(
        self,
        user_id: str,
        text: str,
        language: str,
        prediction: Dict[str, Any]
    ) -> Optional[str]:
        """Save user query"""
        try:
            query_doc = {
                "user_id": user_id,
                "text": text,
                "language": language,
                "prediction": {
                    "label": prediction.get("label"),
                    "confidence": prediction.get("confidence"),
                    "probabilities": prediction.get("probabilities")
                },
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "ai_verification": prediction.get("openai_result") is not None,
                    "mc_dropout": prediction.get("mc_dropout", False)
                }
            }
            result = self.queries.insert_one(query_doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving query: {e}")
            return None
    
    def get_user_queries(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's query history"""
        try:
            queries = list(
                self.queries.find({"user_id": user_id})
                .sort("timestamp", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for query in queries:
                query["_id"] = str(query["_id"])
                query["timestamp"] = query["timestamp"].isoformat()
            
            return queries
        except Exception as e:
            logger.error(f"Error getting queries: {e}")
            return []
    
    def get_query_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user's query statistics"""
        try:
            total = self.queries.count_documents({"user_id": user_id})
            
            # Count by language
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": "$language",
                    "count": {"$sum": 1}
                }}
            ]
            by_language = {doc["_id"]: doc["count"] for doc in self.queries.aggregate(pipeline)}
            
            # Count by prediction
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": "$prediction.label",
                    "count": {"$sum": 1}
                }}
            ]
            by_prediction = {doc["_id"]: doc["count"] for doc in self.queries.aggregate(pipeline)}
            
            return {
                "total_queries": total,
                "by_language": by_language,
                "by_prediction": by_prediction
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# Global database instance
db = Database()
