# flickora/mongodb.py
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()


class MongoDBConnection:
    """
    Singleton MongoDB connection manager for Railway-hosted MongoDB.
    """
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self):
        """
        Connect to MongoDB hosted on Railway.
        Expects MONGODB_URL environment variable in format:
        mongodb://user:password@host:port/database
        or
        mongodb+srv://user:password@host/database
        """
        mongodb_url = os.getenv("MONGODB_URL")

        if not mongodb_url:
            raise ValueError(
                "MONGODB_URL environment variable is not set. "
                "Please add it to your .env file or Railway environment variables."
            )

        try:
            # Create MongoDB client with ServerApi for stability
            self._client = MongoClient(
                mongodb_url,
                server_api=ServerApi('1'),
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=45000,
                serverSelectionTimeoutMS=5000,
            )

            # Test the connection
            self._client.admin.command('ping')

            # Get database name from URL or use default
            db_name = os.getenv("MONGODB_DATABASE", "flickora")
            self._db = self._client[db_name]

            print(f"✅ Successfully connected to MongoDB database: {db_name}")

        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise

    @property
    def client(self):
        """Get MongoDB client instance."""
        if self._client is None:
            self._connect()
        return self._client

    @property
    def db(self):
        """Get MongoDB database instance."""
        if self._db is None:
            self._connect()
        return self._db

    def get_collection(self, collection_name):
        """Get a specific collection from the database."""
        return self.db[collection_name]

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("✅ MongoDB connection closed")


# Global MongoDB connection instance
mongodb_connection = MongoDBConnection()


def get_mongodb():
    """
    Get MongoDB database instance.
    Usage: db = get_mongodb()
    """
    return mongodb_connection.db


def get_collection(collection_name):
    """
    Get a specific MongoDB collection.
    Usage: collection = get_collection('movie_sections')
    """
    return mongodb_connection.get_collection(collection_name)
