"""
MongoDB Vector Database Service
Handles connection and operations for embedding storage in MongoDB Atlas
"""
import os
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MongoDBVectorService:
    """Singleton service for MongoDB vector operations"""

    _instance = None
    _client = None
    _db = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self):
        """Initialize MongoDB connection"""
        try:
            mongodb_url = os.getenv('MONGODB_URL')
            if not mongodb_url:
                raise ValueError("MONGODB_URL not found in environment variables")

            mongodb_database = os.getenv('MONGODB_DATABASE', 'flickora')

            # Connect to MongoDB Atlas
            self._client = MongoClient(
                mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
            )

            # Test connection
            self._client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB Atlas")

            # Get database and collection
            self._db = self._client[mongodb_database]
            self._collection = self._db['movie_embeddings']

            # Create indexes
            self._setup_indexes()

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing MongoDB service: {e}")
            raise

    def _setup_indexes(self):
        """Create necessary indexes for efficient querying"""
        try:
            # Create index on section_id (unique)
            self._collection.create_index('section_id', unique=True)

            # Create index on movie_id for filtering
            self._collection.create_index('movie_id')

            # Create index on section_type for filtering
            self._collection.create_index('section_type')

            # Create compound index for movie_id + section_type
            self._collection.create_index([('movie_id', ASCENDING), ('section_type', ASCENDING)])

            # Create vector search index (Atlas Search)
            # Note: Vector search index must be created via Atlas UI or API
            # This is for the metadata indexes only

            logger.info("MongoDB indexes created successfully")

        except Exception as e:
            logger.warning(f"Error creating indexes (may already exist): {e}")

    def store_embedding(
        self,
        section_id: int,
        movie_id: int,
        section_type: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store or update an embedding in MongoDB

        Args:
            section_id: ID of the MovieSection in PostgreSQL
            movie_id: ID of the Movie
            section_type: Type of section
            embedding: 384-dimensional embedding vector
            metadata: Additional metadata (movie_title, content_preview, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            document = {
                'section_id': section_id,
                'movie_id': movie_id,
                'section_type': section_type,
                'embedding': embedding,
                'dimensions': len(embedding),
                'updated_at': datetime.utcnow(),
            }

            # Add metadata if provided
            if metadata:
                document['metadata'] = metadata

            # Upsert (insert or update)
            result = self._collection.update_one(
                {'section_id': section_id},
                {'$set': document},
                upsert=True
            )

            logger.info(f"Stored embedding for section {section_id} (movie {movie_id})")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding for section {section_id}: {e}")
            return False

    def get_embedding(self, section_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an embedding by section_id

        Args:
            section_id: ID of the MovieSection

        Returns:
            Document with embedding and metadata, or None if not found
        """
        try:
            document = self._collection.find_one({'section_id': section_id})
            return document
        except Exception as e:
            logger.error(f"Error retrieving embedding for section {section_id}: {e}")
            return None

    def delete_embedding(self, section_id: int) -> bool:
        """
        Delete an embedding by section_id

        Args:
            section_id: ID of the MovieSection

        Returns:
            True if deleted, False otherwise
        """
        try:
            result = self._collection.delete_one({'section_id': section_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted embedding for section {section_id}")
                return True
            else:
                logger.warning(f"No embedding found for section {section_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting embedding for section {section_id}: {e}")
            return False

    def delete_embeddings_by_movie(self, movie_id: int) -> int:
        """
        Delete all embeddings for a movie

        Args:
            movie_id: ID of the Movie

        Returns:
            Number of embeddings deleted
        """
        try:
            result = self._collection.delete_many({'movie_id': movie_id})
            logger.info(f"Deleted {result.deleted_count} embeddings for movie {movie_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting embeddings for movie {movie_id}: {e}")
            return 0

    def cosine_similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        movie_id: Optional[int] = None,
        section_types: Optional[List[str]] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search using aggregation pipeline

        Args:
            query_embedding: Query vector
            k: Number of results to return
            movie_id: Optional movie_id filter
            section_types: Optional section type filter
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of documents with similarity scores
        """
        try:
            # Build match filter
            match_filter = {}
            if movie_id:
                match_filter['movie_id'] = movie_id
            if section_types:
                match_filter['section_type'] = {'$in': section_types}

            logger.info(f"MongoDB search - movie_id: {movie_id}, section_types: {section_types}, match_filter: {match_filter}")

            # Convert query embedding to numpy for vectorized operations
            query_vec = np.array(query_embedding)
            query_norm = np.linalg.norm(query_vec)

            # Fetch all documents matching filters
            documents = list(self._collection.find(match_filter))
            logger.info(f"MongoDB find() returned {len(documents)} documents")

            # Calculate cosine similarity for each document
            results = []
            for doc in documents:
                embedding = np.array(doc['embedding'])
                embedding_norm = np.linalg.norm(embedding)

                # Cosine similarity: dot(A, B) / (||A|| * ||B||)
                similarity = np.dot(query_vec, embedding) / (query_norm * embedding_norm)

                # Convert cosine similarity (0-1) to distance (MongoDB uses distance)
                # Distance = 1 - similarity (so lower distance = higher similarity)
                distance = 1 - similarity

                # Filter by minimum similarity
                if similarity >= min_similarity:
                    doc['similarity'] = float(similarity)
                    doc['distance'] = float(distance)
                    results.append(doc)

            # Sort by similarity (descending) and limit to k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:k]

            logger.info(f"Cosine similarity search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error performing cosine similarity search: {e}")
            return []

    def vector_search_atlas(
        self,
        query_embedding: List[float],
        k: int = 10,
        movie_id: Optional[int] = None,
        section_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search using MongoDB Atlas Vector Search
        Note: Requires Atlas Search index to be configured

        Args:
            query_embedding: Query vector
            k: Number of results to return
            movie_id: Optional movie_id filter
            section_types: Optional section type filter

        Returns:
            List of documents with similarity scores
        """
        try:
            # Build pre-filter
            pre_filter = {}
            if movie_id:
                pre_filter['movie_id'] = movie_id
            if section_types:
                pre_filter['section_type'] = {'$in': section_types}

            # Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    '$vectorSearch': {
                        'index': 'vector_index',  # Must be created in Atlas
                        'path': 'embedding',
                        'queryVector': query_embedding,
                        'numCandidates': k * 10,  # Candidates to consider
                        'limit': k,
                    }
                },
                {
                    '$addFields': {
                        'similarity': {'$meta': 'vectorSearchScore'}
                    }
                }
            ]

            # Add pre-filter if exists
            if pre_filter:
                pipeline[0]['$vectorSearch']['filter'] = pre_filter

            results = list(self._collection.aggregate(pipeline))

            logger.info(f"Atlas vector search returned {len(results)} results")
            return results

        except OperationFailure as e:
            # Fall back to cosine similarity if Atlas Search not configured
            logger.warning(f"Atlas Vector Search not available, using cosine similarity: {e}")
            return self.cosine_similarity_search(
                query_embedding, k, movie_id, section_types
            )
        except Exception as e:
            logger.error(f"Error performing Atlas vector search: {e}")
            return []

    def get_embeddings_count(self, movie_id: Optional[int] = None) -> int:
        """
        Get count of embeddings, optionally filtered by movie_id

        Args:
            movie_id: Optional movie_id filter

        Returns:
            Count of embeddings
        """
        try:
            filter_query = {'movie_id': movie_id} if movie_id else {}
            count = self._collection.count_documents(filter_query)
            return count
        except Exception as e:
            logger.error(f"Error counting embeddings: {e}")
            return 0

    def bulk_store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert/update embeddings

        Args:
            embeddings_data: List of embedding documents

        Returns:
            Number of documents inserted/updated
        """
        try:
            if not embeddings_data:
                return 0

            from pymongo import UpdateOne

            operations = []
            for data in embeddings_data:
                operations.append(
                    UpdateOne(
                        {'section_id': data['section_id']},
                        {'$set': data},
                        upsert=True
                    )
                )

            result = self._collection.bulk_write(operations)
            count = result.upserted_count + result.modified_count
            logger.info(f"Bulk stored {count} embeddings")
            return count

        except Exception as e:
            logger.error(f"Error bulk storing embeddings: {e}")
            return 0

    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


# Singleton instance
_mongodb_service = None

def get_mongodb_service() -> MongoDBVectorService:
    """Get or create MongoDB service singleton"""
    global _mongodb_service
    if _mongodb_service is None:
        _mongodb_service = MongoDBVectorService()
    return _mongodb_service
