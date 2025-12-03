# reports/mongodb_models.py
"""
MongoDB models for storing movie sections with vector embeddings.
Uses MongoDB Atlas Vector Search for similarity search.
"""
from datetime import datetime
from typing import List, Dict, Optional
from flickora.mongodb import get_collection
import numpy as np


class MovieSectionMongoDB:
    """
    MongoDB model for EMBEDDINGS ONLY.

    Content is stored in PostgreSQL MovieSection model.
    MongoDB only stores vectors for fast similarity search.

    Collection: movie_embeddings

    Schema:
    {
        "_id": ObjectId,
        "movie_id": int,  # Foreign key to PostgreSQL Movie table
        "section_type": str,  # production, plot_structure, cast_crew, etc.
        "embedding": list[float],  # 384-dimensional vector
        "created_at": datetime,
        "updated_at": datetime
    }
    """

    COLLECTION_NAME = "movie_embeddings"

    # Section type choices (must match Django model)
    SECTION_TYPES = [
        "production",
        "plot_structure",
        "cast_crew",
        "characters",
        "visual_technical",
        "themes",
        "reception",
        "legacy",
    ]

    EMBEDDING_DIMENSION = 384

    @classmethod
    def get_collection(cls):
        """Get the MongoDB collection for movie sections."""
        return get_collection(cls.COLLECTION_NAME)

    @classmethod
    def create(cls, movie_id: int, section_type: str, embedding: np.ndarray) -> str:
        """
        Create embedding document (NO CONTENT - content stays in PostgreSQL).

        Args:
            movie_id: ID of the movie (from PostgreSQL)
            section_type: Type of section (must be in SECTION_TYPES)
            embedding: 384-dimensional numpy array or list

        Returns:
            str: Inserted document ID
        """
        if section_type not in cls.SECTION_TYPES:
            raise ValueError(f"Invalid section_type. Must be one of {cls.SECTION_TYPES}")

        collection = cls.get_collection()

        # Convert embedding to list if numpy array
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        if len(embedding) != cls.EMBEDDING_DIMENSION:
            raise ValueError(f"Embedding must be {cls.EMBEDDING_DIMENSION}-dimensional")

        document = {
            "movie_id": movie_id,
            "section_type": section_type,
            "embedding": embedding,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        result = collection.insert_one(document)
        return str(result.inserted_id)

    @classmethod
    def update_embedding(cls, movie_id: int, section_type: str,
                        embedding: np.ndarray) -> bool:
        """
        Update embedding for a specific movie section.

        Args:
            movie_id: ID of the movie
            section_type: Type of section
            embedding: 384-dimensional numpy array or list

        Returns:
            bool: True if updated successfully
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        if len(embedding) != cls.EMBEDDING_DIMENSION:
            raise ValueError(f"Embedding must be {cls.EMBEDDING_DIMENSION}-dimensional")

        collection = cls.get_collection()
        result = collection.update_one(
            {"movie_id": movie_id, "section_type": section_type},
            {
                "$set": {
                    "embedding": embedding,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        return result.modified_count > 0

    @classmethod
    def get_by_movie_and_type(cls, movie_id: int, section_type: str) -> Optional[Dict]:
        """Get a specific section by movie ID and section type."""
        collection = cls.get_collection()
        return collection.find_one({"movie_id": movie_id, "section_type": section_type})

    @classmethod
    def get_all_by_movie(cls, movie_id: int) -> List[Dict]:
        """Get all sections for a specific movie."""
        collection = cls.get_collection()
        return list(collection.find({"movie_id": movie_id}))

    @classmethod
    def delete_by_movie_and_type(cls, movie_id: int, section_type: str) -> bool:
        """Delete a specific section."""
        collection = cls.get_collection()
        result = collection.delete_one({"movie_id": movie_id, "section_type": section_type})
        return result.deleted_count > 0

    @classmethod
    def vector_search(cls, query_embedding: np.ndarray, k: int = 10,
                     movie_id: Optional[int] = None,
                     min_similarity: float = 0.30) -> List[Dict]:
        """
        Perform vector similarity search using MongoDB Atlas Vector Search.

        Args:
            query_embedding: Query vector (384-dimensional)
            k: Number of results to return
            movie_id: Optional movie ID to filter results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of documents with similarity scores
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        collection = cls.get_collection()

        # Build aggregation pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",  # Name of the vector search index
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,  # Over-retrieve for better results
                    "limit": k * 2,  # Get more than needed before filtering
                }
            },
            {
                "$addFields": {
                    "similarity": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$match": {
                    "similarity": {"$gte": min_similarity}
                }
            }
        ]

        # Add movie filter if specified
        if movie_id is not None:
            pipeline.append({
                "$match": {"movie_id": movie_id}
            })

        # Limit final results
        pipeline.append({"$limit": k})

        results = list(collection.aggregate(pipeline))
        return results

    @classmethod
    def get_sections_without_embeddings(cls, limit: int = 100) -> List[Dict]:
        """Get sections that don't have embeddings yet."""
        collection = cls.get_collection()
        return list(collection.find(
            {"embedding": None},
            limit=limit
        ))

    @classmethod
    def count_sections(cls, has_embedding: Optional[bool] = None) -> int:
        """
        Count sections, optionally filtered by embedding presence.

        Args:
            has_embedding: None (all), True (with embeddings), False (without embeddings)
        """
        collection = cls.get_collection()

        if has_embedding is None:
            return collection.count_documents({})
        elif has_embedding:
            return collection.count_documents({"embedding": {"$ne": None}})
        else:
            return collection.count_documents({"embedding": None})

    @classmethod
    def create_indexes(cls):
        """
        Create necessary indexes for the collection.
        Note: Vector search index must be created via MongoDB Atlas UI.
        """
        collection = cls.get_collection()

        # Create compound index for movie_id + section_type (unique)
        collection.create_index(
            [("movie_id", 1), ("section_type", 1)],
            unique=True,
            name="idx_movie_section_unique"
        )

        # Create index on movie_id for queries
        collection.create_index("movie_id", name="idx_movie_id")

        # Create index on section_type
        collection.create_index("section_type", name="idx_section_type")

        # Create index on generated_at for sorting
        collection.create_index("generated_at", name="idx_generated_at")

        print("✅ MongoDB indexes created successfully")
        print("⚠️  IMPORTANT: You must create the vector search index manually via MongoDB Atlas UI:")
        print("   Index name: vector_index")
        print("   Path: embedding")
        print("   Dimensions: 384")
        print("   Similarity: cosine")
