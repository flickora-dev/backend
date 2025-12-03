# services/mongodb_rag_service.py
"""
MongoDB-based RAG Service for vector similarity search.
Replaces PostgreSQL/pgvector with MongoDB Atlas Vector Search.
"""
from sentence_transformers import SentenceTransformer
from django.conf import settings
import logging
import numpy as np
from reports.mongodb_models import MovieSectionMongoDB
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MongoDBRAGService:
    """
    RAG Service using MongoDB Atlas Vector Search instead of PostgreSQL pgvector.
    """

    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = 384
        # Use shared singleton from OptimizedMongoDBRAGService to avoid loading model twice
        from services.optimized_mongodb_rag_service import _ModelSingleton
        self._models = _ModelSingleton()

    def load_model(self):
        """Return pre-loaded model from singleton"""
        return self._models.embedder

    def generate_embedding(self, text):
        try:
            model = self.load_model()
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False
            )
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _classify_query_type(self, query):
        query_lower = query.lower()

        plot_keywords = ['what happens', 'story', 'plot', 'ending', 'scene', 'character does',
                        'beginning', 'middle', 'climax', 'synopsis', 'summary', 'occurs']
        if any(kw in query_lower for kw in plot_keywords):
            return 'plot'

        technical_keywords = ['cinematography', 'camera', 'visual', 'shot', 'editing',
                             'sound', 'music', 'score', 'design', 'costume', 'lighting',
                             'effects', 'cinematographer']
        if any(kw in query_lower for kw in technical_keywords):
            return 'technical'

        analysis_keywords = ['theme', 'meaning', 'symbol', 'represents', 'analysis',
                            'message', 'philosophical', 'deeper', 'metaphor']
        if any(kw in query_lower for kw in analysis_keywords):
            return 'analysis'

        facts_keywords = ['budget', 'box office', 'award', 'actor', 'director', 'cast',
                         'when', 'where', 'who', 'made', 'produced', 'crew']
        if any(kw in query_lower for kw in facts_keywords):
            return 'facts'

        return 'general'

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

    def search_with_priority(self, query: str, k: int = 5, movie_id: Optional[int] = None):
        """
        Search with query type classification and section priority weights.

        Args:
            query: Search query text
            k: Number of results to return
            movie_id: Optional movie ID to filter results

        Returns:
            List of search results with weighted scores
        """
        query_embedding = self.generate_embedding(query)
        query_type = self._classify_query_type(query)

        # Perform MongoDB vector search
        mongo_results = MovieSectionMongoDB.vector_search(
            query_embedding=query_embedding,
            k=k * 3,  # Over-retrieve for reranking
            movie_id=movie_id,
            min_similarity=0.30
        )

        # Section type weights based on query type
        section_weights = {
            'plot': {
                'plot_structure': 3.5,
                'characters': 2.0,
                'themes': 1.5,
                'production': 1.0,
                'cast_crew': 0.8,
                'visual_technical': 0.5,
                'reception': 0.5,
                'legacy': 0.5,
            },
            'technical': {
                'visual_technical': 3.5,
                'production': 2.0,
                'cast_crew': 1.5,
                'themes': 1.0,
                'plot_structure': 0.8,
                'characters': 0.5,
                'reception': 0.5,
                'legacy': 0.5,
            },
            'analysis': {
                'themes': 3.5,
                'characters': 2.5,
                'visual_technical': 2.0,
                'plot_structure': 1.5,
                'cast_crew': 1.0,
                'production': 0.8,
                'reception': 0.8,
                'legacy': 1.0,
            },
            'facts': {
                'production': 3.5,
                'cast_crew': 2.5,
                'reception': 2.0,
                'legacy': 1.5,
                'plot_structure': 1.0,
                'characters': 0.8,
                'visual_technical': 0.8,
                'themes': 0.5,
            },
            'general': {
                'plot_structure': 2.2,
                'themes': 1.8,
                'characters': 1.6,
                'visual_technical': 1.4,
                'production': 1.2,
                'cast_crew': 1.2,
                'reception': 1.0,
                'legacy': 1.0,
            }
        }

        weights = section_weights.get(query_type, section_weights['general'])

        # Apply weights to results
        for result in mongo_results:
            weight = weights.get(result['section_type'], 1.0)
            result['weighted_score'] = result['similarity'] * weight

        # Sort by weighted score and take top k
        reranked = sorted(mongo_results, key=lambda x: x['weighted_score'], reverse=True)[:k]

        logger.info(f"Query type: {query_type}, Retrieved {len(reranked)} sections from MongoDB")

        return reranked

    def search(self, query: str, k: int = 5, movie_id: Optional[int] = None):
        """Standard search interface."""
        return self.search_with_priority(query, k, movie_id)

    def search_with_scores(self, query: str, k: int = 5, movie_id: Optional[int] = None):
        """
        Search and return results with detailed scores.
        Fetches content from PostgreSQL (MongoDB only has embeddings).

        Returns:
            List of dicts with section info, similarity, and weighted scores
        """
        results = self.search_with_priority(query, k, movie_id)

        # Get content from PostgreSQL (MongoDB only has vectors!)
        from movies.models import Movie
        from reports.models import MovieSection

        movie_ids = [r['movie_id'] for r in results]
        movies = {m.id: m for m in Movie.objects.filter(id__in=movie_ids)}

        # Fetch sections from PostgreSQL by (movie_id, section_type)
        sections = {}
        for result in results:
            key = (result['movie_id'], result['section_type'])
            if key not in sections:
                try:
                    section = MovieSection.objects.get(
                        movie_id=result['movie_id'],
                        section_type=result['section_type']
                    )
                    sections[key] = section
                except MovieSection.DoesNotExist:
                    sections[key] = None

        return [
            {
                'section_id': str(result['_id']),
                'similarity': result['similarity'],
                'weighted_score': result['weighted_score'],
                'movie_id': result['movie_id'],
                'movie_title': movies.get(result['movie_id']).title if result['movie_id'] in movies else 'Unknown',
                'section_type': result['section_type'],
                'content': sections.get((result['movie_id'], result['section_type'])).content if sections.get((result['movie_id'], result['section_type'])) else '',
                'word_count': sections.get((result['movie_id'], result['section_type'])).word_count if sections.get((result['movie_id'], result['section_type'])) else 0,
            }
            for result in results
        ]

    def search_for_recommendations(self, query: str, k: int = 10, filters: Optional[Dict] = None):
        """
        Search for movie recommendations with filters.

        Args:
            query: Search query text
            k: Number of results
            filters: Optional filters (genres, year_from, year_to)
        """
        query_embedding = self.generate_embedding(query)

        # Perform MongoDB vector search (no filtering yet, do it in Python)
        mongo_results = MovieSectionMongoDB.vector_search(
            query_embedding=query_embedding,
            k=k * 3,
            min_similarity=0.30
        )

        # Apply filters if provided
        if filters:
            from movies.models import Movie

            # Get movie IDs that match filters
            movie_queryset = Movie.objects.all()

            if 'genres' in filters:
                movie_queryset = movie_queryset.filter(genres__name__in=filters['genres'])
            if 'year_from' in filters:
                movie_queryset = movie_queryset.filter(year__gte=filters['year_from'])
            if 'year_to' in filters:
                movie_queryset = movie_queryset.filter(year__lte=filters['year_to'])

            valid_movie_ids = set(movie_queryset.values_list('id', flat=True))
            mongo_results = [r for r in mongo_results if r['movie_id'] in valid_movie_ids]

        # Prioritize themes and characters sections for recommendations
        for result in mongo_results:
            if result['section_type'] in ['themes', 'characters']:
                result['weighted_score'] = result['similarity'] * 1.5
            else:
                result['weighted_score'] = result['similarity']

        reranked = sorted(mongo_results, key=lambda x: x['weighted_score'], reverse=True)

        # Ensure diversity - max 2 sections per movie
        diverse_results = []
        movie_counts = {}

        for result in reranked:
            movie_id = result['movie_id']
            if movie_counts.get(movie_id, 0) < 2:
                diverse_results.append(result)
                movie_counts[movie_id] = movie_counts.get(movie_id, 0) + 1

            if len(diverse_results) >= k:
                break

        # Get movie titles
        from movies.models import Movie
        movie_ids = [r['movie_id'] for r in diverse_results]
        movies = {m.id: m for m in Movie.objects.filter(id__in=movie_ids)}

        return [
            {
                'section_id': str(result['_id']),
                'similarity': result['similarity'],
                'weighted_score': result['weighted_score'],
                'movie_id': result['movie_id'],
                'movie_title': movies.get(result['movie_id']).title if result['movie_id'] in movies else 'Unknown',
                'section_type': result['section_type'],
                'content': result['content'],
            }
            for result in diverse_results
        ]

    def search_for_comparison(self, query: str, movie_titles: List[str], k: int = 8):
        """
        Search for comparison between specific movies.

        Args:
            query: Search query text
            movie_titles: List of movie titles to compare
            k: Number of results
        """
        from movies.models import Movie

        query_embedding = self.generate_embedding(query)

        # Find movies by titles
        movies = Movie.objects.filter(title__in=movie_titles)
        movie_ids = [m.id for m in movies]

        if not movie_ids:
            # If no movies found, use standard search
            return self.search_with_scores(query, k=k, movie_id=None)

        # Search all movies first, then filter
        mongo_results = MovieSectionMongoDB.vector_search(
            query_embedding=query_embedding,
            k=k * 2,
            min_similarity=0.30
        )

        # Filter by movie IDs
        mongo_results = [r for r in mongo_results if r['movie_id'] in movie_ids]

        # Prioritize themes, characters, visual_technical
        for result in mongo_results:
            if result['section_type'] in ['themes', 'characters', 'visual_technical']:
                result['weighted_score'] = result['similarity'] * 1.8
            else:
                result['weighted_score'] = result['similarity']

        reranked = sorted(mongo_results, key=lambda x: x['weighted_score'], reverse=True)[:k]

        # Get movie objects for titles
        movies_dict = {m.id: m for m in movies}

        return [
            {
                'section_id': str(result['_id']),
                'similarity': result['similarity'],
                'weighted_score': result['weighted_score'],
                'movie_id': result['movie_id'],
                'movie_title': movies_dict.get(result['movie_id']).title if result['movie_id'] in movies_dict else 'Unknown',
                'section_type': result['section_type'],
                'content': result['content'],
            }
            for result in reranked
        ]

    def search_by_genre_or_theme(self, query: str, k: int = 10):
        """
        Search by genres and themes.

        Args:
            query: Search query text
            k: Number of results
        """
        query_embedding = self.generate_embedding(query)

        # Perform MongoDB vector search
        mongo_results = MovieSectionMongoDB.vector_search(
            query_embedding=query_embedding,
            k=k * 2,
            min_similarity=0.30
        )

        # Filter by section types - prioritize themes and legacy
        mongo_results = [
            r for r in mongo_results
            if r['section_type'] in ['themes', 'legacy', 'characters', 'plot_structure']
        ]

        # Ensure diversity - one section per movie
        diverse_results = []
        seen_movies = set()

        for result in mongo_results:
            if result['movie_id'] not in seen_movies:
                diverse_results.append(result)
                seen_movies.add(result['movie_id'])

            if len(diverse_results) >= k:
                break

        # Get movie titles
        from movies.models import Movie
        movie_ids = [r['movie_id'] for r in diverse_results]
        movies = {m.id: m for m in Movie.objects.filter(id__in=movie_ids)}

        return [
            {
                'section_id': str(result['_id']),
                'similarity': result['similarity'],
                'weighted_score': result['similarity'],
                'movie_id': result['movie_id'],
                'movie_title': movies.get(result['movie_id']).title if result['movie_id'] in movies else 'Unknown',
                'section_type': result['section_type'],
                'content': result['content'],
            }
            for result in diverse_results
        ]
