# services/optimized_mongodb_rag_service.py
"""
Optimized MongoDB-based RAG Service for global chat.
Replaces PostgreSQL/pgvector with MongoDB Atlas Vector Search.
"""
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging
from reports.mongodb_models import MovieSectionMongoDB
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class _ModelSingleton:
    """
    Singleton to load models once and reuse them.
    Shared between MongoDBRAGService and OptimizedMongoDBRAGService.
    """
    _instance = None
    _embedder = None
    _reranker = None
    _use_reranking = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_ModelSingleton, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        """Load models once"""
        if self._embedder is None:
            logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
            self._embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")

        # CrossEncoder disabled for performance (saves 1-2s per query)
        # Only provides ~20-25% relevance improvement on CPU
        # Re-enable by setting USE_RERANKER = True if needed
        USE_RERANKER = False

        if USE_RERANKER and self._reranker is None:
            logger.info("Loading cross-encoder for re-ranking...")
            try:
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self._use_reranking = True
                logger.info("Cross-encoder loaded - expect ~25% improvement in relevance")
            except Exception as e:
                logger.warning(f"Cross-encoder not available: {e}")
                self._use_reranking = False
        else:
            logger.info("Cross-encoder disabled for performance (saves 1-2s per query)")

    @property
    def embedder(self):
        return self._embedder

    @property
    def reranker(self):
        return self._reranker

    @property
    def use_reranking(self):
        return self._use_reranking


class OptimizedMongoDBRAGService:
    """
    Optimized RAG service for global chat using MongoDB Atlas Vector Search.

    Features:
    - MongoDB Atlas Vector Search (replaces PostgreSQL/pgvector)
    - Model singleton pattern
    - Embedding cache (LRU, last 100 queries)
    - Query type classification
    - Section priority weights
    - Conversation context enhancement
    - Diversity enforcement (max 2 sections per movie)
    """

    def __init__(self):
        # Get singleton instance with pre-loaded models
        self._models = _ModelSingleton()
        self.embedder = self._models.embedder
        self.reranker = self._models.reranker
        self.use_reranking = self._models.use_reranking
        self.embedding_dim = 384
        # Simple LRU cache for embeddings (last 100 queries)
        self._embedding_cache = {}

    def generate_embedding(self, text):
        """
        Generate normalized embeddings with caching.
        """
        try:
            # Check cache first (for identical queries)
            cache_key = hash(text[:500])  # Hash first 500 chars
            if cache_key in self._embedding_cache:
                logger.debug("Using cached embedding")
                return self._embedding_cache[cache_key]

            # Generate new embedding
            embedding = self.embedder.encode(
                text,
                normalize_embeddings=True,  # Normalization for better cosine distance
                show_progress_bar=False,
                convert_to_numpy=True
            )
            result = embedding.astype('float32')

            # Cache it (simple LRU: keep last 100)
            if len(self._embedding_cache) > 100:
                # Remove oldest (first) item
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[cache_key] = result

            return result
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _classify_query_type(self, query):
        """Classify query into plot, technical, analysis, facts, or general."""
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

    def search_optimized(self, query: str, k: int = 10, min_similarity: float = 0.30,
                        movie_id: Optional[int] = None, use_reranking: bool = False,
                        conversation_context: Optional[str] = None):
        """
        Optimized search with:
        - Higher k (8-12 vs 5)
        - Minimum similarity threshold (0.25-0.35)
        - Cross-encoder re-ranking (DISABLED by default for speed)
        - Conversation context
        - Query type classification
        - Section priority weights
        - Diversity enforcement

        Args:
            query: User query
            k: Number of results (8-12 recommended)
            min_similarity: Minimum cosine similarity (0.25-0.35)
            movie_id: Optional movie filter
            use_reranking: Use cross-encoder re-ranking (slow, disabled by default)
            conversation_context: Last N messages for context

        Returns:
            List of search results with scores
        """
        # Include conversation context in embedding
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"{conversation_context}\n\nCurrent question: {query}"

        query_embedding = self.generate_embedding(enhanced_query)
        query_type = self._classify_query_type(query)

        # Retrieve with higher k for re-ranking
        initial_k = k * 2 if use_reranking and self.use_reranking else k

        # Perform MongoDB vector search
        mongo_results = MovieSectionMongoDB.vector_search(
            query_embedding=query_embedding,
            k=initial_k * 2,  # Over-retrieve for reranking and filtering
            movie_id=movie_id,
            min_similarity=min_similarity
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

        # Optional: Cross-encoder re-ranking (DISABLED by default)
        if use_reranking and self.use_reranking:
            logger.info("Using cross-encoder re-ranking")
            pairs = [[query, r['content']] for r in mongo_results]
            rerank_scores = self.reranker.predict(pairs)

            for i, result in enumerate(mongo_results):
                # Combine vector similarity with cross-encoder score
                result['weighted_score'] = (
                    0.6 * result['weighted_score'] +
                    0.4 * rerank_scores[i]
                )

        # Sort by weighted score
        reranked = sorted(mongo_results, key=lambda x: x['weighted_score'], reverse=True)

        # Diversity enforcement: max 2 sections per movie
        diverse_results = []
        movie_counts = {}

        for result in reranked:
            mid = result['movie_id']
            if movie_counts.get(mid, 0) < 2:
                diverse_results.append(result)
                movie_counts[mid] = movie_counts.get(mid, 0) + 1

            if len(diverse_results) >= k:
                break

        logger.info(
            f"Query type: {query_type}, Retrieved {len(diverse_results)} sections from MongoDB, "
            f"Similarity range: {diverse_results[0]['similarity']:.3f} - {diverse_results[-1]['similarity']:.3f}"
        )

        return diverse_results

    def search_with_conversation_context(self, query: str, conversation_history: List[Dict],
                                         k: int = 10, movie_id: Optional[int] = None):
        """
        Search with conversation context enhancement.

        Args:
            query: Current user query
            conversation_history: List of previous messages
            k: Number of results
            movie_id: Optional movie filter

        Returns:
            List of search results
        """
        # Extract conversation context from last N messages
        context_manager = ConversationMemoryManager()
        context_text = context_manager.get_context_text(conversation_history, last_n=5)

        return self.search_optimized(
            query=query,
            k=k,
            movie_id=movie_id,
            conversation_context=context_text
        )

    def search_with_scores(self, query: str, k: int = 10, min_similarity: float = 0.30,
                          movie_id: Optional[int] = None, conversation_context: Optional[str] = None):
        """
        Wrapper method for compatibility with GlobalChatService.
        Returns results in format compatible with old PostgreSQL RAG service.
        Fetches content from PostgreSQL (MongoDB only has embeddings).

        Returns dict with keys: section_id, similarity, movie_title, section_type, content, movie_id
        PLUS a 'section' key with a simple object for backward compatibility.
        """
        results = self.search_optimized(
            query=query,
            k=k,
            min_similarity=min_similarity,
            movie_id=movie_id,
            conversation_context=conversation_context
        )

        # Get content from PostgreSQL (MongoDB only has vectors!)
        from movies.models import Movie
        from reports.models import MovieSection

        movie_ids = list(set([r['movie_id'] for r in results]))
        movies = {m.id: m for m in Movie.objects.filter(id__in=movie_ids)}

        # Fetch sections from PostgreSQL by (movie_id, section_type)
        sections_dict = {}
        for r in results:
            key = (r['movie_id'], r['section_type'])
            if key not in sections_dict:
                try:
                    section = MovieSection.objects.get(
                        movie_id=r['movie_id'],
                        section_type=r['section_type']
                    )
                    sections_dict[key] = section
                except MovieSection.DoesNotExist:
                    sections_dict[key] = None

        # Create simple section-like objects for backward compatibility
        formatted_results = []
        for r in results:
            movie = movies.get(r['movie_id'])
            if not movie:
                continue

            # Get actual section from PostgreSQL
            section_key = (r['movie_id'], r['section_type'])
            pg_section = sections_dict.get(section_key)

            if not pg_section:
                continue  # Skip if section doesn't exist in PostgreSQL

            # Create a simple object that mimics MovieSection
            class SimpleSection:
                def __init__(self, mongo_data, pg_section_obj, movie_obj):
                    self.id = str(mongo_data['_id'])
                    self.movie = movie_obj
                    self.section_type = mongo_data['section_type']
                    self.content = pg_section_obj.content
                    self.word_count = pg_section_obj.word_count

                def get_section_type_display(self):
                    return self.section_type.replace('_', ' ').title()

            formatted_results.append({
                'section': SimpleSection(r, pg_section, movie),
                'section_id': str(r['_id']),
                'similarity': r['similarity'],
                'weighted_score': r.get('weighted_score', r['similarity']),
                'movie_title': movie.title,
                'movie_id': r['movie_id'],
                'section_type': r['section_type'],
                'content': pg_section.content
            })

        return formatted_results


class ConversationMemoryManager:
    """
    Manages conversation context for enhanced RAG queries.
    """

    def __init__(self, max_context_length=500):
        self.max_context_length = max_context_length

    def get_context_text(self, conversation_history: List[Dict], last_n: int = 5) -> str:
        """
        Extract context from last N messages.

        Args:
            conversation_history: List of {role, content} dicts
            last_n: Number of recent messages to include

        Returns:
            Context text string
        """
        if not conversation_history:
            return ""

        # Get last N messages
        recent_messages = conversation_history[-last_n:]

        # Build context string
        context_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_parts.append(f"{role.capitalize()}: {content}")

        context = "\n".join(context_parts)

        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]

        return context

    def extract_referenced_movies(self, conversation_history: List[Dict]) -> List[str]:
        """
        Extract movie titles mentioned in conversation.

        Args:
            conversation_history: List of messages

        Returns:
            List of movie titles
        """
        # Simple extraction - look for quoted titles or common patterns
        # Could be enhanced with NER or regex patterns
        movie_titles = []

        for msg in conversation_history:
            content = msg.get('content', '')

            # Look for quoted titles
            import re
            quoted = re.findall(r'"([^"]+)"', content)
            movie_titles.extend(quoted)

            # Look for "the movie X" patterns
            movie_patterns = re.findall(r'the movie ([A-Z][a-zA-Z\s]+)', content)
            movie_titles.extend(movie_patterns)

        return list(set(movie_titles))  # Remove duplicates
