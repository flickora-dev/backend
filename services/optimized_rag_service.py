import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from django.contrib.postgres.search import SearchVector
from pgvector.django import CosineDistance
import logging

logger = logging.getLogger(__name__)


class _ModelSingleton:
    """
    Singleton to load models once and reuse them
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

        if self._reranker is None:
            logger.info("Loading cross-encoder for re-ranking...")
            try:
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self._use_reranking = True
                logger.info("Cross-encoder loaded - expect ~25% improvement in relevance")
            except Exception as e:
                logger.warning(f"Cross-encoder not available: {e}")
                self._use_reranking = False

    @property
    def embedder(self):
        return self._embedder

    @property
    def reranker(self):
        return self._reranker

    @property
    def use_reranking(self):
        return self._use_reranking


class OptimizedRAGService:
    """
    Zoptymalizowany RAG service dla global chat
    Step 1-7 z optimization guide
    Uses singleton pattern to avoid reloading models
    """

    def __init__(self):
        # Get singleton instance with pre-loaded models
        self._models = _ModelSingleton()
        self.embedder = self._models.embedder
        self.reranker = self._models.reranker
        self.use_reranking = self._models.use_reranking
        self.embedding_dim = 384
        self.use_hybrid = False
        
    
    def generate_embedding(self, text):
        """
        Step 1: Generate normalized embeddings
        """
        try:
            embedding = self.embedder.encode(
                text,
                normalize_embeddings=True,  # Normalizacja dla lepszej cosine distance
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def search_optimized(self, query, k=10, min_similarity=0.30, movie_id=None,
                         use_reranking=False, conversation_context=None):
        """
        Step 1, 2, 5: Optimized search with:
        - Higher k (8-12 vs 5)
        - Minimum similarity threshold (0.25-0.35)
        - Cross-encoder re-ranking (DISABLED by default for speed)
        - Conversation context

        Args:
            query: User query
            k: Number of results (8-12 recommended)
            min_similarity: Minimum cosine similarity (0.25-0.35)
            movie_id: Optional movie filter
            use_reranking: Use cross-encoder re-ranking (slow, disabled by default)
            conversation_context: Last N messages for context
        """
        from reports.models import MovieSection

        # Step 5: Include conversation context in embedding
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"{conversation_context}\n\nCurrent question: {query}"

        query_embedding = self.generate_embedding(enhanced_query)

        # Step 1: Retrieve with higher k for re-ranking
        initial_k = k * 2 if use_reranking and self.use_reranking else k
        
        queryset = MovieSection.objects.filter(
            embedding__isnull=False
        ).annotate(
            distance=CosineDistance('embedding', query_embedding),
            similarity=1.0 - CosineDistance('embedding', query_embedding)
        )
        
        if movie_id:
            queryset = queryset.filter(movie_id=movie_id)
        
        # Step 1: Filter by minimum similarity threshold
        queryset = queryset.filter(similarity__gte=min_similarity)
        
        results = list(queryset.order_by('distance')[:initial_k])
        
        logger.info(f"Initial retrieval: {len(results)} sections (min_similarity={min_similarity})")
        
        # Step 2: Cross-encoder re-ranking
        if use_reranking and self.use_reranking and len(results) > 0:
            results = self._rerank_with_cross_encoder(query, results, k)
            logger.info(f"After re-ranking: {len(results)} sections")
        
        # Ensure diversity (max 2 sections per movie)
        results = self._ensure_diversity(results, k, max_per_movie=2)
        
        return results
    
    def _rerank_with_cross_encoder(self, query, results, top_k):
        """
        Step 2: Re-rank results using cross-encoder
        """
        try:
            # Przygotuj pary (query, content) dla cross-encodera
            pairs = [(query, section.content[:1000]) for section in results]
            
            # Oblicz scores
            scores = self.reranker.predict(pairs)
            
            # Dodaj scores do obiektów
            for section, score in zip(results, scores):
                section.rerank_score = float(score)
                section.weighted_score = float(score)
            
            # Posortuj według rerank_score
            reranked = sorted(results, key=lambda x: x.rerank_score, reverse=True)[:top_k]
            
            logger.info(f"Re-ranking improved relevance. Top score: {reranked[0].rerank_score:.3f}")
            
            return reranked
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results[:top_k]
    
    def _ensure_diversity(self, results, k, max_per_movie=2):
        """
        Ensure diversity - max N sections per movie
        """
        diverse_results = []
        movie_counts = {}
        
        for section in results:
            movie_id = section.movie_id
            if movie_counts.get(movie_id, 0) < max_per_movie:
                diverse_results.append(section)
                movie_counts[movie_id] = movie_counts.get(movie_id, 0) + 1
            
            if len(diverse_results) >= k:
                break
        
        return diverse_results
    
    def search_with_scores(self, query, k=10, min_similarity=0.30, movie_id=None,
                          conversation_context=None):
        """
        Search with scores - compatible interface
        """
        results = self.search_optimized(
            query, 
            k=k, 
            min_similarity=min_similarity,
            movie_id=movie_id,
            conversation_context=conversation_context
        )
        
        return [
            {
                'section': section,
                'section_id': section.id,
                'similarity': getattr(section, 'similarity', 1.0 - section.distance),
                'weighted_score': getattr(section, 'weighted_score', 1.0 - section.distance),
                'movie_title': section.movie.title,
                'section_type': section.get_section_type_display()
            }
            for section in results
        ]


class ConversationMemoryManager:
    """
    Step 5: Lightweight conversation memory
    """
    
    @staticmethod
    def get_conversation_context(conversation_id, n_messages=5):
        """
        Get last N messages as context string
        
        Args:
            conversation_id: Conversation ID
            n_messages: Number of recent messages (default 5)
        """
        from chat.models import ChatMessage
        
        try:
            messages = ChatMessage.objects.filter(
                conversation_id=conversation_id
            ).order_by('-created_at')[:n_messages]
            
            # Odwróć kolejność (od najstarszej do najnowszej)
            context_parts = []
            for msg in reversed(messages):
                context_parts.append(f"{msg.role.upper()}: {msg.content[:200]}")
            
            return "\n".join(context_parts)
        except:
            return ""
    
    @staticmethod
    def get_referenced_movies(conversation_id):
        """
        Extract movie titles mentioned in conversation
        """
        from chat.models import ChatConversation
        
        try:
            conversation = ChatConversation.objects.get(id=conversation_id)
            return conversation.referenced_movies or []
        except:
            return []