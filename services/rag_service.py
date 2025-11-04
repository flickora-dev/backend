from sentence_transformers import SentenceTransformer
from django.conf import settings
import logging
from pgvector.django import CosineDistance
import threading
from reports.models import MovieSection

logger = logging.getLogger(__name__)

# Global model instance and lock for thread safety
_model = None
_model_lock = threading.Lock()


class RAGService:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = 384

    def load_model(self):
        global _model
        
        # Use global singleton model to avoid concurrent loading issues
        if _model is None:
            with _model_lock:
                # Double-check after acquiring lock
                if _model is None:
                    logger.info(f"Loading embedding model: {self.model_name}")
                    try:
                        # Simple initialization without device specification
                        _model = SentenceTransformer(self.model_name)
                        _model.eval()
                        logger.info("Model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading model: {e}")
                        raise
        
        return _model
    
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
    
    def search_with_priority(self, query, k=5, movie_id=None):
        from reports.models import MovieSection
        
        query_embedding = self.generate_embedding(query)
        query_type = self._classify_query_type(query)
        
        queryset = MovieSection.objects.filter(
            embedding__isnull=False
        ).annotate(
            distance=CosineDistance('embedding', query_embedding)
        )
        
        if movie_id:
            queryset = queryset.filter(movie_id=movie_id)
        
        results = list(queryset.order_by('distance')[:k*3])
        
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
        
        for section in results:
            weight = weights.get(section.section_type, 1.0)
            section.weighted_score = (1.0 - section.distance) * weight
        
        reranked = sorted(results, key=lambda x: x.weighted_score, reverse=True)[:k]
        
        logger.info(f"Query type: {query_type}, Retrieved {len(reranked)} sections")
        
        return reranked
    
    def search(self, query, k=5, movie_id=None):
        return self.search_with_priority(query, k, movie_id)
    
    def search_with_scores(self, query, k=5, movie_id=None):
        results = self.search_with_priority(query, k, movie_id)
        
        return [
            {
                'section': section,
                'section_id': section.id,
                'similarity': 1.0 - section.distance,
                'weighted_score': section.weighted_score,
                'movie_title': section.movie.title,
                'section_type': section.get_section_type_display()
            }
            for section in results
        ]
    
    def search_for_recommendations(self, query, k=10, filters=None):
        
        query_embedding = self.generate_embedding(query)
        
        queryset = MovieSection.objects.filter(
            embedding__isnull=False
        ).annotate(
            distance=CosineDistance('embedding', query_embedding)
        )
        
        # Filtruj po gatunkach, latach itp.
        if filters:
            if 'genres' in filters:
                queryset = queryset.filter(movie__genres__name__in=filters['genres'])
            if 'year_from' in filters:
                queryset = queryset.filter(movie__year__gte=filters['year_from'])
            if 'year_to' in filters:
                queryset = queryset.filter(movie__year__lte=filters['year_to'])
        
        # Pobierz więcej wyników niż potrzeba
        results = list(queryset.order_by('distance')[:k*3])
        
        # Priorytetyzuj sekcje themes i characters dla rekomendacji
        for section in results:
            if section.section_type in ['themes', 'characters']:
                section.weighted_score = (1.0 - section.distance) * 1.5
            else:
                section.weighted_score = (1.0 - section.distance)
        
        reranked = sorted(results, key=lambda x: x.weighted_score, reverse=True)
        
        # Zapewnij różnorodność - max 2 sekcje z jednego filmu
        diverse_results = []
        movie_counts = {}
        
        for section in reranked:
            movie_id = section.movie_id
            if movie_counts.get(movie_id, 0) < 2:
                diverse_results.append(section)
                movie_counts[movie_id] = movie_counts.get(movie_id, 0) + 1
            
            if len(diverse_results) >= k:
                break
        
        return diverse_results

    def search_for_comparison(self, query, movie_titles, k=8):
        """
        Wyszukiwanie dla porównań między filmami
        """
        from reports.models import MovieSection
        from movies.models import Movie
        
        query_embedding = self.generate_embedding(query)
        
        # Znajdź filmy po tytułach
        movies = Movie.objects.filter(title__in=movie_titles)
        movie_ids = [m.id for m in movies]
        
        queryset = MovieSection.objects.filter(
            embedding__isnull=False,
            movie_id__in=movie_ids
        ).annotate(
            distance=CosineDistance('embedding', query_embedding)
        )
        
        results = list(queryset.order_by('distance')[:k*2])
        
        # Priorytetyzuj sekcje themes, characters, visual_technical
        for section in results:
            if section.section_type in ['themes', 'characters', 'visual_technical']:
                section.weighted_score = (1.0 - section.distance) * 1.8
            else:
                section.weighted_score = (1.0 - section.distance)
        
        reranked = sorted(results, key=lambda x: x.weighted_score, reverse=True)[:k]
        
        return reranked

    def search_by_genre_or_theme(self, query, k=10):
        """
        Wyszukiwanie po gatunkach i tematach
        """
        from reports.models import MovieSection
        
        query_embedding = self.generate_embedding(query)
        
        # Priorytetyzuj sekcje themes i legacy
        queryset = MovieSection.objects.filter(
            embedding__isnull=False,
            section_type__in=['themes', 'legacy', 'characters', 'plot_structure']
        ).annotate(
            distance=CosineDistance('embedding', query_embedding)
        )
        
        results = list(queryset.order_by('distance')[:k*2])
        
        # Zapewnij różnorodność filmów
        diverse_results = []
        seen_movies = set()
        
        for section in results:
            if section.movie_id not in seen_movies:
                diverse_results.append(section)
                seen_movies.add(section.movie_id)
            
            if len(diverse_results) >= k:
                break
        
        return diverse_results