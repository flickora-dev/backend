import openai
from django.conf import settings
from services.rag_service import RAGService
import logging
import re

logger = logging.getLogger(__name__)


class GlobalChatService:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        )
        self.model = "meta-llama/llama-3.3-8b-instruct:free"
        self.rag = RAGService()
    
    def chat(self, user_message):
        """
        Global chat - odpowiedzi o filmach z całej bazy
        """
        try:
            query_type = self._classify_query_type(user_message)
            
            # Wybierz odpowiednią strategię wyszukiwania
            if query_type == 'recommendation':
                results = self._handle_recommendation(user_message)
            elif query_type == 'comparison':
                results = self._handle_comparison(user_message)
            elif query_type == 'genre_theme':
                results = self._handle_genre_theme(user_message)
            else:
                results = self.rag.search_with_scores(user_message, k=8, movie_id=None)
            
            # Przygotuj kontekst
            context_parts = []
            for r in results[:8]:
                section = r.get('section') if isinstance(r, dict) else r
                
                context_parts.append(
                    f"[{section.movie.title} ({section.movie.year}) - {section.get_section_type_display()}]\n"
                    f"{section.content[:700]}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            
            # System prompt dla global chat
            system_prompt = self._get_system_prompt(query_type, context)
            
            # Wywołaj LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Ogranicz długość odpowiedzi
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            if len(sentences) > 7:
                answer = '. '.join(sentences[:7]) + '.'
            
            return {
                'message': answer,
                'sources': self._format_sources(results[:8]),
                'query_type': query_type
            }
            
        except Exception as e:
            logger.error(f"Global chat error: {e}")
            return {
                'message': "Sorry, I encountered an error. Please try again.",
                'sources': [],
                'query_type': 'error'
            }
    
    def _classify_query_type(self, query):
        """
        Klasyfikuj typ zapytania dla global chat
        """
        query_lower = query.lower()
        
        recommendation_keywords = [
            'recommend', 'suggestion', 'should i watch', 'similar to',
            'like', 'what movie', 'looking for', 'want to watch',
            'good movies', 'best movies', 'top movies'
        ]
        if any(kw in query_lower for kw in recommendation_keywords):
            return 'recommendation'
        
        comparison_keywords = [
            'compare', 'versus', 'vs', 'difference between',
            'better than', 'similar', 'both', 'either'
        ]
        if any(kw in query_lower for kw in comparison_keywords):
            return 'comparison'
        
        genre_theme_keywords = [
            'genre', 'theme', 'about', 'exploring', 'dealing with',
            'drama', 'comedy', 'thriller', 'action', 'sci-fi',
            'love', 'war', 'family', 'friendship', 'redemption'
        ]
        if any(kw in query_lower for kw in genre_theme_keywords):
            return 'genre_theme'
        
        return 'general'
    
    def _handle_recommendation(self, query):
        """
        Obsługa zapytań o rekomendacje
        """
        results = self.rag.search_for_recommendations(query, k=10)
        
        return [
            {
                'section': section,
                'section_id': section.id,
                'similarity': 1.0 - section.distance,
                'movie_title': section.movie.title,
                'section_type': section.get_section_type_display()
            }
            for section in results
        ]
    
    def _handle_comparison(self, query):
        """
        Obsługa zapytań porównawczych
        """
        # Spróbuj wyodrębnić tytuły filmów z zapytania
        # To jest uproszczona wersja - można to rozbudować
        from movies.models import Movie
        
        movies = Movie.objects.all()[:100]  # Ogranicz dla wydajności
        movie_titles = []
        
        for movie in movies:
            if movie.title.lower() in query.lower():
                movie_titles.append(movie.title)
        
        if len(movie_titles) >= 2:
            results = self.rag.search_for_comparison(query, movie_titles, k=8)
        else:
            results = self.rag.search_with_scores(query, k=8)
        
        return [
            {
                'section': section,
                'section_id': section.id,
                'similarity': 1.0 - section.distance,
                'movie_title': section.movie.title,
                'section_type': section.get_section_type_display()
            }
            for section in results
        ]
    
    def _handle_genre_theme(self, query):
        """
        Obsługa zapytań o gatunki i tematy
        """
        results = self.rag.search_by_genre_or_theme(query, k=10)
        
        return [
            {
                'section': section,
                'section_id': section.id,
                'similarity': 1.0 - section.distance,
                'movie_title': section.movie.title,
                'section_type': section.get_section_type_display()
            }
            for section in results
        ]
    
    def _get_system_prompt(self, query_type, context):
        """
        Zwróć odpowiedni system prompt dla typu zapytania
        """
        base_rules = """
CRITICAL RULES:
1. Answer ONLY based on the context provided above
2. If the question is not about movies, politely say: "I can only answer questions about movies based on our database."
3. NEVER use your general knowledge - only use the context
4. Be conversational and concise (4-6 sentences)
5. ALWAYS mention specific movie titles when relevant
6. If context doesn't fully answer the question, say what you know
"""
        
        if query_type == 'recommendation':
            return f"""You are a movie recommendation expert.

Context from movie analyses:
{context}

{base_rules}

RECOMMENDATION RULES:
- Suggest 2-4 specific movies from the context
- Explain WHY each movie fits the request
- Mention key themes, genres, or elements that match
- Be enthusiastic but honest

Answer the user's question based STRICTLY on this context."""
        
        elif query_type == 'comparison':
            return f"""You are a movie comparison expert.

Context from movie analyses:
{context}

{base_rules}

COMPARISON RULES:
- Compare specific aspects mentioned in the context
- Highlight similarities AND differences
- Be balanced and fair to all movies
- Use concrete examples from the context

Answer the user's question based STRICTLY on this context."""
        
        elif query_type == 'genre_theme':
            return f"""You are a movie genre and theme expert.

Context from movie analyses:
{context}

{base_rules}

GENRE/THEME RULES:
- Identify common themes across movies
- Mention 3-5 specific movies that fit
- Explain how each movie explores the theme/genre
- Be specific about narrative elements

Answer the user's question based STRICTLY on this context."""
        
        else:
            return f"""You are a knowledgeable movie expert assistant.

Context from movie analyses:
{context}

{base_rules}

Answer the user's question based STRICTLY on this context."""
    
    def _format_sources(self, results):
        """
        Formatuj źródła dla odpowiedzi
        """
        sources = []
        for r in results:
            if isinstance(r, dict):
                sources.append({
                    'section_id': r.get('section_id'),
                    'similarity': r.get('similarity', 0),
                    'movie_title': r.get('movie_title'),
                    'section_type': r.get('section_type')
                })
            else:
                sources.append({
                    'section_id': r.id,
                    'similarity': 1.0 - r.distance,
                    'movie_title': r.movie.title,
                    'section_type': r.get_section_type_display()
                })
        
        return sources