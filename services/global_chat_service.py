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
        self.model = "google/gemma-3-4b-it:free"
        self.rag = RAGService()
    
    def chat(self, user_message, conversation_id=None):
        """
        Global chat z pamięcią konwersacji
        """
        try:
            # Pobierz historię konwersacji
            conversation_history = []
            referenced_movies = []
            system_prompt_sent = False
            
            if conversation_id:
                conversation_history, referenced_movies, system_prompt_sent = self._get_conversation_context(conversation_id)
            
            # Klasyfikuj zapytanie
            query_type = self._classify_query_type(user_message, conversation_history)
            
            # Wybierz strategię wyszukiwania
            if query_type == 'follow_up' and referenced_movies:
                results = self._handle_follow_up(user_message, referenced_movies)
            elif query_type == 'recommendation':
                results = self._handle_recommendation(user_message)
            elif query_type == 'comparison':
                results = self._handle_comparison(user_message)
            elif query_type == 'genre_theme':
                results = self._handle_genre_theme(user_message)
            else:
                results = self.rag.search_with_scores(user_message, k=8, movie_id=None)
            
            # Wyodrębnij tytuły filmów
            current_movies = list(set([r['section'].movie.title for r in results[:8]]))
            
            # Przygotuj kontekst z wyników wyszukiwania
            context_parts = []
            for r in results[:6]:  # Ogranicz do 6 najlepszych
                section = r['section']
                context_parts.append(
                    f"[{section.movie.title} ({section.movie.year}) - {section.get_section_type_display()}]\n"
                    f"{section.content[:600]}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Przygotuj wiadomości dla LLM
            messages = []
            
            # KLUCZOWE: System prompt tylko przy pierwszej wiadomości
            if not system_prompt_sent:
                system_prompt = self._get_initial_system_prompt(query_type)
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Dodaj historię konwersacji (ostatnie 6 wiadomości = 3 pary)
            if conversation_history:
                for msg in conversation_history[-6:]:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Dodaj kontekst + aktualne pytanie
            user_message_with_context = f"""Context from movie database:
    {context}

    Question: {user_message}"""
            
            messages.append({
                "role": "user",
                "content": user_message_with_context
            })
            
            # Wywołaj LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=400,
                    temperature=0.7
                )
                
                if not response or not response.choices:
                    raise ValueError("Empty response from OpenRouter")
                
                answer = response.choices[0].message.content.strip()
                
                if not answer:
                    raise ValueError("Empty content in response")
                
            except Exception as api_error:
                logger.error(f"OpenRouter API error: {api_error}")
                answer = self._generate_fallback_response(query_type, results[:3])
            
            # Ogranicz długość odpowiedzi
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            if len(sentences) > 7:
                answer = '. '.join(sentences[:7]) + '.'
            
            return {
                'message': answer,
                'sources': results[:8],
                'query_type': query_type,
                'referenced_movies': current_movies,
                'is_first_message': not system_prompt_sent  # Informacja dla widoku
            }
            
        except Exception as e:
            logger.error(f"Global chat error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'message': "Sorry, I encountered an error. Please try again.",
                'sources': [],
                'query_type': 'error',
                'referenced_movies': [],
                'is_first_message': False
            }

    def _get_conversation_context(self, conversation_id):
        """
        Pobierz historię konwersacji, referenced movies i status system prompt
        """
        from chat.models import ChatConversation, ChatMessage
        
        try:
            conversation = ChatConversation.objects.get(id=conversation_id)
            messages = conversation.messages.all().order_by('created_at')
            
            history = [
                {'role': msg.role, 'content': msg.content}
                for msg in messages
            ]
            
            referenced_movies = conversation.referenced_movies or []
            system_prompt_sent = conversation.system_prompt_sent
            
            return history, referenced_movies, system_prompt_sent
        except:
            return [], [], False

    def _is_follow_up_question(self, message):
        """
        Sprawdź czy pytanie odnosi się do poprzedniego kontekstu
        """
        message_lower = message.lower()
        
        follow_up_indicators = [
            'them', 'those', 'these', 'they', 'their',
            'what about', 'tell me more', 'why',
            'how about', 'and', 'also'
        ]
        
        return any(indicator in message_lower for indicator in follow_up_indicators)

    def _handle_follow_up(self, query, referenced_movies):
        """
        Obsłuż pytanie nawiązujące do poprzednich filmów
        """
        from movies.models import Movie
        
        # Znajdź filmy po tytułach
        movies = Movie.objects.filter(title__in=referenced_movies)
        movie_ids = [m.id for m in movies]
        
        if movie_ids:
            # Wyszukaj w kontekście tych filmów
            from reports.models import MovieSection
            
            query_embedding = self.rag.generate_embedding(query)
            
            queryset = MovieSection.objects.filter(
                embedding__isnull=False,
                movie_id__in=movie_ids
            ).annotate(
                distance=CosineDistance('embedding', query_embedding)
            )
            
            results = list(queryset.order_by('distance')[:10])
            
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
        else:
            # Fallback do standardowego wyszukiwania
            return self.rag.search_with_scores(query, k=8, movie_id=None)
        
    def _classify_query_type(self, query, conversation_history=None):
        """
        Klasyfikuj typ zapytania z uwzględnieniem kontekstu konwersacji
        """
        query_lower = query.lower()
        
        # Jeśli jest historia i pytanie odnosi się do poprzedniego kontekstu
        if conversation_history and self._is_follow_up_question(query):
            return 'follow_up'
        
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
        
        
    def _get_initial_system_prompt(self, query_type):
        """
        System prompt wysyłany tylko przy pierwszej wiadomości
        """
        base_prompt = """You are an expert movie assistant with access to a comprehensive movie database.

    YOUR CORE RULES:
    1. Answer ONLY based on the context provided in each message
    2. ALWAYS mention specific movie titles when discussing films
    3. Be conversational and concise (4-6 sentences per response)
    4. If the question is not about movies, politely redirect: "I can only discuss movies from our database"
    5. When users refer to "them", "those", or "these movies", they mean the films just discussed
    6. Never use information outside the provided context

    YOUR EXPERTISE:
    - Movie recommendations based on themes, genres, and preferences
    - Comparing films across different aspects
    - Analyzing themes, characters, and storytelling
    - Discussing cinematography, direction, and technical elements"""
        
        if query_type == 'recommendation':
            base_prompt += """

    RECOMMENDATION FOCUS:
    - Suggest 2-4 specific movies from the context
    - Explain WHY each recommendation fits
    - Highlight key themes, genres, or unique elements
    - Be enthusiastic but honest about each film"""
        
        elif query_type == 'comparison':
            base_prompt += """

    COMPARISON FOCUS:
    - Compare specific aspects from the context
    - Highlight both similarities AND differences
    - Be balanced and fair to all films
    - Use concrete examples from the analyses"""
        
        elif query_type in ['genre_theme', 'follow_up']:
            base_prompt += """

    THEMATIC FOCUS:
    - Identify common themes across multiple films
    - Mention 3-5 relevant movies from the context
    - Explain how each explores the theme/genre
    - Provide specific narrative or stylistic examples"""
        
        return base_prompt
    
    
    def _get_system_prompt(self, query_type, context, has_history=False):
        """
        Zwróć system prompt z uwzględnieniem historii
        """
        history_instruction = ""
        if has_history:
            history_instruction = """
        CONVERSATION CONTEXT:
        - You have access to the recent conversation history above
        - When user refers to "them", "those", "these", they mean the movies just discussed
        - Maintain context and build upon previous answers
        - Be consistent with what you said before
        """
            
            base_rules = f"""
        CRITICAL RULES:
        1. Answer ONLY based on the context provided above
        2. If the question is not about movies, politely say: "I can only answer questions about movies based on our database."
        3. NEVER use your general knowledge - only use the context
        4. Be conversational and concise (4-6 sentences)
        5. ALWAYS mention specific movie titles when relevant
        6. If context doesn't fully answer the question, say what you know
        {history_instruction}
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
            
            elif query_type == 'genre_theme' or query_type == 'follow_up':
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
    
    def _handle_recommendation(self, query):
        return self.rag.search_for_recommendations(query, k=10)

    def _handle_comparison(self, query):
        """
        Obsługa zapytań porównawczych
        """
        from movies.models import Movie
        
        movies = Movie.objects.all()[:100]
        movie_titles = []
        
        for movie in movies:
            if movie.title.lower() in query.lower():
                movie_titles.append(movie.title)
        
        if len(movie_titles) >= 2:
            return self.rag.search_for_comparison(query, movie_titles, k=8)
        else:
            return self.rag.search_with_scores(query, k=8, movie_id=None)

    def _handle_genre_theme(self, query):
        """
        Obsługa zapytań o gatunki i tematy
        """
        return self.rag.search_by_genre_or_theme(query, k=10)