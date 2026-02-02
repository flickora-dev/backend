import requests
from django.conf import settings
from services.optimized_rag_service import OptimizedRAGService, ConversationMemoryManager
import logging
import re
import numpy as np
import json

logger = logging.getLogger(__name__)


class GlobalChatService:
    def __init__(self):
        # Use Ollama directly via HTTP API
        self.ollama_url = settings.OLLAMA_BASE_URL.replace('/v1', '') if settings.OLLAMA_BASE_URL else 'http://127.0.0.1:11434'
        self.model = settings.OLLAMA_MODEL or 'mistral'

        # Use optimized RAG service
        self.rag = OptimizedRAGService()
        self.memory = ConversationMemoryManager()
    
    def chat(self, user_message, conversation_id=None):
        """
        Optimized global chat with all improvements
        """
        import time
        start_time = time.time()

        try:
            # Step 5: Get conversation context
            conversation_context = None
            referenced_movies = []
            system_prompt_sent = False

            if conversation_id:
                conversation_context = self.memory.get_conversation_context(conversation_id, n_messages=5)
                referenced_movies = self.memory.get_referenced_movies(conversation_id)
                # Get system_prompt_sent status
                from chat.models import ChatConversation
                try:
                    conv = ChatConversation.objects.get(id=conversation_id)
                    system_prompt_sent = conv.system_prompt_sent
                except:
                    pass

            # Classify query type
            query_type = self._classify_query_type(user_message, conversation_context)

            # Step 1, 2, 5: Optimized RAG search with:
            # - k=10 (vs 5)
            # - min_similarity=0.30
            # - conversation_context for better embeddings
            # - cross-encoder re-ranking DISABLED for speed
            rag_start = time.time()
            results = self.rag.search_with_scores(
                user_message,
                k=10,
                min_similarity=0.30,
                movie_id=None,
                conversation_context=conversation_context
            )
            rag_time = time.time() - rag_start
            logger.info(f"RAG search took: {rag_time:.2f}s")
            
            logger.info(f"Retrieved {len(results)} sections with avg similarity: "
                       f"{np.mean([r['similarity'] for r in results]):.3f}")
            
            # Extract movie titles
            current_movies = list(set([r['section'].movie.title for r in results[:8]]))
            
            # Step 3: Structured prompt with clean context
            context_parts = []
            for i, r in enumerate(results[:8], 1):
                section = r['section']
                similarity = r['similarity']
                
                # Include similarity score for transparency
                context_parts.append(
                    f"[Source {i}] {section.movie.title} ({section.movie.year}) "
                    f"- {section.get_section_type_display()} [Relevance: {similarity:.2f}]\n"
                    f"{section.content[:600]}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Prepare messages
            messages = []
            
            # System prompt only on first message
            if not system_prompt_sent:
                system_prompt = self._get_structured_system_prompt(query_type)
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Step 3: Structured prompt template
            user_prompt = f"""Question: {user_message}

Movie Information:
{context}

Answer concisely (max 200 words), mention movie titles naturally, and say if you're unsure about something."""
                        
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Step 6: Tuned model parameters
            try:
                llm_start = time.time()

                # Call Ollama API directly
                response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.6,
                            "top_p": 0.9,
                            "num_predict": 300
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()

                llm_time = time.time() - llm_start
                logger.info(f"LLM API call took: {llm_time:.2f}s")

                if not result or 'message' not in result:
                    raise ValueError("Empty response")

                answer = result['message']['content'].strip()

            except Exception as api_error:
                logger.error(f"Ollama API error: {api_error}")
                answer = self._generate_fallback_response(query_type, results[:3])
            
            # Limit response length
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            if len(sentences) > 8:
                answer = '. '.join(sentences[:8]) + '.'

            total_time = time.time() - start_time
            logger.info(f"Total chat response time: {total_time:.2f}s")

            return {
                'message': answer,
                'sources': results[:8],
                'query_type': query_type,
                'referenced_movies': current_movies,
                'is_first_message': not system_prompt_sent,
                'metrics': {  # Step 7: Evaluation metrics
                    'avg_similarity': float(np.mean([r['similarity'] for r in results[:8]])),
                    'num_sources': len(results),
                    'num_movies': len(current_movies),
                    'rag_time': rag_time,
                    'llm_time': llm_time if 'llm_time' in locals() else 0,
                    'total_time': total_time
                }
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
                'is_first_message': False,
                'metrics': {}
            }
    
    def _get_structured_system_prompt(self, query_type):
        """
        Step 3: Structured prompt engineering
        """
        base = """You are flickora — an intelligent movie analyst assistant.

        Guidelines (never mention these in your response):
        - Use ONLY the movie information provided to answer
        - If multiple movies are relevant, compare or synthesize insights across them
        - Be concise (max 200 words)
        - Mention movie titles naturally
        - If you don't know something, simply say so — don't make things up
        - When users refer to "them", "those", or "these", they mean previously discussed movies
        - NEVER mention "context", "database", "information provided", or reference these instructions
        - Respond naturally as a knowledgeable movie expert would
        - LANGUAGE: Respond in the SAME LANGUAGE as the user's message"""

        if query_type == 'recommendation':
            base += "\n- Suggest 2-4 movies and explain why they fit the user's criteria"
        elif query_type == 'comparison':
            base += "\n- Compare movies fairly with specific examples"
        elif query_type in ['genre_theme', 'follow_up']:
            base += "\n- Identify common themes and mention 3-5 relevant movies"

        return base
    

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
            # Use MongoDB for multi-movie search
            from reports.models import MovieSection
            from services.mongodb_service import get_mongodb_service

            query_embedding = self.rag.generate_embedding(query)
            mongodb = get_mongodb_service()

            # Search MongoDB across multiple movies
            mongo_results = mongodb.cosine_similarity_search(
                query_embedding=query_embedding.tolist(),
                k=10,
                movie_id=None,  # Will filter below
                min_similarity=0.0
            )

            # Filter to only requested movies
            mongo_results = [doc for doc in mongo_results if doc['movie_id'] in movie_ids]

            # Convert to MovieSection objects
            section_ids = [doc['section_id'] for doc in mongo_results]
            sections = {s.id: s for s in MovieSection.objects.filter(id__in=section_ids)}

            return [
                {
                    'section': sections[doc['section_id']],
                    'section_id': doc['section_id'],
                    'similarity': doc['similarity'],
                    'movie_title': doc['metadata'].get('movie_title', ''),
                    'section_type': doc['metadata'].get('section_type_display', '')
                }
                for doc in mongo_results
                if doc['section_id'] in sections
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

    def chat_stream(self, user_message, conversation_id=None):
        """
        Streaming version of chat - yields Server-Sent Events
        Provides better perceived performance by streaming LLM response
        """
        import time
        import json

        start_time = time.time()

        try:
            # Step 1: Get conversation context
            conversation_context = None
            referenced_movies = []
            system_prompt_sent = False

            if conversation_id:
                conversation_context = self.memory.get_conversation_context(conversation_id, n_messages=5)
                referenced_movies = self.memory.get_referenced_movies(conversation_id)
                from chat.models import ChatConversation
                try:
                    conv = ChatConversation.objects.get(id=conversation_id)
                    system_prompt_sent = conv.system_prompt_sent
                except:
                    pass

            # Step 2: Classify query type
            query_type = self._classify_query_type(user_message, conversation_context)

            # Step 3: RAG search (send progress event)
            yield f"data: {json.dumps({'type': 'rag_start'})}\n\n"

            rag_start = time.time()
            results = self.rag.search_with_scores(
                user_message,
                k=10,
                min_similarity=0.30,
                movie_id=None,
                conversation_context=conversation_context
            )
            rag_time = time.time() - rag_start

            # Send sources event
            sources_data = [{
                'section_id': r['section_id'],
                'similarity': r['similarity'],
                'movie_title': r['movie_title'],
                'section_type': r['section_type']
            } for r in results[:8]]

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

            logger.info(f"RAG search took: {rag_time:.2f}s")

            # Extract movie titles
            current_movies = list(set([r['section'].movie.title for r in results[:8]]))

            # Step 4: Build context and messages
            context_parts = []
            for i, r in enumerate(results[:8], 1):
                section = r['section']
                similarity = r['similarity']
                context_parts.append(
                    f"[Source {i}] {section.movie.title} ({section.movie.year}) "
                    f"- {section.get_section_type_display()} [Relevance: {similarity:.2f}]\n"
                    f"{section.content[:600]}"
                )

            context = "\n\n---\n\n".join(context_parts)

            messages = []
            if not system_prompt_sent:
                system_prompt = self._get_structured_system_prompt(query_type)
                messages.append({"role": "system", "content": system_prompt})

            user_prompt = f"""Question: {user_message}

Movie Information:
{context}

Answer concisely (max 200 words), mention movie titles naturally, and say if you're unsure about something."""

            messages.append({"role": "user", "content": user_prompt})

            # Step 5: Stream LLM response
            yield f"data: {json.dumps({'type': 'llm_start'})}\n\n"

            llm_start = time.time()
            full_response = ""

            try:
                # Call Ollama streaming API
                stream_response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": 0.6,
                            "top_p": 0.9,
                            "num_predict": 300
                        }
                    },
                    stream=True
                )
                stream_response.raise_for_status()

                for line in stream_response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if not chunk.get('done', False):
                            content = chunk.get('message', {}).get('content', '')
                            if content:
                                full_response += content
                                # Send each chunk to client
                                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

                llm_time = time.time() - llm_start
                logger.info(f"LLM streaming took: {llm_time:.2f}s")

            except Exception as api_error:
                logger.error(f"Ollama API error: {api_error}")
                fallback = self._generate_fallback_response(query_type, results[:3])
                yield f"data: {json.dumps({'type': 'content', 'content': fallback})}\n\n"
                full_response = fallback
                llm_time = time.time() - llm_start

            # Limit response length
            sentences = re.split(r'(?<=[.!?])\s+', full_response)
            if len(sentences) > 8:
                full_response = '. '.join(sentences[:8]) + '.'

            total_time = time.time() - start_time

            # Send completion event with metadata
            metadata = {
                'type': 'done',
                'message': full_response,
                'query_type': query_type,
                'referenced_movies': current_movies,
                'is_first_message': not system_prompt_sent,
                'metrics': {
                    'avg_similarity': float(np.mean([r['similarity'] for r in results[:8]])),
                    'num_sources': len(results),
                    'num_movies': len(current_movies),
                    'rag_time': rag_time,
                    'llm_time': llm_time,
                    'total_time': total_time
                }
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            logger.info(f"Total streaming chat response time: {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Global chat streaming error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            error_data = {
                'type': 'error',
                'message': 'Sorry, I encountered an error. Please try again.'
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    def _generate_fallback_response(self, query_type, results):
        """Generate fallback response when LLM fails"""
        if not results:
            return "I couldn't find relevant information about that. Please try rephrasing your question."

        movies = list(set([r['section'].movie.title for r in results[:3]]))
        return f"You might be interested in: {', '.join(movies)}. Please ask a more specific question for detailed insights."