import requests
from django.conf import settings
from services.rag_service import RAGService
import logging
import re
from services.global_chat_service import GlobalChatService
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        # Use Ollama directly via HTTP API
        self.ollama_url = settings.OLLAMA_BASE_URL.replace('/v1', '')  # Remove /v1 suffix
        self.model = settings.OLLAMA_MODEL

        self.rag = RAGService()
        self.global_chat = GlobalChatService()
        
    def chat(self, user_message, movie_id=None):
        import time
        start_time = time.time()

        if not movie_id:
            return self.global_chat.chat(user_message)

        # RAG search with timing
        rag_start = time.time()
        if movie_id:
            results = self.rag.search_with_scores(user_message, k=3, movie_id=movie_id)
        else:
            results = self.rag.search_with_scores(user_message, k=5, movie_id=None)
        rag_time = time.time() - rag_start
        logger.info(f"RAG search took: {rag_time:.2f}s (k={3 if movie_id else 5})")

        sections = [r['section'] for r in results]

        context_parts = []
        for s in sections:
            content_length = self._get_context_length(s.section_type, movie_id)

            context_parts.append(
                f"[{s.movie.title} - {s.get_section_type_display()}]\n"
                f"{s.content[:content_length]}"
            )

        context = "\n\n---\n\n".join(context_parts)
        
        if movie_id:
            from movies.models import Movie
            movie = Movie.objects.get(id=movie_id)
            system_prompt = f"""You are a knowledgeable movie assistant discussing "{movie.title}" ({movie.year}).

Context from the movie analysis:
{context}

CRITICAL RULES:
1. Answer ONLY based on the context provided above
2. If the question is not related to this movie or cannot be answered from the context, politely say: "I can only answer questions about {movie.title} based on the movie analysis. Please ask something about the film."
3. NEVER use your general knowledge - only use the context
4. Be conversational and concise (3-5 sentences)
5. If context doesn't fully answer the question, say what you know and that you don't have more information

Answer the user's question based STRICTLY on this context."""
        else:
            system_prompt = f"""You are a knowledgeable movie expert assistant.

Context from movie analyses:
{context}

CRITICAL RULES:
1. Answer ONLY based on the context provided above
2. If the question is not about movies or cannot be answered from the context, politely say: "I can only answer questions about movies based on our movie database. Please ask something about films."
3. NEVER use your general knowledge - only use the context
4. Be conversational and concise (3-5 sentences)
5. Mention movie titles when relevant
6. If context doesn't fully answer the question, say what you know and that you don't have more information

Answer based STRICTLY on this context."""
        
        try:
            llm_start = time.time()

            # Call Ollama API directly
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 250
                    }
                }
            )
            response.raise_for_status()
            result = response.json()

            llm_time = time.time() - llm_start
            logger.info(f"LLM API call took: {llm_time:.2f}s")

            answer = result['message']['content'].strip()

            answer = re.sub(r'<[｜|][^>]*[｜|]>', '', answer)
            answer = re.sub(r'</?s>', '', answer)
            answer = re.sub(r'<</?SYS>>', '', answer)
            answer = answer.strip()

            sentences = answer.split('. ')
            if len(sentences) > 6:
                answer = '. '.join(sentences[:6]) + '.'

            total_time = time.time() - start_time
            logger.info(f"Total movie chat response time: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s)")

            return {
                'message': answer,
                'sources': results
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                'message': "Sorry, I encountered an error. Please try again.",
                'sources': []
            }
    
    def _get_context_length(self, section_type, movie_id):
        """
        Determine how much content to include based on section type and chat mode
        """
        high_priority = ['plot_structure', 'characters', 'themes']
        medium_priority = ['visual_technical', 'production', 'cast_crew']
        low_priority = ['reception', 'legacy']
        
        if movie_id:
            if section_type in high_priority:
                return 1200
            elif section_type in medium_priority:
                return 900
            else:
                return 600
        else:
            if section_type in high_priority:
                return 800
            elif section_type in medium_priority:
                return 600
            else:
                return 400
    
    def process_message(self, message, movie_id=None, conversation_id=None):
        """
        Process message and return result (for API compatibility)
        """
        result = self.chat(message, movie_id)

        sources = []
        for r in result['sources']:
            sources.append({
                'section_id': r['section'].id,
                'similarity': r['similarity'],
                'movie_title': r['section'].movie.title,
                'section_type': r['section'].get_section_type_display()
            })

        return {
            'message': result['message'],
            'sources': sources
        }

    def chat_stream(self, user_message, movie_id=None):
        """
        Streaming version of chat - yields Server-Sent Events
        Provides better perceived performance by streaming LLM response
        """
        import time
        import json

        start_time = time.time()

        try:
            # If no movie_id, delegate to global chat streaming
            if not movie_id:
                yield from self.global_chat.chat_stream(user_message)
                return

            # Step 1: RAG search (send progress event)
            yield f"data: {json.dumps({'type': 'rag_start'})}\n\n"

            rag_start = time.time()
            results = self.rag.search_with_scores(user_message, k=3, movie_id=movie_id)
            rag_time = time.time() - rag_start

            logger.info(f"RAG search took: {rag_time:.2f}s (k=3)")

            # Send sources event
            sources_data = [{
                'section_id': r['section_id'],
                'similarity': r['similarity'],
                'movie_title': r['movie_title'],
                'section_type': r['section_type']
            } for r in results]

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

            # Step 2: Build context
            sections = [r['section'] for r in results]
            context_parts = []

            for s in sections:
                content_length = self._get_context_length(s.section_type, movie_id)
                context_parts.append(
                    f"[{s.movie.title} - {s.get_section_type_display()}]\n"
                    f"{s.content[:content_length]}"
                )

            context = "\n\n---\n\n".join(context_parts)

            # Step 3: Build system prompt
            from movies.models import Movie
            movie = Movie.objects.get(id=movie_id)
            system_prompt = f"""You are a knowledgeable movie assistant discussing "{movie.title}" ({movie.year}).

Context from the movie analysis:
{context}

CRITICAL RULES:
1. Answer ONLY based on the context provided above
2. If the question is not related to this movie or cannot be answered from the context, politely say: "I can only answer questions about {movie.title} based on the movie analysis. Please ask something about the film."
3. NEVER use your general knowledge - only use the context
4. Be conversational and concise (3-5 sentences)
5. If context doesn't fully answer the question, say what you know and that you don't have more information

Answer the user's question based STRICTLY on this context."""

            # Step 4: Stream LLM response
            yield f"data: {json.dumps({'type': 'llm_start'})}\n\n"

            llm_start = time.time()
            full_response = ""

            try:
                # Call Ollama streaming API
                stream_response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 250
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
                fallback = "Sorry, I encountered an error. Please try again."
                yield f"data: {json.dumps({'type': 'content', 'content': fallback})}\n\n"
                full_response = fallback
                llm_time = time.time() - llm_start

            # Clean up response
            full_response = re.sub(r'<[｜|][^>]*[｜|]>', '', full_response)
            full_response = re.sub(r'</?s>', '', full_response)
            full_response = re.sub(r'<</?SYS>>', '', full_response)
            full_response = full_response.strip()

            # Limit response length
            sentences = full_response.split('. ')
            if len(sentences) > 6:
                full_response = '. '.join(sentences[:6]) + '.'

            total_time = time.time() - start_time

            # Send completion event with metadata
            metadata = {
                'type': 'done',
                'message': full_response,
                'metrics': {
                    'rag_time': rag_time,
                    'llm_time': llm_time,
                    'total_time': total_time
                }
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            logger.info(f"Total movie chat streaming time: {total_time:.2f}s (RAG: {rag_time:.2f}s, LLM: {llm_time:.2f}s)")

        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            error_data = {
                'type': 'error',
                'message': 'Sorry, I encountered an error. Please try again.'
            }
            yield f"data: {json.dumps(error_data)}\n\n"