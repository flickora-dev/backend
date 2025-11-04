import openai
from django.conf import settings
from services.rag_service import RAGService
import logging
import re
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        )
        # self.model = "deepseek/deepseek-chat-v3.1:free"
        self.model = "meta-llama/llama-3.3-8b-instruct:free"

        self.rag = RAGService()
    
    def chat(self, user_message, movie_id=None):
        """
        Enhanced chat with better context retrieval
        """
        if not movie_id:
            return self.global_chat.chat(user_message)
        
        if movie_id:
            results = self.rag.search_with_scores(user_message, k=3, movie_id=movie_id)
        else:
            results = self.rag.search_with_scores(user_message, k=5, movie_id=None)
        
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=250,
                temperature=0.7,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            answer = re.sub(r'<[｜|][^>]*[｜|]>', '', answer)
            answer = re.sub(r'</?s>', '', answer)
            answer = re.sub(r'<</?SYS>>', '', answer)
            answer = answer.strip()
            
            sentences = answer.split('. ')
            if len(sentences) > 6:
                answer = '. '.join(sentences[:6]) + '.'
            
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