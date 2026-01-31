from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST
from django.utils import timezone
import json
from .models import ChatConversation, ChatMessage
from services.chat_service import ChatService
from django.views.decorators.csrf import csrf_exempt
import logging

logger = logging.getLogger(__name__) 

@csrf_exempt
@require_POST
def chat_message(request):
    try:
        data = json.loads(request.body)
        message = data.get('message')
        movie_id = data.get('movie_id')
        conversation_id = data.get('conversation_id')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Pobierz lub utwórz konwersację
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(id=conversation_id)
            except ChatConversation.DoesNotExist:
                conversation = None
        else:
            conversation = None
        
        if not conversation:
            conversation = ChatConversation.objects.create(
                conversation_type='movie' if movie_id else 'global',
                movie_id=movie_id,
                system_prompt_sent=False  # DODAJ
            )
        
        # Zapisz wiadomość użytkownika
        ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=message
        )

        # Użyj movie_id z requestu lub z istniejącej konwersacji
        effective_movie_id = movie_id or (conversation.movie_id if conversation.conversation_type == 'movie' else None)

        # Pobierz odpowiedź AI
        chat_service = ChatService()
        if effective_movie_id:
            result = chat_service.chat(message, effective_movie_id)
        else:
            result = chat_service.global_chat.chat(message, conversation_id=conversation.id)

        # WAŻNE: Oznacz że system prompt został wysłany
        if result.get('is_first_message', False):
            conversation.system_prompt_sent = True

        # Zaktualizuj referenced_movies
        if 'referenced_movies' in result and result['referenced_movies']:
            current_movies = conversation.referenced_movies or []
            new_movies = result['referenced_movies']
            conversation.referenced_movies = (current_movies + new_movies)[-10:]

        # Zapisz odpowiedź asystenta
        ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=result['message']
        )

        # Zawsze zapisz konwersację na końcu żeby updated_at się zaktualizował
        conversation.updated_at = timezone.now()
        conversation.save(update_fields=['updated_at', 'system_prompt_sent', 'referenced_movies'])
        
        # Przygotuj odpowiedź
        serialized_sources = [
            {
                'section_id': source['section_id'],
                'similarity': source['similarity'],
                'movie_title': source['movie_title'],
                'section_type': source['section_type']
            }
            for source in result['sources']
        ]
        
        return JsonResponse({
            'message': result['message'],
            'sources': serialized_sources,
            'conversation_id': conversation.id
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_POST
def chat_message_stream(request):
    """
    Streaming endpoint for chat messages
    Returns Server-Sent Events (SSE) for real-time response streaming
    """
    try:
        data = json.loads(request.body)
        message = data.get('message')
        movie_id = data.get('movie_id')
        conversation_id = data.get('conversation_id')

        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)

        # Get or create conversation
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(id=conversation_id)
            except ChatConversation.DoesNotExist:
                conversation = None
        else:
            conversation = None

        if not conversation:
            conversation = ChatConversation.objects.create(
                conversation_type='movie' if movie_id else 'global',
                movie_id=movie_id,
                system_prompt_sent=False
            )

        # Save user message
        ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=message
        )

        # Create streaming generator
        def event_stream():
            chat_service = ChatService()
            full_message = ""
            sources_data = []
            metadata = {}

            try:
                # Stream chat response
                if movie_id:
                    stream_gen = chat_service.chat_stream(message, movie_id=movie_id)
                else:
                    stream_gen = chat_service.global_chat.chat_stream(
                        message,
                        conversation_id=conversation.id
                    )

                for event in stream_gen:
                    # Forward SSE event to client
                    yield event

                    # Parse event to extract data for database storage
                    if event.startswith('data: '):
                        try:
                            event_data = json.loads(event[6:])
                            event_type = event_data.get('type')

                            if event_type == 'sources':
                                sources_data = event_data.get('sources', [])
                            elif event_type == 'content':
                                full_message += event_data.get('content', '')
                            elif event_type == 'done':
                                full_message = event_data.get('message', full_message)
                                metadata = event_data
                        except json.JSONDecodeError:
                            pass

                # After streaming completes, save to database
                if full_message:
                    # Save assistant message
                    ChatMessage.objects.create(
                        conversation=conversation,
                        role='assistant',
                        content=full_message
                    )

                    # Update conversation metadata
                    if metadata.get('is_first_message', False):
                        conversation.system_prompt_sent = True

                    if metadata.get('referenced_movies'):
                        current_movies = conversation.referenced_movies or []
                        new_movies = metadata['referenced_movies']
                        conversation.referenced_movies = (current_movies + new_movies)[-10:]

                # Zawsze zapisz konwersację żeby updated_at się zaktualizował
                from django.utils import timezone
                conversation.updated_at = timezone.now()
                conversation.save(update_fields=['updated_at', 'system_prompt_sent', 'referenced_movies'])

                # Send final metadata event with conversation_id
                final_event = {
                    'type': 'metadata',
                    'conversation_id': conversation.id,
                    'sources': sources_data
                }
                yield f"data: {json.dumps(final_event)}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                error_event = {
                    'type': 'error',
                    'message': 'Sorry, an error occurred during streaming.'
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        # Return StreamingHttpResponse with SSE headers
        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
        return response

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)